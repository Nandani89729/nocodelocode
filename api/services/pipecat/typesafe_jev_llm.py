"""TypeSafe Jev classifier posing as an LLM service for voicemail detection.

Pipecat's ``VoicemailDetector`` runs a classifier branch whose only contract
with the model is three frames -- ``LLMFullResponseStartFrame``, some
``LLMTextFrame`` s, ``LLMFullResponseEndFrame`` -- and a response containing
the literal word ``CONVERSATION`` or ``VOICEMAIL``. Nothing in that contract
requires text generation, so we satisfy it with a single call to TypeSafe's
System One endpoint and one synthesised text frame.

Two properties of the surrounding code shape this class:

- ``RealtimeFeedbackObserver`` only records TTFB when ``"LLM"`` appears in the
  processor name, so the class name keeps it. Renaming this class without that
  substring silently drops the latency metric from workflow run logs.
- ``ClassificationProcessor._process_classification`` leaves ``_decision_made``
  False when the response matches neither keyword, which keeps the classifier
  gate open for the next transcript. We use that as the low-confidence path:
  below the threshold we emit no verdict at all and let more speech arrive,
  rather than forcing a coin-flip the way a generative classifier must.
"""

from typing import Any, Literal

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

#: TypeSafe's own endpoint. Needs a TypeSafe key, which is waitlisted.
TYPESAFE_BASE_URL = "https://api.typesafe.ai/v1/systemone"

#: OpenRouter's Decisions API. Same request and response shape, reached with an
#: ordinary OpenRouter key, so it is the default. Note this is *not* the
#: OpenAI-compatible chat endpoint and chat SDKs cannot call it.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/alpha/decisions"

DEFAULT_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_MODEL = "typesafe/jev-1.13"

#: Options sent to Jev, and the keyword each one maps to. ``carrier`` is split
#: out from ``voicemail`` so retry logic can eventually tell "mailbox full" and
#: "number not in service" apart from a reachable person's greeting, even
#: though both currently end the call the same way.
#:
#: The human/voicemail line is drawn on who the words are addressed to, not on
#: keywords: Jev reads criteria literally, so a keyword list ("mentions leaving
#: a message") made it call any greeting without those words a person.
CLASSIFICATION_CRITERIA: dict[str, str] = {
    "human": (
        "A person responding live to this call: a greeting, their name, or a "
        "question to the caller, and expecting the caller to answer now."
    ),
    "voicemail": (
        "A greeting recorded in advance for whoever calls. It may give a name, "
        "a business or its hours, say the person is busy, away or unable to "
        "answer, or ask the caller to leave a message or speak after the tone, "
        "even without the word 'voicemail'."
    ),
    "carrier": (
        "The phone network rather than the person or business called: number "
        "not in service, mailbox full or not set up, call cannot be completed."
    ),
}

_KEYWORD_BY_OPTION: dict[str, Literal["CONVERSATION", "VOICEMAIL"]] = {
    "human": "CONVERSATION",
    "voicemail": "VOICEMAIL",
    "carrier": "VOICEMAIL",
}

INSTRUCTIONS = (
    "An outbound call was just answered, and this is what the answering side "
    "has said so far. Is it a person talking live to this caller, a greeting "
    "recorded in advance, or a message from the phone network?"
)

#: Appended when the state carries ``answering_side_timing`` (see
#: ``voicemail_timing.describe_timing``). Timing separates a recorded greeting
#: from a live "Hello?" even when the words alone are ambiguous.
TIMING_INSTRUCTIONS = (
    " answering_side_timing describes when the answering side spoke at the start "
    "of the call. A recorded greeting usually starts after a short delay and "
    "fills the time with one long unbroken stretch of speech. A live person "
    "usually says something short and then goes quiet, waiting for the caller."
)


def build_payload(
    utterances: list[str], *, model: str, timing: dict | None = None
) -> dict:
    """The Decisions API request for one classification.

    Shared with ``scripts/voicemail_eval/replay.py`` so offline replay sends
    exactly what the live service sends.
    """
    state: dict[str, Any] = {"answering_side_said": utterances}
    instructions = INSTRUCTIONS
    if timing is not None:
        state["answering_side_timing"] = timing
        instructions += TIMING_INSTRUCTIONS
    return {
        "state": state,
        "model": model,
        "questions": {
            "answered_by": {
                "type": "choice",
                "instructions": instructions,
                "criteria": CLASSIFICATION_CRITERIA,
            }
        },
    }


class TypeSafeJevLLMService(LLMService):
    """Classifies voicemail vs live conversation with one TypeSafe Jev call.

    Drop-in for the ``llm`` argument of ``VoicemailDetector``. Reads the
    transcript out of the classifier branch's context, sends it to Jev as a
    single Choice question, and emits the matching keyword as one text frame.
    """

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        confidence_threshold: float = 0.75,
        timeout_secs: float = 2.0,
        **kwargs,
    ):
        """Initialize the Jev-backed classifier.

        Args:
            api_key: TypeSafe API key, sent as a bearer token.
            aiohttp_session: Session used for the request. Closed in cleanup(),
                matching how the service factory hands out per-service sessions.
            model: Model id or alias. Pin a version once thresholds are tuned;
                an alias can move under you and shift the confidence scale.
            base_url: Full URL of the System One evaluation endpoint.
            confidence_threshold: Below this, no verdict is emitted and the
                detector waits for more transcript.
            timeout_secs: Per-request timeout. On expiry we emit nothing, which
                leaves the existing detector timeout as the backstop.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = aiohttp_session
        self._model = model
        self._base_url = base_url
        self._confidence_threshold = confidence_threshold
        self._timeout = aiohttp.ClientTimeout(total=timeout_secs)

    async def cleanup(self):
        """Close the session this service owns."""
        await super().cleanup()
        if not self._session.closed:
            await self._session.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Answer context frames with a classification; pass everything else on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        try:
            verdict = await self._classify(frame.context)
            await self.stop_ttfb_metrics()
            if verdict:
                await self.push_frame(LLMTextFrame(verdict))
        except TimeoutError:
            # Emitting nothing keeps the classifier gate open, so the next
            # transcript retries. The detector's own delay is the backstop.
            logger.warning(f"{self}: Jev classification timed out; no verdict emitted")
        except Exception as e:
            logger.error(f"{self}: Jev classification failed: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _classify(self, context: LLMContext) -> str | None:
        """Return ``CONVERSATION``/``VOICEMAIL``, or None when unsure.

        Args:
            context: The classifier branch's context. Only user turns are read;
                the branch's system prompt is ignored because the instructions
                travel with the question instead.
        """
        state = _transcript_from(context)
        if not state:
            return None

        payload = build_payload(state, model=self._model)

        async with self._session.post(
            self._base_url,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            body = await response.json()

        answer = body.get("answers", {}).get("answered_by", {})
        option = answer.get("choice")
        confidence = _confidence_of(answer)

        if option not in _KEYWORD_BY_OPTION:
            logger.warning(f"{self}: Jev returned unknown option {option!r}")
            return None

        if confidence < self._confidence_threshold:
            # The ambiguous opening of a call ("Hello?...") reads the same for a
            # person and a greeting. Hold rather than guess.
            logger.debug(
                f"{self}: holding at confidence {confidence:.2f} "
                f"(option={option}, threshold={self._confidence_threshold})"
            )
            return None

        logger.info(
            f"{self}: Jev classified call as {option} "
            f"(confidence={confidence:.2f}, model={body.get('model')})"
        )
        return _KEYWORD_BY_OPTION[option]


def _confidence_of(answer: dict) -> float:
    """Read the answer's confidence, deriving it from probabilities if absent.

    TypeSafe's native endpoint returns ``confidence`` on every choice answer,
    but OpenRouter's Decisions API documents only ``probabilities``. Since
    confidence is by definition a statistic over that distribution -- and
    TypeSafe's docs invite you to compute your own -- we fall back to the
    top option's probability rather than treat a missing field as zero
    confidence, which would wedge the detector into never deciding.
    """
    confidence = answer.get("confidence")
    if isinstance(confidence, (int, float)):
        return float(confidence)

    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        values = [v for v in probabilities.values() if isinstance(v, (int, float))]
        if values:
            return float(max(values))

    return 0.0


def _transcript_from(context: LLMContext) -> list[str]:
    """Collect the user-side utterances from a classifier context."""
    utterances: list[str] = []
    for message in context.get_messages():
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _text_of(message.get("content"))
        if text:
            utterances.append(text)
    return utterances


def _text_of(content: Any) -> str:
    """Flatten a message's content, which may be a string or a parts list."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return " ".join(p for p in parts if p).strip()
    return ""
