"""Frame-contract tests for the TypeSafe Jev voicemail classifier.

The value of these tests is the contract with Pipecat's VoicemailDetector:
its ClassificationProcessor only sees LLMFullResponseStartFrame, LLMTextFrame
and LLMFullResponseEndFrame, and only acts on the literal words CONVERSATION
and VOICEMAIL. If that shape breaks, voicemail detection silently stops
deciding, so it is pinned here rather than left to an integration test.
"""

from contextlib import asynccontextmanager

import pytest

from api.services.pipecat.typesafe_jev_llm import (
    TypeSafeJevLLMService,
    _text_of,
    _transcript_from,
)
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    async def json(self):
        return self._body


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in recording the request payload."""

    def __init__(self, body=None, exc=None):
        self._body = body
        self._exc = exc
        self.closed = False
        self.payload = None

    @asynccontextmanager
    async def _post(self, url, json=None, headers=None, timeout=None):
        self.payload = json
        if self._exc:
            raise self._exc
        yield _FakeResponse(self._body)

    def post(self, *args, **kwargs):
        return self._post(*args, **kwargs)

    async def close(self):
        self.closed = True


def _answer(choice, confidence):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "answered_by": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
            }
        },
    }


def _answer_without_confidence(choice, probabilities):
    """OpenRouter's Decisions API documents probabilities but not confidence."""
    return {
        "model": "typesafe/jev-1.13",
        "answers": {
            "answered_by": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
            }
        },
    }


def _context(*user_texts):
    messages = [{"role": "developer", "content": "ignored system prompt"}]
    messages += [{"role": "user", "content": t} for t in user_texts]
    return LLMContext(messages)


async def _run(service, context):
    """Drive one context frame through the service, capturing pushed frames."""
    pushed = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    service.push_frame = capture
    await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
    return pushed


def _service(session, **kwargs):
    return TypeSafeJevLLMService(api_key="test-key", aiohttp_session=session, **kwargs)


@pytest.mark.parametrize(
    "choice,expected",
    [
        ("human", "CONVERSATION"),
        ("voicemail", "VOICEMAIL"),
        ("carrier", "VOICEMAIL"),
    ],
)
async def test_choice_maps_to_detector_keyword(choice, expected):
    session = _FakeSession(_answer(choice, 0.95))
    pushed = await _run(_service(session), _context("hello?"))

    assert isinstance(pushed[0], LLMFullResponseStartFrame)
    assert isinstance(pushed[-1], LLMFullResponseEndFrame)

    texts = [f.text for f in pushed if isinstance(f, LLMTextFrame)]
    assert texts == [expected]


async def test_low_confidence_emits_no_verdict():
    """Below threshold the detector must stay undecided and wait for more."""
    session = _FakeSession(_answer("voicemail", 0.4))
    pushed = await _run(
        _service(session, confidence_threshold=0.75), _context("hello?")
    )

    assert isinstance(pushed[0], LLMFullResponseStartFrame)
    assert isinstance(pushed[-1], LLMFullResponseEndFrame)
    assert not [f for f in pushed if isinstance(f, LLMTextFrame)]


async def test_confidence_falls_back_to_top_probability():
    """A missing confidence field must not wedge the detector at zero."""
    body = _answer_without_confidence(
        "voicemail", {"human": 0.08, "voicemail": 0.89, "carrier": 0.03}
    )
    pushed = await _run(
        _service(_FakeSession(body), confidence_threshold=0.75), _context("hi there")
    )
    assert [f.text for f in pushed if isinstance(f, LLMTextFrame)] == ["VOICEMAIL"]


async def test_derived_confidence_still_honours_the_threshold():
    body = _answer_without_confidence(
        "voicemail", {"human": 0.36, "voicemail": 0.38, "carrier": 0.26}
    )
    pushed = await _run(
        _service(_FakeSession(body), confidence_threshold=0.75), _context("hello")
    )
    assert not [f for f in pushed if isinstance(f, LLMTextFrame)]


async def test_defaults_target_the_openrouter_decisions_api():
    session = _FakeSession(_answer("human", 0.9))
    service = _service(session)

    assert service._base_url == "https://openrouter.ai/api/alpha/decisions"
    assert service._model == "typesafe/jev-1.13"


async def test_request_shape_sends_only_user_turns():
    session = _FakeSession(_answer("human", 0.9))
    await _run(_service(session), _context("hi", "who is this?"))

    assert session.payload["model"] == "typesafe/jev-1.13"
    assert session.payload["state"] == {"answering_side_said": ["hi", "who is this?"]}
    question = session.payload["questions"]["answered_by"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"human", "voicemail", "carrier"}


async def test_empty_transcript_makes_no_request():
    session = _FakeSession(_answer("human", 0.9))
    pushed = await _run(_service(session), _context())

    assert session.payload is None
    assert not [f for f in pushed if isinstance(f, LLMTextFrame)]


async def test_api_failure_still_closes_the_response():
    """A failed call must not strand the detector mid-response."""
    session = _FakeSession(exc=RuntimeError("boom"))
    pushed = await _run(_service(session), _context("hello?"))

    assert isinstance(pushed[0], LLMFullResponseStartFrame)
    assert isinstance(pushed[-1], LLMFullResponseEndFrame)
    assert not [f for f in pushed if isinstance(f, LLMTextFrame)]


async def test_unknown_option_is_ignored():
    session = _FakeSession(_answer("fax", 0.99))
    pushed = await _run(_service(session), _context("beep"))

    assert not [f for f in pushed if isinstance(f, LLMTextFrame)]


def test_text_of_handles_content_parts():
    assert _text_of("plain") == "plain"
    assert (
        _text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        == "a b"
    )
    assert _text_of([{"type": "image", "url": "x"}]) == ""
    assert _text_of(None) == ""


def test_transcript_skips_non_user_roles():
    context = LLMContext(
        [
            {"role": "developer", "content": "prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "CONVERSATION"},
            {"role": "user", "content": "  who is this?  "},
        ]
    )
    assert _transcript_from(context) == ["hello", "who is this?"]
