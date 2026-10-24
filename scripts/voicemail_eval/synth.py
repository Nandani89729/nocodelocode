"""Synthesise a labelled voicemail-eval set: scripted openings, TTS, a phone line, ASR.

Twenty acted calls cannot give an accuracy figure with a useful margin, and in
acted calls the callee waits for the agent's opener, which hides the timing
signal a real greeting has. This builds several hundred calls whose audio
starts at answer, the way a real voicemail greeting does:

1. ``scripts``: an LLM writes varied openings per category (voicemail
   greetings, live answers, carrier messages, and the traps from
   ``scenarios.md``).
2. ``audio``: Kokoro speaks each opening with a random voice and pace, placed on
   a 10 s timeline with human-like pauses, beeps or SIT tones. The mix goes
   through an 8 kHz mu-law, 300-3400 Hz channel with noise, and faster-whisper
   transcribes it, so the classifier sees real ASR errors, not script text.

The output ``dataset.jsonl`` has the shape ``harvest.py`` writes, so
``timing.py`` and ``replay.py`` run on it unchanged.

Runs in its own environment, since Kokoro and Whisper are not API deps::

    uv venv .synth && uv pip install --python .synth kokoro-onnx \
        faster-whisper soundfile numpy scipy requests
    # models: kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx
    # releases, into scripts/voicemail_eval/synthetic/models/
    export OPENROUTER_API_KEY=...
    .synth/Scripts/python scripts/voicemail_eval/synth.py scripts --n 400
    .synth/Scripts/python scripts/voicemail_eval/synth.py audio

Keep in mind what this measures: accuracy on audio built from assumptions
about how greetings and people sound. The per-category breakdown and the traps
are there so that assumption does not do all the work, but the number is a
regression baseline and a way to compare classifiers, not a production rate.
"""

import argparse
import io
import json
import os
import random
import sys
import threading
import time
import wave
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "synthetic"
MODELS = OUT / "models"

SCRIPT_MODEL = "openai/gpt-4.1"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

#: category -> (label, share of the set, what the LLM is asked to write).
CATEGORIES: dict[str, tuple[str, float, str]] = {
    "vm_personal": (
        "voicemail",
        0.25,
        "Personal voicemail greetings recorded by the phone's owner. Vary length "
        "(one to three sentences), formality and wording; some give a name, some "
        "don't; some mention the tone or a callback, some don't.",
    ),
    "vm_business": (
        "voicemail",
        0.10,
        "Voicemail greetings for a small business, office, clinic or department "
        "after hours or when nobody picks up.",
    ),
    "vm_default": (
        "voicemail",
        0.07,
        "Default greetings played by a phone carrier or voicemail service when "
        "the owner never recorded one, e.g. 'The person you are calling is not "
        "available. At the tone, please record your message.'",
    ),
    "vm_short": (
        "voicemail",
        0.07,
        "Very short or informal voicemail greetings that avoid the words "
        "'voicemail' and 'message', e.g. 'Hey, it's Dan. You know what to do.'",
    ),
    "carrier": (
        "carrier",
        0.07,
        "Automated network announcements: number not in service, mailbox full, "
        "mailbox not set up, call cannot be completed, number changed.",
    ),
    "hu_short": (
        "human",
        0.24,
        "How a real person answers a call from an unknown number, in one to five "
        "words: 'Hello?', 'Yeah?', 'Priya speaking.', 'Hi, who's this?'.",
    ),
    "hu_long": (
        "human",
        0.12,
        "A real person answering with a full first sentence: a receptionist's "
        "greeting, someone saying they're busy, asking who is calling and why.",
    ),
    "hu_trap": (
        "human",
        0.08,
        "A real person whose first words sound like a voicemail greeting, e.g. "
        "'Hello, you've reached me, go ahead.', 'Can't talk now, call me back in "
        "ten minutes.', or a flat 'This is Mark Davis.'",
    ),
}

#: What a person says after a few seconds of silence from the caller.
FOLLOW_UPS = [
    "Hello?",
    "Hello? Anyone there?",
    "Hi, who is this?",
    "Hello, can you hear me?",
    "Yes? Hello?",
    "Is anybody there?",
]

SR_TTS = 24000
SR_LINE = 8000
SR_OUT = 16000  # Dograh stores user.wav at the pipeline rate
CALL_S = 10.0
TRANSCRIBE_UNTIL_S = 8.0
TRUNCATION_SECONDS = (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------
# Step 1: scripts
# --------------------------------------------------------------------------


def _ask_llm(
    api_key: str, brief: str, k: int, avoid: list[str], model: str = SCRIPT_MODEL
) -> list[str]:
    import requests

    prompt = (
        f"Write {k} distinct examples of the following, as spoken aloud on a "
        f"phone call.\n\n{brief}\n\nUse names, businesses and phrasing from "
        "many English-speaking places (US, UK, India, Nigeria, Philippines, "
        "Australia, Latin America). Vary tone and length. Output only JSON: "
        '{"lines": ["...", ...]}.'
    )
    if avoid:
        prompt += "\n\nDo not repeat these:\n" + "\n".join(avoid[-40:])
    resp = requests.post(
        OPENROUTER_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 1.0,
            # Without a cap OpenRouter reserves the model's whole output window
            # and a limited-credit account gets a 402.
            "max_tokens": 3000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise SystemExit(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
    body = json.loads(resp.json()["choices"][0]["message"]["content"])
    return [str(x).strip() for x in body.get("lines", []) if str(x).strip()]


NAMES = [
    "Sarah", "Dan", "Priya", "Marcus", "Aisha", "Tom", "Chen", "Olu", "Maria",
    "Kevin", "Fatima", "Jake", "Anjali", "Rosa", "Sam", "Nadia", "Chris", "Imani",
    "Arjun", "Ellie", "Joseph", "Mei", "Carlos", "Grace", "Tunde", "Laura",
]
BUSINESSES = [
    "Brightline Accounts", "Greenfield Dental", "Harbor Plumbing", "Summit Realty",
    "Northside Pharmacy", "Oak Street Clinic", "Riverside Auto", "Bluebird Travel",
    "Metro Insurance", "the front desk", "the billing team", "customer support",
]
_VM_OPEN = ["Hi, you've reached {n}.", "Hey, this is {n}.", "Hello, it's {n}.",
            "You've reached the phone of {n}.", "Hi there, {n} here.", "{n} speaking, well, not really."]
_VM_BODY = ["I can't take your call right now.", "I'm away from my phone.",
            "Sorry I missed you.", "I'm not available at the moment.", "I'm driving or in a meeting."]
_VM_CLOSE = ["Please leave a message after the tone.", "Leave your name and number and I'll call you back.",
             "Leave a message and I'll get back to you soon.", "Text me if it's urgent.",
             "I'll return your call as soon as I can.", "Talk soon."]
_BIZ = ["Thank you for calling {b}. Our office is currently closed. Please leave a message and we'll return your call next business day.",
        "You've reached {b}. All of our staff are busy. Please leave your name, number and a brief message.",
        "Hello, you've reached {b}. We're unable to take your call. Leave a message after the tone.",
        "Thanks for calling {b}. Our hours are nine to five, Monday to Friday. Please leave a message."]
_DEFAULT = ["The person you are calling is not available. At the tone, please record your message.",
            "The number you have dialed, {d}, is not available. Please leave a message after the tone.",
            "The subscriber you have called is not available. Please record your message after the tone. When you are finished, hang up.",
            "Your call has been forwarded to an automated voice messaging system. {d} is not available.",
            "Please leave your message for {d}.", "The Google subscriber you have called is not available. Please leave a message after the tone."]
_SHORT = ["Hey, it's {n}. You know what to do.", "{n}. Go.", "It's {n}. Beep.", "{n} here. Say something.",
          "Yep, {n}. Leave it.", "Hi. It's {n}. Speak.", "You got {n}. Do your thing.", "{n}. Talk to me after this."]
_CARRIER = ["The number you have dialed is not in service. Please check the number and try again.",
            "The mailbox belonging to {d} is full and cannot accept any messages at this time. Goodbye.",
            "The person you are trying to reach has a voice mailbox that has not been set up yet. Goodbye.",
            "We're sorry, your call cannot be completed as dialed. Please check the number and dial again.",
            "The number you have reached has been changed, disconnected, or is no longer in service.",
            "All circuits are busy now. Please try your call again later.",
            "The wireless customer you are calling is not available. Please try your call again later."]
_HU_SHORT = ["Hello?", "Hello.", "Yeah?", "Hi.", "Yes?", "Hello, who's this?", "{n} speaking.",
             "This is {n}.", "Hi, who is this?", "Hello, yes?", "Mm, hello?", "Speaking.", "Hey.",
             "Yeah, hello?", "Hello, hello?", "Who's calling?"]
_HU_LONG = ["Good morning, {b}, this is {n}, how can I help you?", "Hi, yeah, this is {n}, sorry I'm just in the car, who's this?",
            "Hello? Sorry, I'm in the middle of something, can you make it quick?",
            "Hi, is this about the appointment? I got a text earlier.", "Hello, {n} speaking, who am I talking to?",
            "Yes, hello, I'm here, sorry, the signal is bad, can you hear me?",
            "Thanks for calling {b}, you're speaking with {n}.", "Hello? I don't recognise this number, who is it?"]
_HU_TRAP = ["Hello, you've reached me, go ahead.", "Can't talk now, call me back in ten minutes.",
            "This is {n}. Leave it, I mean, what's up?", "{n}'s phone, {n} speaking.",
            "Hi, I'm not available right now, I'm kidding, it's {n}, what's up?",
            "Hi, can you call me back later? I'm driving.", "Yeah, I'm busy at the moment, what is this regarding?",
            "{b}, please hold. Okay, sorry, how can I help?"]


def _digits(rng: random.Random) -> str:
    return " ".join(str(rng.randint(0, 9)) for _ in range(7))


def offline_scripts(n: int, seed: int, existing: dict[str, list[str]]) -> dict[str, list[str]]:
    """Template-built openings, for when no LLM credit is available."""
    rng = random.Random(seed)
    fill = lambda t: t.format(n=rng.choice(NAMES), b=rng.choice(BUSINESSES), d=_digits(rng))  # noqa: E731
    makers = {
        "vm_personal": lambda: " ".join([fill(rng.choice(_VM_OPEN)), rng.choice(_VM_BODY), rng.choice(_VM_CLOSE)][: rng.choice([2, 3, 3])]),
        "vm_business": lambda: fill(rng.choice(_BIZ)),
        "vm_default": lambda: fill(rng.choice(_DEFAULT)),
        "vm_short": lambda: fill(rng.choice(_SHORT)),
        "carrier": lambda: fill(rng.choice(_CARRIER)),
        "hu_short": lambda: fill(rng.choice(_HU_SHORT)),
        "hu_long": lambda: fill(rng.choice(_HU_LONG)),
        "hu_trap": lambda: fill(rng.choice(_HU_TRAP)),
    }
    out: dict[str, list[str]] = {}
    for cat, (_, share, _) in CATEGORIES.items():
        want = max(1, round(share * n))
        lines = list(existing.get(cat, []))[:want]
        # Openings may repeat (most people say "Hello?"); voice, pace, noise
        # and timing still differ per call.
        while len(lines) < want:
            lines.append(makers[cat]())
        out[cat] = lines
    return out


def make_scripts(args: argparse.Namespace) -> int:
    if args.offline:
        path = OUT / "scripts.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        out = offline_scripts(args.n, 7, existing)
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        for cat, lines in out.items():
            print(f"  {cat:<12} {len(lines)}")
        print(f"OK wrote {sum(map(len, out.values()))} openings to {path}")
        return 0
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scripts.json"
    # Saved after every category, so a failed run resumes where it stopped.
    out: dict[str, list[str]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    for cat, (_, share, brief) in CATEGORIES.items():
        want = max(1, round(share * args.n))
        lines: list[str] = out.get(cat, [])
        while len(lines) < want:
            batch = _ask_llm(
                api_key, brief, min(30, want - len(lines) + 5), lines, args.model
            )
            seen = {x.lower() for x in lines}
            lines += [x for x in batch if x.lower() not in seen]
            out[cat] = lines[:want]
            path.write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(f"  {cat:<12} {len(out[cat])}")
    print(f"OK wrote {sum(map(len, out.values()))} openings to {path}")
    return 0


# --------------------------------------------------------------------------
# Step 2: audio
# --------------------------------------------------------------------------


def _tone(freqs: list[float], secs: float, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR_TTS * secs)) / SR_TTS
    return (level * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def _sit() -> np.ndarray:
    """The three rising Special Information Tones that precede intercepts."""
    return np.concatenate(
        [_tone([913.8], 0.274), _tone([1370.6], 0.274), _tone([1776.7], 0.380)]
    )


def _silence(secs: float) -> np.ndarray:
    return np.zeros(int(SR_TTS * max(0.0, secs)), dtype=np.float32)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink-ish background noise, drawn the same way for every label."""
    from scipy.signal import lfilter

    white = rng.standard_normal(n).astype(np.float32)
    return lfilter([0.049922, -0.095993, 0.050612, -0.004408],
                   [1, -2.494956, 2.017265, -0.522189], white).astype(np.float32)


def _phone_line(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """24 kHz float -> 8 kHz band-limited mu-law round trip -> 16 kHz int16."""
    from scipy.signal import butter, resample_poly, sosfilt

    line = resample_poly(audio, 1, 3)
    line = sosfilt(butter(4, [300, 3400], btype="band", fs=SR_LINE, output="sos"), line)
    peak = np.max(np.abs(line)) or 1.0
    line = line / peak * rng.uniform(0.3, 0.9)
    mu = 255.0
    q = np.round(np.sign(line) * np.log1p(mu * np.abs(line)) / np.log1p(mu) * 127)
    line = np.sign(q) * ((1 + mu) ** (np.abs(q) / 127) - 1) / mu
    out = resample_poly(line, 2, 1)
    return (np.clip(out, -1, 1) * 32767).astype(np.int16)


def _timeline(
    tts, text: str, label: str, cat: str, rng: np.random.Generator, voice: str
) -> tuple[np.ndarray, dict]:
    lang = "en-gb" if voice.startswith("b") else "en-us"
    speed = float(rng.uniform(0.85, 1.15))

    def say(t: str) -> np.ndarray:
        samples, sr = tts.create(t, voice=voice, speed=speed, lang=lang)
        assert sr == SR_TTS
        return samples.astype(np.float32)

    onset = float(rng.uniform(0.2, 1.8) if label != "human" else rng.uniform(0.2, 1.5))
    parts = [_silence(onset)]
    meta = {"voice": voice, "speed": round(speed, 2), "onset_s": round(onset, 2)}

    if label == "carrier":
        if rng.random() < 0.6:
            parts += [_sit(), _silence(0.3)]
            meta["sit"] = True
        parts.append(say(text))
    elif label == "voicemail":
        parts.append(say(text))
        if rng.random() < 0.6:
            parts += [_silence(rng.uniform(0.2, 0.6)), _tone([1000.0], rng.uniform(0.4, 0.8))]
            meta["beep"] = True
    else:
        parts.append(say(text))
        # A person waits for the caller, then prompts again.
        repeats = 0
        while repeats < 2 and rng.random() < (0.6 if cat == "hu_short" else 0.4):
            parts += [_silence(rng.uniform(1.5, 4.0)), say(str(rng.choice(FOLLOW_UPS)))]
            repeats += 1
        meta["follow_ups"] = repeats

    audio = np.concatenate(parts)[: int(SR_TTS * CALL_S)]
    audio = np.pad(audio, (0, int(SR_TTS * CALL_S) - len(audio)))
    if rng.random() < 0.7:
        snr_db = float(rng.uniform(10, 40))
        speech_rms = np.sqrt(np.mean(audio[audio != 0] ** 2)) if np.any(audio) else 0.05
        noise = _noise(len(audio), rng)
        noise *= speech_rms / (np.sqrt(np.mean(noise**2)) * 10 ** (snr_db / 20))
        audio = audio + noise
        meta["snr_db"] = round(snr_db, 1)
    return audio, meta


def _utterances(whisper, pcm16k: np.ndarray) -> tuple[list[str], dict[str, list[str]]]:
    """Transcribe the opening, one utterance per pause, like finals from STT."""
    audio = pcm16k.astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(
        audio[: int(SR_OUT * TRANSCRIBE_UNTIL_S)], language="en", beam_size=1
    )
    segs = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    utts: list[tuple[float, str]] = []
    last_end = None
    for start, end, text in segs:
        if utts and last_end is not None and start - last_end < 0.8:
            utts[-1] = (end, utts[-1][1] + " " + text)
        else:
            utts.append((end, text))
        last_end = end
    opening = [t for _, t in utts]
    by_seconds: dict[str, list[str]] = {}
    if utts:
        t0 = utts[0][0]
        for cutoff in TRUNCATION_SECONDS:
            by_seconds[str(cutoff)] = [t for end, t in utts if end - t0 <= cutoff]
    return opening, by_seconds


OPENAI_URL = "https://api.openai.com/v1"
OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse",
]
#: Delivery varied the same way for every label, so accent and pace carry no
#: information about the answer.
ACCENTS = ["American", "British", "Indian", "Nigerian", "Filipino", "Australian", "Hispanic"]


_local = threading.local()


class _Pacer:
    """Spaces requests across all threads to stay under a per-minute limit."""

    def __init__(self, per_minute: float):
        self._gap = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._gap
        time.sleep(max(0.0, start - now))


def _post_with_retry(pacer: "_Pacer", what: str, send):
    """Send via ``send()``, pacing first and waiting out 429s and 5xx."""
    import re

    for attempt in range(8):
        pacer.wait()
        resp = send()
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503) and attempt < 7:
            m = re.search(r"try again in ([\d.]+)s", resp.text)
            time.sleep(float(m.group(1)) + 1 if m else 5 * (attempt + 1))
            continue
        raise SystemExit(f"OpenAI {what} HTTP {resp.status_code}: {resp.text[:300]}")


def _session(key: str):
    """One requests.Session per worker thread."""
    import requests

    if getattr(_local, "http", None) is None:
        _local.http = requests.Session()
        _local.http.headers["Authorization"] = f"Bearer {key}"
    return _local.http


def _openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    env = HERE.parent.parent / "api" / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY (environment or api/.env).")
    return key


#: Paced to the account's per-minute limits (a new OpenAI project allows 10
#: requests per minute to each of TTS and whisper-1); override with
#: --tts-rpm / --asr-rpm.
TTS_PACER = _Pacer(10)
ASR_PACER = _Pacer(10)


class OpenAITTS:
    """``Kokoro.create``-shaped wrapper over OpenAI speech, cached on disk."""

    model = "gpt-4o-mini-tts"

    def __init__(self, key: str, cache: Path, accent: str = "American"):
        self._key = key
        self._cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.accent = accent

    def with_accent(self, accent: str) -> "OpenAITTS":
        """A per-call copy, so concurrent calls never share an accent."""
        return OpenAITTS(self._key, self._cache, accent)

    @property
    def _http(self):
        return _session(self._key)

    def create(self, text: str, voice: str, speed: float, lang: str):
        import hashlib

        pace = "slightly fast" if speed > 1.05 else "slightly slow" if speed < 0.95 else "natural"
        instructions = f"Speak with a {self.accent} English accent at a {pace} pace."
        digest = hashlib.sha1(f"{self.model}|{voice}|{instructions}|{text}".encode()).hexdigest()
        path = self._cache / f"{digest}.pcm"
        if not path.exists():
            resp = _post_with_retry(
                TTS_PACER,
                "TTS",
                lambda: self._http.post(
                    f"{OPENAI_URL}/audio/speech",
                    json={"model": self.model, "voice": voice, "input": text,
                          "instructions": instructions, "response_format": "pcm"},
                    timeout=120,
                ),
            )
            path.write_bytes(resp.content)
        pcm = np.frombuffer(path.read_bytes(), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0, SR_TTS  # pcm output is 24 kHz


class _Segment:
    def __init__(self, start: float, end: float, text: str):
        self.start, self.end, self.text = start, end, text


class OpenAIWhisper:
    """``WhisperModel.transcribe``-shaped wrapper over OpenAI's whisper-1."""

    def __init__(self, key: str):
        self._key = key

    @property
    def _http(self):
        return _session(self._key)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR_OUT)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        resp = _post_with_retry(
            ASR_PACER,
            "transcription",
            lambda: self._http.post(
                f"{OPENAI_URL}/audio/transcriptions",
                files={"file": ("call.wav", buf.getvalue(), "audio/wav")},
                data={"model": "whisper-1", "language": language,
                      "response_format": "verbose_json",
                      "timestamp_granularities[]": "segment"},
                timeout=120,
            ),
        )
        segs = resp.json().get("segments") or []
        return [_Segment(s["start"], s["end"], s["text"]) for s in segs], None


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR_OUT)
        wf.writeframes(pcm.tobytes())


def make_audio(args: argparse.Namespace) -> int:
    scripts = json.loads((OUT / "scripts.json").read_text(encoding="utf-8"))
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset = OUT / "dataset.jsonl"
    done: set[str] = set()
    if dataset.exists():
        # Drop a half-written last line left by an interrupted run.
        kept = []
        for line in dataset.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["scenario_id"])
                kept.append(line)
            except (json.JSONDecodeError, KeyError):
                pass
        dataset.write_text("".join(k + "\n" for k in kept), encoding="utf-8")

    if args.backend == "openai":
        key = _openai_key()
        tts = OpenAITTS(key, OUT / "tts_cache")
        voices = OPENAI_VOICES
        whisper = OpenAIWhisper(key)
    else:
        from faster_whisper import WhisperModel
        from kokoro_onnx import Kokoro

        tts = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
        voices = sorted(v for v in tts.get_voices() if v[:2] in ("af", "am", "bf", "bm"))
        whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    items = [
        (cat, i, text)
        for cat, lines in scripts.items()
        for i, text in enumerate(lines)
        if f"syn-{cat}-{i:03d}" not in done
    ]
    if args.limit:
        items = items[: args.limit]
    # Local models are not thread-safe; the API backend is I/O bound.
    workers = args.workers if isinstance(tts, OpenAITTS) else 1

    def make_one(cat: str, i: int, text: str) -> dict:
        sid = f"syn-{cat}-{i:03d}"
        label = CATEGORIES[cat][0]
        rng = np.random.default_rng(zlib.crc32(f"{args.seed}:{sid}".encode()))
        voice = str(rng.choice(voices))
        call_tts = tts
        if isinstance(tts, OpenAITTS):
            call_tts = tts.with_accent(str(rng.choice(ACCENTS)))
        audio, meta = _timeline(call_tts, text, label, cat, rng, voice)
        if isinstance(call_tts, OpenAITTS):
            meta["accent"] = call_tts.accent
        pcm = _phone_line(audio, rng)
        _write_wav(audio_dir / f"{sid}_user.wav", pcm)
        opening, by_seconds = _utterances(whisper, pcm)
        return {
                "scenario_id": sid,
                "category": cat,
                "expected_label": label,
                "run_id": sid,
                "workflow_id": None,
                "detector_verdict": None,
                "call_status": None,
                "answered_by_raw": None,
                "answered_by": None,
                "opening_utterances": opening,
                "opening_by_seconds": by_seconds,
                "transcript": text,
                "synth": meta,
            }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    started = time.perf_counter()
    made = failed = 0
    total = len(done) + len(items)
    with dataset.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(make_one, *item): item for item in items}
        for fut in as_completed(futures):
            try:
                record = fut.result()
            except (Exception, SystemExit) as e:  # one bad call must not sink the run
                failed += 1
                print(f"  ! {futures[fut][0]}-{futures[fut][1]}: {e}", flush=True)
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            made += 1
            if made % 20 == 0:
                rate = (time.perf_counter() - started) / made
                left = (len(items) - made - failed) * rate
                print(f"  {len(done) + made}/{total}  {rate:.1f}s/call  ~{left / 60:.0f} min left", flush=True)
    print(f"OK {made} new call(s), {failed} failed; dataset at {dataset}")
    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("scripts", help="Write openings with an LLM.")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--model", default=SCRIPT_MODEL)
    p.add_argument(
        "--offline", action="store_true", help="Build openings from templates, no LLM."
    )
    p = sub.add_parser("audio", help="Speak, degrade and transcribe each opening.")
    p.add_argument(
        "--backend",
        choices=("openai", "local"),
        default="openai",
        help="openai: gpt-4o-mini-tts + whisper-1. local: Kokoro + faster-whisper.",
    )
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--limit", type=int, default=0, help="Stop after this many new calls.")
    p.add_argument("--workers", type=int, default=4, help="Parallel API calls.")
    # Just under the 10 RPM limit: pacing at the limit itself trips the
    # rolling window.
    p.add_argument("--tts-rpm", type=float, default=8.0)
    p.add_argument("--asr-rpm", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.step == "audio":
        TTS_PACER = _Pacer(args.tts_rpm)
        ASR_PACER = _Pacer(args.asr_rpm)
    sys.exit((make_scripts if args.step == "scripts" else make_audio)(args))
