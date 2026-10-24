"""Temporal speech-activity features for telling voicemail from a live answer.

A pre-recorded greeting and a live "Hello?" sound alike, but they are shaped
differently in time: a greeting is one long stretch of speech spread evenly
across the opening seconds, while a person says something short and then goes
quiet waiting for the caller. These features describe that shape from Silero
VAD segments on the callee's audio alone.

Definitions follow Saurav, "Real-Time Voicemail Detection in Telephony Audio
Using Temporal Speech Activity Features" (arXiv:2604.09675, Appendix A), where
the first five features carry ~98% of a tree model's importance.
"""

import io
import math
import wave
from functools import lru_cache
from importlib import resources
from math import gcd

import numpy as np

from pipecat.audio.vad.silero import SileroOnnxModel

VAD_SAMPLE_RATE = 16000
_CHUNK = 512  # Silero's frame size at 16 kHz: 32 ms
_CHUNK_MS = 1000.0 * _CHUNK / VAD_SAMPLE_RATE

#: Hysteresis and padding from the paper; minimum durations are Silero's own
#: ``get_speech_timestamps`` defaults.
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.35
SPEECH_PAD_MS = 30.0
MIN_SPEECH_MS = 250.0
MIN_SILENCE_MS = 100.0

SpeechSegment = tuple[float, float]  # (start_ms, end_ms)


@lru_cache(maxsize=1)
def _model_path() -> str:
    return str(resources.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))


@lru_cache(maxsize=1)
def _model() -> SileroOnnxModel:
    # Loading the ONNX session costs ~0.5 s, far more than scoring a window.
    # The model is stateful, so this suits offline, one-call-at-a-time use.
    return SileroOnnxModel(_model_path(), force_onnx_cpu=True)


def read_wav_mono(data: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes into mono int16 samples and a sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * wf.getsampwidth()}-bit")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    return pcm, rate


def _to_vad_rate(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Float32 audio at 16 kHz.

    Silero also accepts 8 kHz, but the paper measured 89% vs 96% accuracy when
    telephony audio was fed at its native rate instead of being resampled.
    """
    audio = pcm.astype(np.float32) / 32768.0
    if sample_rate == VAD_SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(VAD_SAMPLE_RATE, sample_rate)
    return resample_poly(audio, VAD_SAMPLE_RATE // g, sample_rate // g).astype(
        np.float32
    )


def speech_segments(
    pcm: np.ndarray, sample_rate: int, *, window_ms: float | None = None
) -> list[SpeechSegment]:
    """Run Silero VAD over ``pcm`` and return speech segments in milliseconds.

    Args:
        pcm: Mono int16 samples of the callee's audio, starting at call answer.
        sample_rate: Sample rate of ``pcm``.
        window_ms: If given, only the first ``window_ms`` of audio is analysed.
    """
    audio = _to_vad_rate(pcm, sample_rate)
    if window_ms is not None:
        audio = audio[: int(VAD_SAMPLE_RATE * window_ms / 1000.0)]

    model = _model()
    model.reset_states()
    probs = [
        float(model(audio[i : i + _CHUNK], VAD_SAMPLE_RATE)[0][0])
        for i in range(0, len(audio) - _CHUNK + 1, _CHUNK)
    ]
    total_ms = 1000.0 * len(audio) / VAD_SAMPLE_RATE
    return _segments_from_probs(probs, total_ms)


def _segments_from_probs(probs: list[float], total_ms: float) -> list[SpeechSegment]:
    """Turn per-frame speech probabilities into padded, merged segments."""
    raw: list[SpeechSegment] = []
    start: float | None = None
    silence_since: float | None = None

    for i, p in enumerate(probs):
        t = i * _CHUNK_MS
        if start is None:
            if p >= SPEECH_THRESHOLD:
                start, silence_since = t, None
            continue
        if p >= SILENCE_THRESHOLD:
            silence_since = None
            continue
        if silence_since is None:
            silence_since = t
        if t + _CHUNK_MS - silence_since >= MIN_SILENCE_MS:
            raw.append((start, silence_since))
            start, silence_since = None, None

    if start is not None:
        raw.append((start, len(probs) * _CHUNK_MS))

    padded = [
        (max(0.0, s - SPEECH_PAD_MS), min(total_ms, e + SPEECH_PAD_MS))
        for s, e in raw
        if e - s >= MIN_SPEECH_MS
    ]
    merged: list[SpeechSegment] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def temporal_features(
    segments: list[SpeechSegment], window_ms: float
) -> dict[str, float]:
    """The paper's 15 features for speech segments inside a detection window.

    Segments are clipped to ``[0, window_ms]`` first. Empty-input values follow
    the paper's Appendix A (e.g. ``first_onset_ms`` is the whole window).
    """
    W = float(window_ms)
    segs = [(max(0.0, s), min(W, e)) for s, e in segments if s < W and e > 0]
    segs = [(s, e) for s, e in segs if e > s]
    durations = [e - s for s, e in segs]
    n = len(segs)
    total = float(sum(durations))
    half = W / 2.0
    first_half = sum(max(0.0, min(e, half) - s) for s, e in segs)

    if n:
        gaps = [segs[0][0], W - segs[-1][1]]
        gaps += [segs[i][0] - segs[i - 1][1] for i in range(1, n)]
        max_silence = max(gaps)
    else:
        max_silence = W

    mean = total / n if n else 0.0
    std = math.sqrt(sum((d - mean) ** 2 for d in durations) / n) if n > 1 else 0.0

    return {
        "speech_ratio": total / W,
        "num_segments": float(n),
        "mean_seg_ms": mean,
        "longest_seg_ms": max(durations) if n else 0.0,
        "first_onset_ms": segs[0][0] if n else W,
        "first_seg_ms": durations[0] if n else 0.0,
        "total_speech_ms": total,
        "speech_first_half_ratio": first_half / total if total else 0.0,
        "last_speech_end_ratio": segs[-1][1] / W if n else 0.0,
        "silence_ratio": 1.0 - total / W,
        "max_silence_ms": max_silence,
        "speech_in_first_10s": float(any(s < 10000.0 for s, _ in segs)),
        "std_seg_ms": std,
        "short_seg_count": float(sum(d < 300.0 for d in durations)),
        "long_seg_count": float(sum(d > 1000.0 for d in durations)),
    }


def describe_timing(features: dict[str, float], window_ms: float) -> dict:
    """A compact, plainly named view of the features for a language classifier.

    Only the five features that carry nearly all of the signal in the paper,
    in seconds and shares, so a reader need not know the feature definitions.
    """
    speech = features["total_speech_ms"] > 0
    return {
        "listened_for_seconds": round(window_ms / 1000.0, 1),
        "first_speech_after_seconds": (
            round(features["first_onset_ms"] / 1000.0, 1) if speech else None
        ),
        "first_speech_lasted_seconds": round(features["first_seg_ms"] / 1000.0, 1),
        "separate_bursts_of_speech": int(features["num_segments"]),
        "share_of_time_speaking": round(features["speech_ratio"], 2),
        "share_of_speech_in_first_half": (
            round(features["speech_first_half_ratio"], 2) if speech else None
        ),
    }
