"""Tests for the temporal voicemail features and the Jev payload that carries them."""

import io
import wave

import numpy as np
import pytest

from api.services.pipecat.typesafe_jev_llm import (
    INSTRUCTIONS,
    TIMING_INSTRUCTIONS,
    build_payload,
)
from api.services.pipecat.voicemail_timing import (
    _segments_from_probs,
    describe_timing,
    read_wav_mono,
    speech_segments,
    temporal_features,
)

W = 5000.0


def test_greeting_shape():
    """One long segment after a short delay: the voicemail signature."""
    f = temporal_features([(800.0, 4800.0)], W)

    assert f["num_segments"] == 1
    assert f["first_onset_ms"] == 800.0
    assert f["first_seg_ms"] == 4000.0
    assert f["speech_ratio"] == pytest.approx(0.8)
    assert f["silence_ratio"] == pytest.approx(0.2)
    # 1700 ms of the 4000 ms falls in [0, 2500].
    assert f["speech_first_half_ratio"] == pytest.approx(1700 / 4000)
    assert f["last_speech_end_ratio"] == pytest.approx(4800 / 5000)
    assert f["max_silence_ms"] == 800.0
    assert f["long_seg_count"] == 1
    assert f["short_seg_count"] == 0
    assert f["std_seg_ms"] == 0.0


def test_live_answer_shape():
    """A short "Hello?" then expectant silence is front-loaded."""
    f = temporal_features([(300.0, 800.0), (2000.0, 2200.0)], W)

    assert f["num_segments"] == 2
    assert f["speech_first_half_ratio"] == 1.0
    assert f["max_silence_ms"] == 2800.0  # trailing gap after the last burst
    assert f["short_seg_count"] == 1
    assert f["mean_seg_ms"] == 350.0
    assert f["std_seg_ms"] == 150.0


def test_silent_callee_uses_appendix_defaults():
    f = temporal_features([], W)

    assert f["first_onset_ms"] == W
    assert f["max_silence_ms"] == W
    assert f["speech_ratio"] == 0.0
    assert f["speech_first_half_ratio"] == 0.0
    assert f["speech_in_first_10s"] == 0.0


def test_segments_are_clipped_to_the_window():
    f = temporal_features([(4000.0, 9000.0), (6000.0, 7000.0)], W)

    assert f["num_segments"] == 1
    assert f["total_speech_ms"] == 1000.0


def test_hysteresis_holds_through_a_dip_and_ends_on_silence():
    chunk = 32.0
    # 20 frames speech, 1 frame dip above the silence threshold, 10 frames
    # speech, then silence long enough to close the segment.
    probs = [0.9] * 20 + [0.4] + [0.9] * 10 + [0.1] * 10
    segs = _segments_from_probs(probs, len(probs) * chunk)

    assert len(segs) == 1
    start, end = segs[0]
    assert start == 0.0
    assert end == pytest.approx(31 * chunk + 30.0)


def test_short_blips_are_dropped():
    probs = [0.1] * 5 + [0.9] * 3 + [0.1] * 20  # ~96 ms, under the minimum
    assert _segments_from_probs(probs, len(probs) * 32.0) == []


def _wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def test_wav_round_trip_and_silence_has_no_speech():
    pcm, rate = read_wav_mono(_wav(np.zeros(8000 * 3), 8000))

    assert rate == 8000
    assert len(pcm) == 24000
    assert speech_segments(pcm, rate) == []


def test_describe_timing_uses_seconds_and_hides_empty_values():
    view = describe_timing(temporal_features([(800.0, 4800.0)], W), W)
    assert view == {
        "listened_for_seconds": 5.0,
        "first_speech_after_seconds": 0.8,
        "first_speech_lasted_seconds": 4.0,
        "separate_bursts_of_speech": 1,
        "share_of_time_speaking": 0.8,
        "share_of_speech_in_first_half": 0.42,
    }

    empty = describe_timing(temporal_features([], W), W)
    assert empty["first_speech_after_seconds"] is None
    assert empty["share_of_speech_in_first_half"] is None


def test_payload_without_timing_is_unchanged():
    payload = build_payload(["hello?"], model="m")

    assert payload["state"] == {"answering_side_said": ["hello?"]}
    assert payload["questions"]["answered_by"]["instructions"] == INSTRUCTIONS


def test_payload_with_timing_adds_state_and_instructions():
    view = {"listened_for_seconds": 5.0}
    payload = build_payload(["hello?"], model="m", timing=view)

    assert payload["state"]["answering_side_timing"] == view
    assert payload["questions"]["answered_by"]["instructions"] == (
        INSTRUCTIONS + TIMING_INSTRUCTIONS
    )
