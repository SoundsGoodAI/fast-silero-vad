from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fast_silero_vad.constants import VAD_CONFIG_FILE
from fast_silero_vad.segmenter.segmenter import SpeechSegmenter


def _write_model_config(
    tmp_path: Path,
    *,
    samplerate: int = 8000,
    device: str = "cpu",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 32,
    min_silence_duration_ms: int = 32,
    speech_pad_ms: int = 0,
    max_speech_duration_ms: int = 10000,
) -> str:
    model_dir = tmp_path / "vad_model"
    model_dir.mkdir()
    (model_dir / VAD_CONFIG_FILE).write_text(
        f"device: {device}\n"
        "model_type: streaming_vad\n"
        f"model_samplerate: {samplerate}\n"
        f"threshold: {threshold}\n"
        f"min_speech_duration_ms: {min_speech_duration_ms}\n"
        f"min_silence_duration_ms: {min_silence_duration_ms}\n"
        f"speech_pad_ms: {speech_pad_ms}\n"
        f"max_speech_duration_ms: {max_speech_duration_ms}\n",
        encoding="utf8",
    )
    return str(model_dir)


def _segmenter(
    tmp_path: Path,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 32,
    min_silence_duration_ms: int = 32,
    speech_pad_ms: int = 0,
    max_speech_duration_ms: int = 10000,
) -> SpeechSegmenter:
    model_dir = _write_model_config(
        tmp_path,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_ms=max_speech_duration_ms,
    )
    return SpeechSegmenter(OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))


def _probs(values: list[float]) -> np.typing.NDArray[np.float32]:
    return np.array(values, dtype=np.float32)


def test_threshold_uses_config_value(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, threshold=0.05)

    assert segmenter.threshold == pytest.approx(0.05)


def test_threshold_is_inclusive(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, threshold=0.5)

    assert segmenter(_probs([0.5]), total_samples=256, final=True) == [
        {"start": 0.0, "end": 0.032}
    ]


def test_silence_only_never_triggers_even_when_final(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    assert segmenter(_probs([0.0, 0.0, 0.0]), total_samples=768, final=True) == []
    assert segmenter.triggered is False


def test_active_speech_flushes_on_final_without_silence_boundary(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(tmp_path)

    assert segmenter(_probs([0.8, 0.7]), total_samples=512, final=False) == []
    assert segmenter.triggered is True

    assert segmenter(_probs([]), total_samples=512, final=True) == [
        {"start": 0.0, "end": 0.064}
    ]
    assert segmenter.triggered is False


def test_final_flush_discards_too_short_speech(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, min_speech_duration_ms=40)

    assert segmenter(_probs([0.8]), total_samples=256, final=True) == []
    assert segmenter.triggered is False


def test_silence_closure_discards_too_short_speech(tmp_path: Path) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=40, min_silence_duration_ms=32
    )

    assert (
        segmenter(_probs([0.8, 0.0, 0.0, 0.0]), total_samples=1024, final=False) == []
    )
    assert segmenter.triggered is False


def test_segment_closes_after_configured_silence_duration(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    segments = segmenter(
        _probs([0.8, 0.7, 0.0, 0.0, 0.0]), total_samples=1280, final=False
    )

    assert segments == [{"start": 0.0, "end": 0.064}]
    assert segmenter.triggered is False


def test_below_threshold_frames_close_active_speech(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, threshold=0.5, min_silence_duration_ms=32)

    segments = segmenter(
        _probs([0.8, 0.4, 0.4, 0.0, 0.0, 0.0]), total_samples=1536, final=False
    )

    assert segments == [{"start": 0.0, "end": 0.032}]


def test_speech_before_min_silence_cancels_pending_silence(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    segments = segmenter(
        _probs([0.8, 0.0, 0.8, 0.0, 0.0, 0.0]), total_samples=1536, final=False
    )

    assert segments == [{"start": 0.0, "end": 0.096}]


def test_multiple_segments_can_close_in_one_probability_batch(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    segments = segmenter(
        _probs([0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
        total_samples=2048,
        final=False,
    )

    assert segments == [{"start": 0.0, "end": 0.032}, {"start": 0.128, "end": 0.16}]


def test_padding_is_clamped_to_audio_bounds(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, min_silence_duration_ms=32, speech_pad_ms=15)

    segments = segmenter(
        _probs([0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0]), total_samples=1792, final=False
    )

    assert segments == [{"start": 0.017, "end": 0.111}]


def test_segmenter_preserves_state_across_calls(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    assert segmenter(_probs([0.8]), total_samples=256, final=False) == []
    assert segmenter(_probs([0.0]), total_samples=512, final=False) == []
    assert segmenter(_probs([0.0]), total_samples=768, final=False) == [
        {"start": 0.0, "end": 0.032}
    ]


def test_segmenter_carries_only_owned_active_probabilities(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)

    segmenter(_probs([0.8] * 1000), total_samples=256000, final=False)

    assert not isinstance(segmenter.segment_probabilities.base, np.ndarray)


def test_empty_nonfinal_batch_does_not_advance_processed_samples(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(tmp_path)

    assert segmenter(_probs([]), total_samples=0, final=False) == []
    assert segmenter.processed_samples == 0

    assert segmenter(_probs([0.8]), total_samples=256, final=False) == []
    assert segmenter.speech_start == 0


def test_padded_start_is_preserved_across_calls(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, speech_pad_ms=16)

    assert segmenter(_probs([0.0, 0.8]), total_samples=512, final=False) == []
    assert segmenter(_probs([]), total_samples=512, final=True) == [
        {"start": 0.016, "end": 0.064}
    ]


def test_long_segment_splits_at_lowest_probability_inside_safe_slice(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=160
    )

    segments = segmenter(
        _probs([0.9, 0.8, 0.1, 0.7, 0.6, 0.9]),
        total_samples=1536,
        final=True,
    )

    assert segments == [{"start": 0.0, "end": 0.064}, {"start": 0.064, "end": 0.192}]


def test_long_segment_internal_splits_do_not_overlap_when_padded(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(
        tmp_path,
        min_speech_duration_ms=32,
        max_speech_duration_ms=96,
        speech_pad_ms=16,
    )

    segments = segmenter(_probs([0.8] * 6), total_samples=1536, final=True)

    assert segments == [
        {"start": 0.0, "end": 0.064},
        {"start": 0.064, "end": 0.128},
        {"start": 0.128, "end": 0.192},
    ]


def test_long_segment_split_clamps_pending_silence_start(tmp_path: Path) -> None:
    segmenter = _segmenter(
        tmp_path,
        threshold=0.5,
        min_speech_duration_ms=32,
        max_speech_duration_ms=128,
        min_silence_duration_ms=200,
    )

    segments = segmenter(
        _probs([0.8, 0.8, 0.4, 0.1, 0.1]), total_samples=1280, final=False
    )

    assert segments == [{"start": 0.0, "end": 0.096}]
    assert segmenter.triggered is True
    assert segmenter.speech_start == 768
    assert segmenter.silence_start == 768


def test_long_segment_with_equal_probabilities_splits_near_center(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=96
    )

    segments = segmenter(_probs([0.8] * 6), total_samples=1536, final=True)

    assert segments == [
        {"start": 0.0, "end": 0.064},
        {"start": 0.064, "end": 0.128},
        {"start": 0.128, "end": 0.192},
    ]


def test_long_segment_equal_minima_choose_candidate_closest_to_center(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=160
    )

    segments = segmenter(
        _probs([0.9, 0.1, 0.8, 0.1, 0.6, 0.9]), total_samples=1536, final=True
    )

    assert segments == [{"start": 0.0, "end": 0.096}, {"start": 0.096, "end": 0.192}]


def test_long_segment_ignores_unique_edge_minimum_outside_center(
    tmp_path: Path,
) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=160
    )

    segments = segmenter(
        _probs([0.9, 0.51, 0.8, 0.6, 0.7, 0.9]), total_samples=1536, final=True
    )

    assert segments == [{"start": 0.0, "end": 0.096}, {"start": 0.096, "end": 0.192}]


def test_long_segment_splits_across_streaming_calls(tmp_path: Path) -> None:
    segmenter = _segmenter(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=96
    )

    assert segmenter(_probs([0.8, 0.8, 0.8]), total_samples=768, final=False) == []
    assert segmenter(_probs([0.8]), total_samples=1024, final=False) == [
        {"start": 0.0, "end": 0.064}
    ]
    assert segmenter(_probs([]), total_samples=1024, final=True) == [
        {"start": 0.064, "end": 0.128}
    ]


def test_random_streaming_batches_match_one_shot_segmentation(tmp_path: Path) -> None:
    random = np.random.default_rng(20260710)
    probabilities = random.random(10000, dtype=np.float32)
    one_shot_root = tmp_path / "one-shot"
    one_shot_root.mkdir()
    one_shot = _segmenter(
        one_shot_root,
        threshold=0.3,
        min_speech_duration_ms=100,
        max_speech_duration_ms=1000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
    )
    expected = one_shot(
        probabilities,
        total_samples=len(probabilities) * one_shot.chunk_samples,
        final=True,
    )

    streaming_root = tmp_path / "streaming"
    streaming_root.mkdir()
    streaming = _segmenter(
        streaming_root,
        threshold=0.3,
        min_speech_duration_ms=100,
        max_speech_duration_ms=1000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
    )
    actual = []
    start = 0
    while start < len(probabilities):
        end = min(len(probabilities), start + int(random.integers(1, 200)))
        actual.extend(
            streaming(
                probabilities[start:end],
                total_samples=end * streaming.chunk_samples,
                final=end == len(probabilities),
            )
        )
        start = end

    assert actual == expected
