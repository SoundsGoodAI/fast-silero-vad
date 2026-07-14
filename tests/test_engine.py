from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fast_silero_vad.constants import VAD_CONFIG_FILE
from fast_silero_vad.engine.engine import VADEngine


def _audio(num_samples: int) -> np.typing.NDArray[np.float32]:
    return np.full(num_samples, 100 / (2**15 - 1), dtype=np.float32)


class FakeSession:
    def __init__(self) -> None:
        self.model_inputs: list[np.typing.NDArray[np.float32]] = []

    def run(
        self,
        input_feed: dict[str, np.typing.NDArray[np.float32]],
        output_names: list[str],
    ) -> tuple[
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.float32],
        np.typing.NDArray[np.float32],
    ]:
        model_audio = input_feed["input_vad"]
        self.model_inputs.append(model_audio.copy())
        num_chunks = model_audio.shape[1] // 256
        probabilities = np.arange(num_chunks, dtype=np.float32)[np.newaxis, :]
        return (
            probabilities,
            input_feed["cached_left_context"] + 1.0,
            input_feed["h_recurrent_state"] + 2.0,
            input_feed["c_recurrent_state"] + 3.0,
        )


def _write_model_config(tmp_path: Path) -> str:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / VAD_CONFIG_FILE).write_text(
        "model_type: streaming_vad\ndevice: cpu\nmodel_samplerate: 8000\n",
        encoding="utf8",
    )
    return str(model_dir)


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    session = FakeSession()
    monkeypatch.setattr(
        "fast_silero_vad.engine.engine.get_onnxruntime_session",
        lambda path, config: (
            session,
            [
                "input_vad",
                "cached_left_context",
                "h_recurrent_state",
                "c_recurrent_state",
            ],
            [
                "output_vad",
                "output_cached_left_context",
                "output_h_recurrent_state",
                "output_c_recurrent_state",
            ],
            [(1, "input_vad_len"), (1, 32), (1, 3), (1, 3)],
            [(1, "output_vad_len"), (1, 32), (1, 3), (1, 3)],
        ),
    )
    return session


def test_engine_buffers_until_full_chunk(
    tmp_path: Path, fake_session: FakeSession
) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))

    probabilities = engine(_audio(255), final=False)

    assert probabilities.tolist() == []
    assert fake_session.model_inputs == []
    assert len(engine.pending_audio) == 255
    assert engine.total_samples == 255


def test_engine_processes_full_chunks_and_keeps_remainder(
    tmp_path: Path, fake_session: FakeSession
) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))

    probabilities = engine(_audio(300), final=False)

    assert probabilities.tolist() == [0.0]
    assert fake_session.model_inputs[0].shape == (1, 256)
    assert len(engine.pending_audio) == 44
    assert engine.pending_audio.base is None
    assert engine.total_samples == 300
    assert np.all(engine.cached_left_context == 1.0)
    assert np.all(engine.h_recurrent_state == 2.0)
    assert np.all(engine.c_recurrent_state == 3.0)


def test_engine_final_processes_pending_audio(
    tmp_path: Path, fake_session: FakeSession
) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))

    assert engine(_audio(300), final=False).tolist() == [0.0]
    assert engine(_audio(212), final=True).tolist() == [0.0]

    assert [model_input.shape for model_input in fake_session.model_inputs] == [
        (1, 256),
        (1, 256),
    ]
    assert len(engine.pending_audio) == 0
    assert engine.total_samples == 512


def test_engine_buffers_final_tail_shorter_than_one_chunk(
    tmp_path: Path, fake_session: FakeSession
) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))

    probabilities = engine(_audio(255), final=True)

    assert probabilities.tolist() == []
    assert len(engine.pending_audio) == 255
    assert fake_session.model_inputs == []


def test_engine_resamples_normalized_float_audio(
    tmp_path: Path, fake_session: FakeSession
) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))
    engine.apply_samplerate(16000)

    probabilities = engine(np.full(512, 0.25, dtype=np.float32), final=True)

    assert probabilities.tolist() == [0.0]
    assert fake_session.model_inputs[0].shape == (1, 257)
    np.testing.assert_allclose(fake_session.model_inputs[0], 8191 / (2**15 - 1))
    assert engine.total_samples == 257


def test_engine_reset_clears_state(tmp_path: Path, fake_session: FakeSession) -> None:
    model_dir = _write_model_config(tmp_path)
    engine = VADEngine(model_dir, OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE))
    engine(_audio(260), final=False)

    engine.reset()

    assert engine.total_samples == 0
    assert engine.resampler_state is None
    assert engine.pending_audio.tolist() == []
    assert np.all(engine.cached_left_context == 0.0)
    assert np.all(engine.h_recurrent_state == 0.0)
    assert np.all(engine.c_recurrent_state == 0.0)
