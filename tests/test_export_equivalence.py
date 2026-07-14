"""Integration tests against an original Silero VAD JIT checkpoint."""

import os
import shutil
from argparse import Namespace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest


def _get_packaged_silero_vad_jit_path() -> Path:
    """Return the ``silero-vad`` package's bundled JIT checkpoint path."""

    spec = find_spec("silero_vad")
    if spec is None or spec.origin is None:
        pytest.skip(
            "Install the export extra to use --download-silero-vad.",
            allow_module_level=False,
        )
    model_path = Path(spec.origin).resolve().parent / "data" / "silero_vad.jit"
    if not model_path.is_file():
        pytest.fail(f"silero-vad package does not contain {model_path}.")
    return model_path


@pytest.fixture(scope="module")
def silero_vad_jit_path(pytestconfig: pytest.Config) -> Path:
    """Return an explicit or checksum-verified Silero VAD checkpoint."""

    value = os.environ.get("SILERO_VAD_JIT_PATH")
    if value is None:
        if pytestconfig.getoption("--download-silero-vad"):
            return _get_packaged_silero_vad_jit_path()
        pytest.skip(
            "Set SILERO_VAD_JIT_PATH or pass --download-silero-vad to run "
            "export equivalence tests."
        )

    model_path = Path(value)
    if not model_path.is_file():
        pytest.fail(f"SILERO_VAD_JIT_PATH does not exist: {model_path}")
    return model_path


@pytest.mark.parametrize(
    ("vad_branch", "use_custom_op"),
    (
        pytest.param("16k", False, id="16k-standard-onnx"),
        pytest.param("8k", False, id="8k-standard-onnx"),
        pytest.param("16k", True, id="16k-custom-op"),
        pytest.param("8k", True, id="8k-custom-op"),
    ),
)
def test_exported_model_matches_original_jit(
    tmp_path: Path, silero_vad_jit_path: Path, vad_branch: str, use_custom_op: bool
) -> None:
    """Compare stateful chunk probabilities from original and exported models."""

    torch = pytest.importorskip("torch", reason="Install the export extra.")
    if use_custom_op and shutil.which("c++") is None:
        pytest.skip("A C++ compiler is required for the custom-op export test.")

    from fast_silero_vad import VAD
    from fast_silero_vad.export.package_model import main as package_model

    samplerate = 16000 if vad_branch == "16k" else 8000
    chunk_samples = 512 if vad_branch == "16k" else 256
    model_dir = tmp_path / f"model-{vad_branch}-{use_custom_op}"
    package_model(
        Namespace(
            model_path=str(silero_vad_jit_path),
            output_dir_path=str(model_dir),
            model_type="offline_vad",
            vad_branch=vad_branch,
            threshold=0.5,
            min_speech_duration_ms=100,
            max_speech_duration_ms=30000,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            use_onnxruntime_custom_op=use_custom_op,
            debug=False,
        )
    )
    for license_file in ("LICENSE", "NOTICE", "SILERO-VAD-LICENSE.txt"):
        assert model_dir.joinpath(license_file).is_file()

    original_model = torch.jit.load(str(silero_vad_jit_path), map_location="cpu")
    original_model.reset_states()
    exported_model = VAD(str(model_dir))

    random = np.random.default_rng(20260710)
    pcm = random.integers(-12000, 12001, size=(12, chunk_samples), dtype=np.int16)
    original_probabilities = []
    exported_probabilities = []
    with torch.inference_mode():
        for index, pcm_chunk in enumerate(pcm):
            waveform = pcm_chunk.astype(np.float32) / np.float32(2**15 - 1)
            original_output = original_model(
                torch.from_numpy(waveform[np.newaxis, :]), samplerate
            )
            original_probabilities.append(float(original_output[0, 0]))
            exported_output = exported_model.engine(
                waveform, final=index == len(pcm) - 1
            )
            exported_probabilities.append(float(exported_output[0]))

    np.testing.assert_allclose(
        exported_probabilities, original_probabilities, rtol=1e-4, atol=1e-6
    )
