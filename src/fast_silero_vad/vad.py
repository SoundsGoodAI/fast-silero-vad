"""Fast Silero VAD runtime wrapper.

The wrapper combines the configured stateful VAD engine with the speech
segmenter. Offline bundles reset before each complete audio input. Streaming
bundles preserve state across non-final chunks and reset after the final chunk.
"""

from time import perf_counter

import numpy as np
from omegaconf import OmegaConf

from .constants import MODEL_TYPE_STREAMING_VAD, VAD_CONFIG_FILE
from .engine.engine import VADEngine
from .segmenter.segmenter import SpeechSegmenter
from .utils import VADInferenceError, validate_model_dir


class VAD:
    """Run normalized mono audio through a packaged VAD pipeline."""

    def __init__(self, model_dir: str, validate: bool = True) -> None:
        """Initialize the VAD engine and segmenter.

        Parameters
        ----------
        model_dir : str
            Path to an unpacked VAD model bundle.
        validate : bool, default=True
            Whether to validate the bundle before initializing the runtime.
        """

        if validate:
            validate_model_dir(model_dir)

        model_config = OmegaConf.load(f"{model_dir}/{VAD_CONFIG_FILE}")
        self.is_offline = model_config.model_type != MODEL_TYPE_STREAMING_VAD
        self.engine = VADEngine(model_dir, model_config)
        self.segmenter = SpeechSegmenter(model_config)
        self.timing = {"engine_sec": 0.0, "segmenter_sec": 0.0, "calls": 0}

    def reset(self) -> None:
        """Reset engine recurrent state, pending audio, and segmenter state."""

        self.engine.reset()
        self.segmenter.reset()

    def apply_samplerate(self, samplerate: int) -> None:
        """Set the sample rate expected for subsequent input audio.

        Parameters
        ----------
        samplerate : int
            Sample rate of incoming audio. The engine resamples to the packaged
            model sample rate when necessary.

        Raises
        ------
        VADInferenceError
            Raised when ``samplerate`` is not a positive integer.
        """

        if not isinstance(samplerate, int) or samplerate <= 0:
            raise VADInferenceError(
                f"samplerate must be a positive integer, got {samplerate}."
            )

        self.engine.apply_samplerate(samplerate)

    def __call__(
        self,
        audio: np.typing.NDArray[np.float32 | np.float64],
        final: bool = True,
    ) -> list[dict[str, float]]:
        """Process one complete offline input or one streaming audio chunk.

        Parameters
        ----------
        audio : np.typing.NDArray[np.float32 | np.float64]
            One-dimensional floating-point audio normalized to ``[-1.0, 1.0]``.
            Input is converted to float32 before inference.
        final : bool, default=True
            Whether no more audio will arrive for the current stream. Streaming
            bundles preserve state when false and reset after a final call.
            Offline bundles always treat every input as final.

        Returns
        -------
        list[dict[str, float]]
            Speech segments finalized by this input, with ``start`` and ``end``
            fields. The list can be empty when no segment boundary is reached.

        Raises
        ------
        VADInferenceError
            Raised when ``audio`` is not one-dimensional.
        """

        if audio.ndim != 1:
            raise VADInferenceError(
                f"audio must be one-dimensional, got {audio.shape}."
            )

        audio = audio.astype(np.float32, copy=False)
        final = self.is_offline or final

        timer = perf_counter()
        probabilities = self.engine(audio, final)
        self.timing["engine_sec"] += perf_counter() - timer

        timer = perf_counter()
        segments = self.segmenter(probabilities, self.engine.total_samples, final)
        self.timing["segmenter_sec"] += perf_counter() - timer
        self.timing["calls"] += 1

        if final:
            self.reset()
        return segments
