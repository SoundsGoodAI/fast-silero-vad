"""CPU ONNX Runtime engine for VAD bundles.

The engine accepts normalized mono float32 audio chunks, optionally resamples
them to the model sample rate, buffers partial model chunks, and runs the
exported VAD ONNX graph whenever at least one complete model chunk is
available. It keeps ONNX recurrent state, resampler state, and pending audio
between calls.
"""

from audioop import ratecv

import numpy as np
from omegaconf import DictConfig

from ..constants import VAD_ONNX_FILE
from ..utils import get_initial_states, get_onnxruntime_session


class VADEngine:
    """Run chunked audio through a stateful VAD ONNX model."""

    def __init__(self, model_dir: str, model_config: DictConfig) -> None:
        """Initialize the VAD ONNX Runtime session.

        Parameters
        ----------
        model_dir : str
            Path to an unpacked VAD model bundle containing ``vad.onnx``.
        model_config : DictConfig
            VAD model configuration loaded from ``model_config.yaml``.

        Notes
        -----
        Model bundle validation is expected to happen before engine
        construction through ``VAD``.
        """

        self.model_samplerate = model_config.model_samplerate
        self.signal_samplerate = model_config.model_samplerate
        self.chunk_samples = 512 if model_config.model_samplerate == 16000 else 256
        self.resampler_state: tuple[int, tuple[tuple[int, int], ...]] | None = None
        self.pending_audio = np.empty(0, dtype=np.float32)

        self.total_samples = 0
        self.max_signed_int = np.float32(2**15 - 1)
        self.pcm_sample_width = 2
        self.pcm_channels = 1

        (
            self.vad,
            input_names,
            self.output_names,
            input_shapes,
            _,
        ) = get_onnxruntime_session(f"{model_dir}/{VAD_ONNX_FILE}", model_config)

        input_shapes_by_name = dict(zip(input_names, input_shapes, strict=True))
        self.context_samples = int(input_shapes_by_name["cached_left_context"][1])
        self.hidden_dim = int(input_shapes_by_name["h_recurrent_state"][1])

        self.reset()

    def reset(self) -> None:
        """Reset recurrent, resampling, counters, and pending audio state."""

        (
            self.cached_left_context,
            self.h_recurrent_state,
            self.c_recurrent_state,
        ) = get_initial_states(self.context_samples, self.hidden_dim)

        self.total_samples = 0
        self.resampler_state = None
        self.pending_audio = np.empty(0, dtype=np.float32)

    def apply_samplerate(self, samplerate: int) -> None:
        """Set the sample rate expected for subsequent input chunks.

        Parameters
        ----------
        samplerate : int
            Sample rate of incoming audio. If it differs from the packaged
            model sample rate, the engine resamples chunks while carrying
            ``audioop.ratecv`` state between calls.
        """

        self.signal_samplerate = samplerate

    def __call__(
        self, audio: np.typing.NDArray[np.float32], final: bool
    ) -> np.typing.NDArray[np.float32]:
        """Run VAD on the next normalized mono audio chunk.

        Parameters
        ----------
        audio : np.typing.NDArray[np.float32]
            One-dimensional float32 audio normalized to ``[-1.0, 1.0]``.
        final : bool
            Whether this is the last chunk of the current audio stream. When
            false, only complete model chunks are consumed and any remainder is
            kept in ``pending_audio``. When true, all pending audio is consumed
            if it is at least one model chunk long.

        Returns
        -------
        np.typing.NDArray[np.float32]
            One-dimensional float32 array of chunk-level speech probabilities.
            If not enough audio is buffered for one model chunk, an empty array
            is returned.
        """

        if self.signal_samplerate != self.model_samplerate:
            pcm = np.clip(
                audio * self.max_signed_int,
                -self.max_signed_int,
                self.max_signed_int,
            ).astype(np.int16)

            if self.resampler_state is None:
                pcm = np.hstack((pcm[:1], pcm))

            resampled_pcm, self.resampler_state = ratecv(
                pcm.tobytes(),
                self.pcm_sample_width,
                self.pcm_channels,
                self.signal_samplerate,
                self.model_samplerate,
                self.resampler_state,
            )
            audio = (
                np.frombuffer(resampled_pcm, dtype=np.int16).astype(np.float32)
                / self.max_signed_int
            )

        self.total_samples += len(audio)
        self.pending_audio = np.hstack((self.pending_audio, audio))

        if len(self.pending_audio) < self.chunk_samples:
            return np.empty(0, dtype=np.float32)

        process_samples = len(self.pending_audio)
        if not final:
            process_samples = process_samples // self.chunk_samples * self.chunk_samples

        model_audio = self.pending_audio[:process_samples]
        self.pending_audio = self.pending_audio[process_samples:].copy()

        vad_inputs = {
            "input_vad": model_audio[np.newaxis, :],
            "cached_left_context": self.cached_left_context,
            "h_recurrent_state": self.h_recurrent_state,
            "c_recurrent_state": self.c_recurrent_state,
        }

        (
            vad_output,
            self.cached_left_context,
            self.h_recurrent_state,
            self.c_recurrent_state,
        ) = self.vad.run(input_feed=vad_inputs, output_names=self.output_names)

        return vad_output.squeeze(axis=0)
