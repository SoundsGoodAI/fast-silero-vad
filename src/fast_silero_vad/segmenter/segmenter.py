"""Convert VAD probabilities into timestamped speech segments.

The ONNX VAD model emits one probability per fixed-size audio chunk. The
``SpeechSegmenter`` keeps the small amount of post-processing state needed to
turn that probability stream into finalized speech intervals with minimum
speech duration, maximum speech duration, minimum silence duration, and padding
rules. Padding is applied at natural VAD boundaries; forced maximum-duration
splits are emitted as adjacent non-overlapping segments.
"""

import numpy as np
from numba import njit
from omegaconf import DictConfig


class SpeechSegmenter:
    """Stateful post-processor for chunk-level VAD probabilities."""

    def __init__(self, model_config: DictConfig) -> None:
        """Initialize segment thresholds and model timing.

        Parameters
        ----------
        model_config : DictConfig
            VAD model configuration loaded from ``model_config.yaml``.
        """

        self.sampling_rate = model_config.model_samplerate
        self.chunk_samples = 512 if model_config.model_samplerate == 16000 else 256
        self.threshold = model_config.threshold
        # Bidirectional smoothing keeps forced max-duration split decisions from
        # following single-chunk probability glitches; alpha=0.5 gives each
        # one-sided smoother about one chunk of mean delay before averaging.
        self.smooth_alpha = 0.5
        self.min_speech_samples = round(
            model_config.min_speech_duration_ms * model_config.model_samplerate / 1000,
        )
        self.max_speech_samples = round(
            model_config.max_speech_duration_ms * model_config.model_samplerate / 1000,
        )
        self.min_silence_samples = round(
            model_config.min_silence_duration_ms * model_config.model_samplerate / 1000,
        )
        self.speech_pad_samples = round(
            model_config.speech_pad_ms * model_config.model_samplerate / 1000
        )
        self.reset()

    def reset(self) -> None:
        """Reset active speech, output boundary, silence, and sample counters."""

        self.processed_samples = 0
        self.triggered = False
        self.segment_start = 0
        self.speech_start = 0
        self.silence_start = -1
        self.segment_probabilities = np.empty(0, dtype=np.float32)

    def __call__(
        self,
        probabilities: np.typing.NDArray[np.float32],
        total_samples: int,
        final: bool,
    ) -> list[dict[str, float]]:
        """Process one batch of VAD probabilities.

        Parameters
        ----------
        probabilities : np.typing.NDArray[np.float32]
            One-dimensional array of speech probabilities, one value per VAD
            model chunk.
        total_samples : int
            Total number of model-sample-rate audio samples observed so far in
            the current stream or complete offline segment.
        final : bool
            Whether no more probabilities will arrive for the current audio.
            If true, an active speech segment is flushed at ``total_samples``.

        Returns
        -------
        list[dict[str, float]]
            Finalized speech segments with ``start`` and ``end`` fields. Times
            are reported in seconds. Segments produced by maximum-duration
            splits are adjacent and do not overlap.
        """
        (
            segment_starts,
            segment_ends,
            self.segment_probabilities,
            self.processed_samples,
            self.triggered,
            self.segment_start,
            self.speech_start,
            self.silence_start,
        ) = self.process_probabilities(
            probabilities,
            self.segment_probabilities,
            self.sampling_rate,
            self.processed_samples,
            total_samples,
            self.chunk_samples,
            self.threshold,
            self.smooth_alpha,
            self.min_speech_samples,
            self.max_speech_samples,
            self.min_silence_samples,
            self.speech_pad_samples,
            self.triggered,
            self.segment_start,
            self.speech_start,
            self.silence_start,
        )

        segments = [
            {"start": float(segment_start), "end": float(segment_end)}
            for segment_start, segment_end in zip(
                segment_starts, segment_ends, strict=True
            )
        ]

        if final:
            segments.extend(self.flush(total_samples))

        return segments

    def flush(self, total_samples: int) -> list[dict[str, float]]:
        """Finalize an open speech segment at end of audio.

        Parameters
        ----------
        total_samples : int
            End position of the current audio in model-sample-rate samples.

        Returns
        -------
        list[dict[str, float]]
            Empty list when no valid speech is active, otherwise one finalized
            speech segment ending at ``total_samples``.
        """

        if not self.triggered:
            return []

        self.triggered = False
        self.silence_start = -1
        self.segment_probabilities = np.empty(0, dtype=np.float32)

        if total_samples - self.speech_start < self.min_speech_samples:
            return []

        start = round(self.segment_start / self.sampling_rate, 3)
        end = round(total_samples / self.sampling_rate, 3)

        return [{"start": start, "end": end}]

    @staticmethod
    @njit(nogil=True, cache=True)
    def process_probabilities(
        probabilities: np.typing.NDArray[np.float32],
        segment_probabilities: np.typing.NDArray[np.float32],
        sampling_rate: int,
        processed_samples: int,
        total_samples: int,
        chunk_samples: int,
        threshold: float,
        smooth_alpha: float,
        min_speech_samples: int,
        max_speech_samples: int,
        min_silence_samples: int,
        speech_pad_samples: int,
        triggered: bool,
        segment_start: int,
        speech_start: int,
        silence_start: int,
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float32],
        int,
        bool,
        int,
        int,
        int,
    ]:
        """Scan chunk probabilities and update segmenter state.

        Parameters
        ----------
        probabilities : np.typing.NDArray[np.float32]
            New speech probabilities, one value per VAD model chunk.
        segment_probabilities : np.typing.NDArray[np.float32]
            Probabilities carried from the currently active speech segment in a
            previous call. Empty when no segment is active.
        sampling_rate : int
            Model sampling rate used to convert sample offsets to seconds.
        processed_samples : int
            Number of model-rate samples processed before this batch.
        total_samples : int
            Number of model-rate samples available in the current audio stream.
            This caps padded natural segment endings.
        chunk_samples : int
            Number of samples represented by one VAD probability.
        threshold : float
            Probability threshold that opens or continues a speech segment.
        smooth_alpha : float
            Exponential smoothing factor used only when selecting forced split
            points for overlong speech segments.
        min_speech_samples : int
            Minimum unpadded speech duration required for an emitted segment.
        max_speech_samples : int
            Maximum speech duration before the active segment is split.
        min_silence_samples : int
            Silence duration required to close a natural speech segment.
        speech_pad_samples : int
            Padding added to natural VAD boundaries. Forced max-duration splits
            are emitted without padding to keep adjacent chunks non-overlapping.
        triggered : bool
            Whether a speech segment was active before this batch.
        segment_start : int
            Padded start sample of the active output segment.
        speech_start : int
            Unpadded start sample of the active speech region.
        silence_start : int
            First below-threshold sample in the current trailing silence, or
            ``-1`` when no closing silence is being tracked.

        Returns
        -------
        tuple[
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float32],
            int,
            bool,
            int,
            int,
            int,
        ]
            Finalized segment start times in seconds, finalized segment end
            times in seconds, carried active probabilities, updated processed
            sample count, updated active flag, updated padded segment start,
            updated unpadded speech start, and updated silence start.

        Notes
        -----
        Long speech intervals are split after they exceed
        ``max_speech_samples``. Active probabilities are smoothed in both time
        directions before split scoring. Split candidates are scored by
        dividing smoothed speech probability by ``p * (1 - p)``, where ``p`` is
        the share of the active interval before the candidate split. This keeps
        splits near low-probability regions while biasing away from the segment
        edges, so both sides remain at least ``min_speech_samples`` long.
        Active and smoothing buffers are bounded by ``max_speech_samples``
        rather than by the duration of the complete input recording.
        """

        min_speech_chunks = (min_speech_samples + chunk_samples - 1) // chunk_samples
        max_speech_chunks = (max_speech_samples + chunk_samples - 1) // chunk_samples
        max_segments = len(probabilities) // min_speech_chunks + 2

        segment_starts = np.empty(max_segments, dtype=np.float64)
        segment_ends = np.empty(max_segments, dtype=np.float64)

        active_capacity = max(max_speech_chunks + 1, len(segment_probabilities) + 1)
        active_probabilities = np.empty(active_capacity, dtype=np.float32)
        active_probabilities[: len(segment_probabilities)] = segment_probabilities
        smoothed_probabilities = np.empty(active_capacity, dtype=np.float32)
        active_count = len(segment_probabilities)
        num_segments = 0

        for probability in probabilities:
            if probability >= threshold:
                if not triggered:
                    triggered = True
                    speech_start = processed_samples
                    segment_start = max(0, speech_start - speech_pad_samples)
                    active_count = 0
                silence_start = -1
            else:
                if triggered and silence_start == -1:
                    silence_start = processed_samples

            if triggered:
                active_probabilities[active_count] = probability
                active_count += 1

                if active_count > max_speech_chunks:
                    candidate_start = min_speech_chunks
                    candidate_end = max_speech_chunks - min_speech_chunks

                    q = 1.0 - smooth_alpha
                    smoothed_probabilities[0] = active_probabilities[0]
                    for idx in range(1, active_count):
                        smoothed_probabilities[idx] = (
                            smooth_alpha * active_probabilities[idx]
                            + q * smoothed_probabilities[idx - 1]
                        )

                    backward_probability = active_probabilities[active_count - 1]
                    smoothed_probabilities[active_count - 1] = (
                        smoothed_probabilities[active_count - 1] + backward_probability
                    ) / 2.0
                    for idx in range(active_count - 2, -1, -1):
                        backward_probability = (
                            smooth_alpha * active_probabilities[idx]
                            + q * backward_probability
                        )
                        smoothed_probabilities[idx] = (
                            smoothed_probabilities[idx] + backward_probability
                        ) / 2.0

                    split_chunks = candidate_start
                    split_share = candidate_start / active_count
                    min_split_score = smoothed_probabilities[candidate_start] / (
                        split_share * (1.0 - split_share)
                    )
                    for idx in range(candidate_start + 1, candidate_end + 1):
                        split_share = idx / active_count
                        split_score = smoothed_probabilities[idx] / (
                            split_share * (1.0 - split_share)
                        )
                        if split_score < min_split_score:
                            min_split_score = split_score
                            split_chunks = idx

                    split_sample = speech_start + split_chunks * chunk_samples
                    segment_starts[num_segments] = round(
                        segment_start / sampling_rate, 3
                    )
                    segment_ends[num_segments] = round(split_sample / sampling_rate, 3)
                    num_segments += 1

                    active_count -= split_chunks
                    for idx in range(active_count):
                        active_probabilities[idx] = active_probabilities[
                            split_chunks + idx
                        ]

                    speech_start = split_sample
                    segment_start = split_sample
                    if -1 < silence_start < speech_start:
                        silence_start = speech_start

                if (
                    probability < threshold
                    and processed_samples - silence_start >= min_silence_samples
                ):
                    speech_end = silence_start
                    triggered = False
                    silence_start = -1
                    if speech_end - speech_start >= min_speech_samples:
                        segment_starts[num_segments] = round(
                            segment_start / sampling_rate, 3
                        )
                        segment_ends[num_segments] = round(
                            min(total_samples, speech_end + speech_pad_samples)
                            / sampling_rate,
                            3,
                        )
                        num_segments += 1
                    active_count = 0

            processed_samples += chunk_samples

        segment_starts = segment_starts[:num_segments]
        segment_ends = segment_ends[:num_segments]
        segment_probabilities = active_probabilities[:active_count].copy()

        return (
            segment_starts,
            segment_ends,
            segment_probabilities,
            processed_samples,
            triggered,
            segment_start,
            speech_start,
            silence_start,
        )
