"""Voice activity detection model used during model packaging.

This module contains a compact PyTorch implementation of a Silero VAD model.
It is written with TorchScript export in mind: the forward path avoids
data-dependent Python branches where possible and keeps all recurrent state in
explicit tensor arguments.
"""

import torch


class VAD(torch.nn.Module):
    """Single-channel voice activity detector.

    The model consumes one waveform batch shaped (1, num_samples), splits it
    into fixed-size chunks, prepends the previous left context to each chunk,
    runs the Silero convolutional STFT and encoder stack, and returns one speech
    probability per chunk.
    """

    def __init__(
        self,
        chunk_samples: int,
        context_samples: int,
        cutoff: int,
        hidden_dim1: int,
        hidden_dim2: int,
        encoder_kernel_size: int,
        encoder_stride: int,
        encoder_padding: int,
        device: torch.device,
    ) -> None:
        """Initialize the reconstructed Silero VAD layers.

        Parameters
        ----------
        chunk_samples : int
            Number of waveform samples decoded per VAD step.
        context_samples : int
            Number of previous waveform samples prepended to each step.
        cutoff : int
            Number of magnitude STFT bins retained after combining real and
            imaginary convolution outputs.
        hidden_dim1 : int
            Channel count used by the middle convolutional encoder layers.
        hidden_dim2 : int
            Channel count used by the first/final convolutional encoder layers
            and recurrent state.
        encoder_kernel_size : int
            Kernel size for all convolutional encoder layers.
        encoder_stride : int
            Stride used by the downsampling convolutional encoder layers.
        encoder_padding : int
            Padding used by all convolutional encoder layers.
        device : torch.device
            Device where model parameters are allocated.
        """

        super().__init__()

        self.chunk_samples = chunk_samples
        self.context_samples = context_samples
        self.cutoff = cutoff

        self.relu = torch.nn.ReLU()
        self.stft_conv = torch.nn.Conv1d(
            1,
            cutoff * 2,
            chunk_samples // 2,
            stride=chunk_samples // 4,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.conv1 = torch.nn.Conv1d(
            cutoff,
            hidden_dim2,
            encoder_kernel_size,
            padding=encoder_padding,
            dtype=torch.float32,
            device=device,
        )
        self.conv2 = torch.nn.Conv1d(
            hidden_dim2,
            hidden_dim1,
            encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_padding,
            dtype=torch.float32,
            device=device,
        )
        self.conv3 = torch.nn.Conv1d(
            hidden_dim1,
            hidden_dim1,
            encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_padding,
            dtype=torch.float32,
            device=device,
        )
        self.conv4 = torch.nn.Conv1d(
            hidden_dim1,
            hidden_dim2,
            encoder_kernel_size,
            padding=encoder_padding,
            dtype=torch.float32,
            device=device,
        )
        self.rnn = torch.nn.LSTM(
            hidden_dim2,
            hidden_dim2,
            batch_first=True,
            dtype=torch.float32,
            device=device,
        )
        self.depthwise_conv = torch.nn.Conv1d(
            hidden_dim2, 1, 1, dtype=torch.float32, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        left_context: torch.Tensor,
        h_state: torch.Tensor,
        c_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute speech probabilities for a waveform segment.

        Parameters
        ----------
        x : torch.Tensor
            Mono waveform shaped (1, num_samples). The batch dimension
            is part of the exported graph contract and is always equal to one.
        left_context : torch.Tensor
            Cached waveform context shaped (1, context_samples). Pass the
            returned context into the next call to continue a stream.
        h_state : torch.Tensor
            LSTM hidden state shaped (1, hidden_dim2).
        c_state : torch.Tensor
            LSTM cell state shaped (1, hidden_dim2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Speech probabilities shaped (1, num_chunks), updated waveform
            context shaped (1, context_samples), updated hidden state, and
            updated cell state.
        """

        pad_len = (
            self.chunk_samples - x.size(1) % self.chunk_samples
        ) % self.chunk_samples

        x = torch.cat(
            (x, torch.zeros(1, pad_len, dtype=torch.float32, device=x.device)),
            dim=1,
        )

        num_chunks = x.size(1) // self.chunk_samples
        chunks = x.reshape(1, num_chunks, self.chunk_samples)
        chunk_tails = chunks[
            :, :, self.chunk_samples - self.context_samples : self.chunk_samples
        ]
        contexts = torch.cat(
            (left_context.unsqueeze(1), chunk_tails[:, : num_chunks - 1, :]),
            dim=1,
        )
        left_context = chunk_tails[:, num_chunks - 1 : num_chunks, :].squeeze(1)

        chunk = torch.cat((contexts, chunks), dim=2)
        chunk = chunk.reshape(num_chunks, 1, self.context_samples + self.chunk_samples)
        chunk = torch.nn.functional.pad(
            chunk,
            (0, self.context_samples),
            mode="reflect",
        )

        chunk = self.stft_conv(chunk)
        chunk = torch.pow(chunk, 2)
        chunk = chunk[:, : self.cutoff, :] + chunk[:, self.cutoff :, :]
        chunk = torch.sqrt(chunk)

        chunk = self.relu(self.conv1(chunk))
        chunk = self.relu(self.conv2(chunk))
        chunk = self.relu(self.conv3(chunk))
        chunk = self.relu(self.conv4(chunk))
        chunk = chunk.squeeze(2).unsqueeze(0)

        chunk, (h_state, c_state) = self.rnn(
            chunk,
            (h_state.unsqueeze(0), c_state.unsqueeze(0)),
        )
        h_state = h_state.squeeze(0)
        c_state = c_state.squeeze(0)

        chunk = self.depthwise_conv(self.relu(chunk).transpose(1, 2))
        outputs = torch.nn.functional.sigmoid(chunk).squeeze(1)

        return outputs, left_context, h_state, c_state


def get_init_states(
    left_cache_len: int,
    hidden_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create initial waveform context and recurrent VAD states.

    Parameters
    ----------
    left_cache_len : int
        Number of waveform samples cached between VAD chunks.
    hidden_dim : int
        Hidden dimension of the VAD LSTM.
    device : torch.device
        Device where state tensors are allocated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Zero left context shaped (1, left_cache_len) and zero hidden/cell
        states shaped (1, hidden_dim).
    """

    cached_left_pad = torch.zeros(1, left_cache_len, dtype=torch.float32, device=device)
    h_state = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    c_state = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)

    return cached_left_pad, h_state, c_state
