#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Package a Silero-compatible VAD model for runtime inference.

The packager converts one Silero JIT checkpoint branch into the local VAD
module, exports the model to ONNX, writes the runtime configuration, and
creates the final model.tgz bundle.
"""

import logging
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import OrderedDict
from hashlib import file_digest
from pathlib import Path

import torch

from fast_silero_vad.constants import (
    VAD_BRANCHES,
    VAD_CONFIG_FILE,
    VAD_CONVOLUTION_STRIDE,
    VAD_MODEL_TYPES,
    VAD_ONNX_FILE,
    VAD_PACKAGING_LOG,
)
from fast_silero_vad.custom_op.build_vad_custom_op import build_vad_custom_op
from fast_silero_vad.export.export_vad_onnx import export_vad_to_onnx
from fast_silero_vad.model.vad import VAD
from fast_silero_vad.utils import prepare_output_dir, prepare_vad_config, tar_model

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path) -> None:
    """Write packaging logs to the model directory and standard error.

    Parameters
    ----------
    log_path : Path
        Packaging log path inside the output model directory.
    """

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
        level=logging.INFO,
        force=True,
    )
    logger.info("Writing packaging log to %s.", log_path)


def parse_args() -> Namespace:
    """Parse command-line arguments for VAD bundle export.

    Returns
    -------
    Namespace
        Parsed packaging options.
    """

    parser = ArgumentParser(
        description=(
            "Export a Fast Silero VAD bundle from the original Silero JIT checkpoint."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the original Silero VAD JIT checkpoint.",
    )
    parser.add_argument(
        "--output-dir-path",
        type=str,
        required=True,
        help="Output directory for the unpacked bundle and model.tgz.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=VAD_MODEL_TYPES,
        help="Runtime layout to package.",
    )
    parser.add_argument(
        "--vad-branch",
        type=str,
        default="16k",
        choices=tuple(VAD_BRANCHES),
        help="Silero checkpoint branch to export.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Speech probability threshold.",
    )
    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        required=True,
        help="Minimum speech segment duration in milliseconds.",
    )
    parser.add_argument(
        "--max-speech-duration-ms",
        type=int,
        required=True,
        help="Maximum speech segment duration in milliseconds before forced splitting.",
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        required=True,
        help=(
            "Minimum silence duration in milliseconds before closing a speech segment."
        ),
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        required=True,
        help="Padding in milliseconds added around natural VAD boundaries.",
    )
    parser.add_argument(
        "--use-onnxruntime-custom-op",
        action="store_true",
        help="Replace the ONNX frontend with the optimized CPU custom op.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep intermediate ONNX files.",
    )
    return parser.parse_args()


def modify_state_dict(
    model_state_dict: OrderedDict[str, torch.Tensor], vad_branch: str
) -> OrderedDict[str, torch.Tensor]:
    """Convert one Silero JIT state dict branch to the local VAD layout.

    The Silero JIT bundle contains both _model and _model_8k branches.
    This function keeps tensors from the requested branch, drops the other
    branch, and renames modules to match model.vad.vad.VAD.

    Parameters
    ----------
    model_state_dict : OrderedDict[str, torch.Tensor]
        State dict loaded from the Silero JIT checkpoint.
    vad_branch : str
        VAD branch selector, either ``16k`` or ``8k``.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Converted VAD state dict compatible with model.vad.vad.VAD.

    Raises
    ------
    ValueError
        Raised when the selected Silero branch has no matching tensors.
    """

    branch_prefix = VAD_BRANCHES[vad_branch]
    converted_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        if not key.startswith(branch_prefix):
            continue

        key = key.removeprefix(branch_prefix)
        key = key.replace("stft.forward_basis_buffer", "stft_conv.weight")
        key = key.replace("encoder.0.reparam_conv", "conv1")
        key = key.replace("encoder.1.reparam_conv", "conv2")
        key = key.replace("encoder.2.reparam_conv", "conv3")
        key = key.replace("encoder.3.reparam_conv", "conv4")
        key = key.replace("decoder.decoder.2", "depthwise_conv")
        key = key.replace("decoder.rnn.weight_ih", "rnn.weight_ih_l0")
        key = key.replace("decoder.rnn.weight_hh", "rnn.weight_hh_l0")
        key = key.replace("decoder.rnn.bias_ih", "rnn.bias_ih_l0")
        key = key.replace("decoder.rnn.bias_hh", "rnn.bias_hh_l0")
        converted_state_dict[key] = value

    if not converted_state_dict:
        raise ValueError(
            f"No tensors found for VAD branch {vad_branch} with prefix "
            f"{branch_prefix}.",
        )

    return converted_state_dict


def infer_vad_params(
    model_state_dict: OrderedDict[str, torch.Tensor],
) -> dict[str, int | torch.device]:
    """Infer the local VAD constructor parameters from converted Silero weights.

    The JIT checkpoint stores convolution weights but not module attributes such
    as stride, padding, or recurrent context length. The stored shapes are enough
    to recover the Silero layout used by model.vad.vad.VAD:
    the STFT kernel is half a VAD chunk, the right context is one eighth of a
    chunk, the model sample rate follows from Silero's 32 ms chunk duration,
    and the middle encoder convolutions downsample with the configured VAD
    convolution stride.

    Parameters
    ----------
    model_state_dict : OrderedDict[str, torch.Tensor]
        Converted VAD state dict.

    Returns
    -------
    dict[str, int | torch.device]
        Constructor parameters for model.vad.vad.VAD.

    Raises
    ------
    ValueError
        If the checkpoint tensors do not match a supported Silero layout.
    """

    stft_weight = model_state_dict["stft_conv.weight"]
    conv1_weight = model_state_dict["conv1.weight"]
    conv2_weight = model_state_dict["conv2.weight"]
    conv3_weight = model_state_dict["conv3.weight"]
    conv4_weight = model_state_dict["conv4.weight"]
    rnn_weight_ih = model_state_dict["rnn.weight_ih_l0"]
    depthwise_weight = model_state_dict["depthwise_conv.weight"]

    if stft_weight.ndim != 3 or stft_weight.size(1) != 1:
        raise ValueError(
            "Expected stft_conv.weight to have shape "
            f"(2 * cutoff, 1, filter_length), got {tuple(stft_weight.size())}.",
        )
    if stft_weight.size(0) % 2 != 0:
        raise ValueError(
            "Expected stft_conv.weight output channels to be even, got "
            f"{stft_weight.size(0)}.",
        )

    cutoff = stft_weight.size(0) // 2
    chunk_samples = stft_weight.size(2) * 2
    if chunk_samples == 512:
        model_samplerate = 16000
    elif chunk_samples == 256:
        model_samplerate = 8000
    else:
        raise ValueError(
            "Expected inferred Silero VAD chunk size to be 256 or 512, got "
            f"{chunk_samples}.",
        )

    context_samples = chunk_samples // 8
    hidden_dim2 = conv1_weight.size(0)
    hidden_dim1 = conv2_weight.size(0)
    encoder_kernel_size = conv1_weight.size(2)

    expected_shapes = {
        "conv1.weight": (hidden_dim2, cutoff, encoder_kernel_size),
        "conv2.weight": (hidden_dim1, hidden_dim2, encoder_kernel_size),
        "conv3.weight": (hidden_dim1, hidden_dim1, encoder_kernel_size),
        "conv4.weight": (hidden_dim2, hidden_dim1, encoder_kernel_size),
        "rnn.weight_ih_l0": (hidden_dim2 * 4, hidden_dim2),
        "depthwise_conv.weight": (1, hidden_dim2, 1),
    }
    actual_shapes = {
        "conv1.weight": tuple(conv1_weight.shape),
        "conv2.weight": tuple(conv2_weight.shape),
        "conv3.weight": tuple(conv3_weight.shape),
        "conv4.weight": tuple(conv4_weight.shape),
        "rnn.weight_ih_l0": tuple(rnn_weight_ih.shape),
        "depthwise_conv.weight": tuple(depthwise_weight.shape),
    }

    for name, expected_shape in expected_shapes.items():
        if actual_shapes[name] != expected_shape:
            raise ValueError(
                f"Expected {name} to have shape {expected_shape}, got "
                f"{actual_shapes[name]}.",
            )

    vad_params = {
        "model_samplerate": model_samplerate,
        "chunk_samples": chunk_samples,
        "context_samples": context_samples,
        "cutoff": cutoff,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "encoder_kernel_size": encoder_kernel_size,
        "encoder_stride": VAD_CONVOLUTION_STRIDE,
        "encoder_padding": encoder_kernel_size // 2,
        "device": torch.device("cpu"),
    }

    return vad_params


def main(opts: Namespace) -> None:
    """Run the full model packaging pipeline for one checkpoint.

    The pipeline prepares the output directory, converts the checkpoint to the
    local inference graph, exports ONNX artifacts, writes runtime config, and
    creates model.tgz.

    Parameters
    ----------
    opts : Namespace
        Parsed packaging options.
    """

    output_dir_path = prepare_output_dir(opts.output_dir_path)

    configure_logging(output_dir_path / VAD_PACKAGING_LOG)
    logger.info("Prepared output directory: %s.", output_dir_path)
    logger.info(
        "Packaging Silero VAD branch %s as %s.", opts.vad_branch, opts.model_type
    )
    logger.info(
        "Segmenter parameters: threshold=%s, min_speech_duration_ms=%s, "
        "max_speech_duration_ms=%s, min_silence_duration_ms=%s, speech_pad_ms=%s.",
        opts.threshold,
        opts.min_speech_duration_ms,
        opts.max_speech_duration_ms,
        opts.min_silence_duration_ms,
        opts.speech_pad_ms,
    )
    if opts.debug:
        logger.info("Debug mode is enabled; intermediate ONNX graphs will be kept.")

    model_path = Path(opts.model_path)
    logger.info("Hashing source checkpoint: %s.", model_path)
    with open(model_path, "rb") as model_file:
        opts.source_model_sha256 = file_digest(model_file, "sha256").hexdigest()
    logger.info("Source checkpoint SHA-256: %s.", opts.source_model_sha256)

    logger.info("Loading Silero JIT checkpoint.")
    model_state_dict = torch.jit.load(
        model_path, map_location=torch.device("cpu")
    ).state_dict()
    logger.info(
        "Converting source state dict for branch %s to the local VAD layout.",
        opts.vad_branch,
    )
    model_state_dict = modify_state_dict(model_state_dict, opts.vad_branch)
    logger.info("Converted state dict contains %s tensors.", len(model_state_dict))

    logger.info("Initializing local VAD module.")
    vad_params = infer_vad_params(model_state_dict)
    vad_init_params = {
        key: value for key, value in vad_params.items() if key != "model_samplerate"
    }
    vad = VAD(**vad_init_params)
    vad.load_state_dict(model_state_dict)
    vad.eval()

    opts.model_samplerate = vad_params["model_samplerate"]
    opts.onnx_custom_op = None
    logger.info(
        "Inferred VAD architecture: model_samplerate=%s, chunk_samples=%s, "
        "context_samples=%s, cutoff=%s, hidden_dim1=%s, hidden_dim2=%s, "
        "encoder_kernel_size=%s.",
        vad_params["model_samplerate"],
        vad_params["chunk_samples"],
        vad_params["context_samples"],
        vad_params["cutoff"],
        vad_params["hidden_dim1"],
        vad_params["hidden_dim2"],
        vad_params["encoder_kernel_size"],
    )

    custom_op_library_path = None
    if opts.use_onnxruntime_custom_op:
        logger.info("Building ONNX Runtime custom op in %s.", output_dir_path)
        custom_op_library_path = build_vad_custom_op(output_dir_path)
        opts.onnx_custom_op = Path(custom_op_library_path).name
        logger.info("Built ONNX Runtime custom op: %s.", custom_op_library_path)
    else:
        logger.info("Using the standard ONNX frontend without a custom op.")

    logger.info(
        "Exporting optimized ONNX model to %s.", output_dir_path / VAD_ONNX_FILE
    )
    export_vad_to_onnx(vad, vad_params, opts, output_dir_path, custom_op_library_path)
    logger.info("Writing runtime config: %s.", output_dir_path / VAD_CONFIG_FILE)
    prepare_vad_config(opts, output_dir_path)

    licenses_dir = Path(__file__).resolve().parents[1] / "licenses"
    logger.info("Copying license files from %s.", licenses_dir)
    for source_name, output_name in (
        ("APACHE-2.0.txt", "LICENSE"),
        ("NOTICE", "NOTICE"),
        ("SILERO-VAD-MIT.txt", "SILERO-VAD-LICENSE.txt"),
    ):
        shutil.copyfile(licenses_dir / source_name, output_dir_path / output_name)
    logger.info("Copied license files.")

    tar_path = tar_model(output_dir_path)
    logger.info("Created model archive: %s.", tar_path)
    logger.info("Finished VAD bundle packaging: %s.", output_dir_path)


def main_cli() -> None:
    """Parse CLI arguments and package one Fast Silero VAD bundle."""

    main(parse_args())


if __name__ == "__main__":
    main_cli()
