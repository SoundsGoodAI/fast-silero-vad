#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""
ONNX export helpers for the standalone Silero-compatible VAD model.

This module exports the reconstructed PyTorch VAD into a CPU ONNX Runtime
artifact. The public entry point writes build provenance, creates a raw
streaming VAD ONNX graph, and lets ONNX Runtime save an optimized vad.onnx
file into the model bundle directory.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from ..constants import (
    ONNX_OPSET_VERSION,
    VAD_CUSTOM_OP,
    VAD_CUSTOM_OP_DOMAIN,
    VAD_INPUT_NAMES,
    VAD_ONNX_FILE,
    VAD_OUTPUT_NAMES,
)
from ..utils import make_version_file
from .model.vad_model import get_init_states


def export_vad_to_onnx(
    vad: torch.nn.Module,
    vad_params: dict[str, int],
    opts: Namespace,
    workdir: Path,
    custom_op_library_path: str | None,
) -> None:
    """Produce the optimized VAD ONNX artifact.

    The function writes VERSION, exports vad-no-opt.onnx, asks ONNX Runtime to
    serialize an optimized vad.onnx, and removes the raw graph unless
    opts.debug is enabled.

    Parameters
    ----------
    vad : torch.nn.Module
        Exportable VAD module.
    vad_params : dict[str, int]
        Constructor parameters inferred from the Silero checkpoint.
    opts : Namespace
        Parsed packaging options. The debug flag controls temporary-file
        cleanup.
    workdir : Path
        Model directory where VERSION and vad.onnx should be written.
    custom_op_library_path : str | None
        Optional ONNX Runtime custom-op library. When provided, the exported
        graph is rewritten to use ``VadFrontend`` for the STFT/frontend
        part before the final ONNX Runtime optimization pass.
    """

    onnxruntime.set_default_logger_severity(4)
    make_version_file(
        str(onnxruntime.__version__), ONNX_OPSET_VERSION, opts, f"{workdir}/VERSION"
    )
    vad_init_states = get_init_states(
        vad_params["context_samples"], vad_params["hidden_dim2"], torch.device("cpu")
    )
    input_vad = (
        torch.ones(
            1,
            vad_params["model_samplerate"],
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        *vad_init_states,
    )
    torch.onnx.export(
        vad,
        input_vad,
        f"{workdir}/vad-no-opt.onnx",
        dynamo=False,
        dynamic_axes={
            "input_vad": {1: "input_vad_len"},
            "output_vad": {1: "output_vad_len"},
        },
        input_names=[
            "input_vad",
            "cached_left_context",
            "h_recurrent_state",
            "c_recurrent_state",
        ],
        output_names=[
            "output_vad",
            "output_cached_left_context",
            "output_h_recurrent_state",
            "output_c_recurrent_state",
        ],
        opset_version=ONNX_OPSET_VERSION,
    )

    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1
    session_opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.optimized_model_filepath = str(workdir / VAD_ONNX_FILE)

    if custom_op_library_path is not None:
        session_opts.optimized_model_filepath = f"{workdir}/vad-no-custom-no-opt.onnx"
        onnxruntime.InferenceSession(
            f"{workdir}/vad-no-opt.onnx",
            session_opts,
            providers=["CPUExecutionProvider"],
        )
        replace_frontend_with_custom_op(
            f"{workdir}/vad-no-custom-no-opt.onnx",
            f"{workdir}/vad-custom-no-opt.onnx",
            vad_params,
        )

        session_opts.register_custom_ops_library(custom_op_library_path)
        session_opts.optimized_model_filepath = str(workdir / VAD_ONNX_FILE)
        onnxruntime.InferenceSession(
            f"{workdir}/vad-custom-no-opt.onnx",
            session_opts,
            providers=["CPUExecutionProvider"],
        )
        verify_onnx_equivalence(
            workdir / "vad-no-custom-no-opt.onnx",
            workdir / VAD_ONNX_FILE,
            custom_op_library_path,
            vad_params,
        )

        if not opts.debug:
            Path(f"{workdir}/vad-no-opt.onnx").unlink()
            Path(f"{workdir}/vad-no-custom-no-opt.onnx").unlink()
            Path(f"{workdir}/vad-custom-no-opt.onnx").unlink()

    else:
        onnxruntime.InferenceSession(
            f"{workdir}/vad-no-opt.onnx",
            session_opts,
            providers=["CPUExecutionProvider"],
        )
        if not opts.debug:
            Path(f"{workdir}/vad-no-opt.onnx").unlink()

    model = onnx.load(workdir / VAD_ONNX_FILE)
    for graph_output in model.graph.output:
        if graph_output.name == "output_cached_left_context":
            graph_output.type.tensor_type.shape.dim[0].dim_value = 1
            graph_output.type.tensor_type.shape.dim[1].dim_value = vad_params[
                "context_samples"
            ]
            break
    onnx.checker.check_model(model)
    onnx.save(model, workdir / VAD_ONNX_FILE)


def get_verification_session(
    model_path: str | Path, custom_op_library_path: str | None = None
) -> onnxruntime.InferenceSession:
    """Create a deterministic single-threaded session for export verification.

    Parameters
    ----------
    model_path : str | Path
        ONNX graph to load.
    custom_op_library_path : str | None
        Custom-op library required by ``model_path``, if any.

    Returns
    -------
    onnxruntime.InferenceSession
        CPU session constrained to one inter-op and one intra-op thread.
    """

    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 4
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1
    if custom_op_library_path is not None:
        session_options.register_custom_ops_library(custom_op_library_path)
    return onnxruntime.InferenceSession(
        str(model_path), session_options, providers=["CPUExecutionProvider"]
    )


def verify_onnx_equivalence(
    reference_model_path: str | Path,
    candidate_model_path: str | Path,
    custom_op_library_path: str,
    vad_params: dict[str, int],
) -> None:
    """Verify that a custom-frontend graph matches the standard ONNX graph.

    Two consecutive inputs exercise multi-chunk inference, final-chunk
    padding, waveform context, and recurrent state propagation. Export fails
    if any output differs beyond normal float32 roundoff.

    Parameters
    ----------
    reference_model_path : str | Path
        Standard ONNX graph produced from the reconstructed JIT branch.
    candidate_model_path : str | Path
        Optimized ONNX graph that uses the custom frontend.
    custom_op_library_path : str
        Shared library implementing the custom frontend operator.
    vad_params : dict[str, int]
        Inferred model dimensions used to construct verification inputs.
    """

    reference_session = get_verification_session(reference_model_path)
    candidate_session = get_verification_session(
        candidate_model_path, custom_op_library_path
    )
    context_samples = vad_params["context_samples"]
    hidden_dim = vad_params["hidden_dim2"]
    reference_states = [
        np.zeros((1, context_samples), dtype=np.float32),
        np.zeros((1, hidden_dim), dtype=np.float32),
        np.zeros((1, hidden_dim), dtype=np.float32),
    ]
    candidate_states = [state.copy() for state in reference_states]

    input_lengths = (
        vad_params["chunk_samples"] * 3 + 17,
        vad_params["chunk_samples"] * 2,
    )
    for input_samples in input_lengths:
        timeline = (
            np.arange(input_samples, dtype=np.float32) / vad_params["model_samplerate"]
        )
        audio = (0.08 * np.sin(2.0 * np.pi * 220.0 * timeline))[np.newaxis, :].astype(
            np.float32
        )

        reference_feed = dict(
            zip(VAD_INPUT_NAMES, [audio, *reference_states], strict=True)
        )
        candidate_feed = dict(
            zip(VAD_INPUT_NAMES, [audio, *candidate_states], strict=True)
        )
        reference_outputs = reference_session.run(
            list(VAD_OUTPUT_NAMES), reference_feed
        )
        candidate_outputs = candidate_session.run(
            list(VAD_OUTPUT_NAMES), candidate_feed
        )

        for name, reference, candidate in zip(
            VAD_OUTPUT_NAMES, reference_outputs, candidate_outputs, strict=True
        ):
            if name == "output_vad":
                relative_tolerance, absolute_tolerance = 1e-4, 1e-6
            elif name == "output_cached_left_context":
                relative_tolerance, absolute_tolerance = 0.0, 0.0
            else:
                # Different float32 FFT arithmetic orders can accumulate in
                # recurrent state while probabilities remain nearly identical.
                relative_tolerance, absolute_tolerance = 1e-3, 2e-4
            np.testing.assert_allclose(
                candidate,
                reference,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                err_msg=f"Custom ONNX output {name} differs from the standard graph.",
            )

        reference_states = reference_outputs[1:]
        candidate_states = candidate_outputs[1:]


def get_producer_index(nodes: list[onnx.NodeProto], value_name: str) -> int:
    """Return the node index that produces an ONNX graph value.

    Parameters
    ----------
    nodes : list[onnx.NodeProto]
        ONNX nodes to search in graph order.
    value_name : str
        Intermediate or output value name whose producer should be found.

    Returns
    -------
    int
        Index of the first node that lists ``value_name`` in its outputs.

    Raises
    ------
    ValueError
        If no node in ``nodes`` produces ``value_name``.
    """

    for i, node in enumerate(nodes):
        if value_name in node.output:
            return i
    raise ValueError(f"Could not find producer for {value_name}.")


def replace_frontend_with_custom_op(
    input_onnx_path: str,
    output_onnx_path: str,
    vad_params: dict[str, int],
) -> None:
    """Replace the exported Silero STFT/frontend subgraph with one custom op.

    The reconstructed PyTorch VAD exports the Silero frontend as ordinary ONNX
    operators. For the optimized CPU bundle, this function prunes that frontend
    and inserts ``com.soundsgoodai::VadFrontend`` with two outputs: frontend
    features and updated cached left context. The remaining encoder and LSTM
    graph stays unchanged.

    Parameters
    ----------
    input_onnx_path : str
        Source ONNX graph after the first ONNX Runtime optimization pass.
    output_onnx_path : str
        Destination ONNX graph containing the custom frontend node.
    vad_params : dict[str, int]
        VAD constructor parameters inferred from the Silero checkpoint. The
        frontend feature shape and cached-context shape are restored from these
        values after graph surgery.

    Raises
    ------
    ValueError
        Raised when the expected exported frontend nodes cannot be found.
    """

    model = onnx.load(input_onnx_path)
    nodes = list(model.graph.node)
    feature_input_name = None
    for node in nodes:
        if "conv1.weight" in node.input:
            feature_input_name = node.input[0]
            break

    if feature_input_name is None:
        raise ValueError("Could not find conv1.weight consumer in VAD ONNX graph.")

    removed_node_indexes = {
        get_producer_index(nodes, feature_input_name),
        get_producer_index(nodes, "output_cached_left_context"),
    }

    for node in nodes:
        for i, input_name in enumerate(node.input):
            if input_name == feature_input_name:
                node.input[i] = "input_features"

    custom_node = onnx.helper.make_node(
        VAD_CUSTOM_OP,
        inputs=["input_vad", "cached_left_context"],
        outputs=["input_features", "output_cached_left_context"],
        domain=VAD_CUSTOM_OP_DOMAIN,
    )
    model.graph.node.clear()
    model.graph.node.extend(
        [custom_node]
        + [node for i, node in enumerate(nodes) if i not in removed_node_indexes]
    )

    # Prune dead nodes.
    required = {output.name for output in model.graph.output}
    kept_nodes = []
    for node in reversed(model.graph.node):
        if any(output_name in required for output_name in node.output):
            kept_nodes.append(node)
            required.update(input_name for input_name in node.input if input_name)

    model.graph.node.clear()
    model.graph.node.extend(reversed(kept_nodes))
    model.opset_import.extend([onnx.helper.make_opsetid(VAD_CUSTOM_OP_DOMAIN, 1)])

    # Prune unused initializers.
    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_inputs
    ]

    model.graph.initializer.clear()
    model.graph.initializer.extend(initializers)

    # Update custom frontend shapes.
    model.graph.value_info.clear()
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            "input_features",
            onnx.TensorProto.FLOAT,
            ["num_chunks", vad_params["cutoff"], 4],
        )
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_onnx_path)
