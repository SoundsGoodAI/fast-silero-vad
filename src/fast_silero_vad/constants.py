"""Constants shared by the Fast Silero VAD runtime and exporter."""

FAST_SILERO_VAD_VERSION = "0.1.0"

MODEL_TYPE_STREAMING_VAD = "streaming_vad"
MODEL_TYPE_OFFLINE_VAD = "offline_vad"
VAD_MODEL_TYPES = (MODEL_TYPE_STREAMING_VAD, MODEL_TYPE_OFFLINE_VAD)

VAD_CONFIG_FILE = "model_config.yaml"
VAD_ONNX_FILE = "vad.onnx"
VAD_MODEL_TARBALL = "model.tgz"
VAD_PACKAGING_LOG = "package.log"

VAD_INPUT_NAMES = (
    "input_vad",
    "cached_left_context",
    "h_recurrent_state",
    "c_recurrent_state",
)
VAD_OUTPUT_NAMES = (
    "output_vad",
    "output_cached_left_context",
    "output_h_recurrent_state",
    "output_c_recurrent_state",
)

ONNX_OPSET_VERSION = 20
VAD_CONVOLUTION_STRIDE = 2
VAD_BRANCHES = {"16k": "_model.", "8k": "_model_8k."}

VAD_CUSTOM_OP = "VadFrontend"
VAD_CUSTOM_OP_DOMAIN = "com.soundsgoodai"
VAD_ORT_HEADER_PATHS = (
    "onnxruntime/core/session/onnxruntime_c_api.h",
    "onnxruntime/core/session/onnxruntime_ep_c_api.h",
    "onnxruntime/core/session/onnxruntime_cxx_api.h",
    "onnxruntime/core/session/onnxruntime_cxx_inline.h",
    "onnxruntime/core/session/onnxruntime_float16.h",
)
