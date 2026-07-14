#define ORT_API_MANUAL_INIT
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

constexpr size_t kFramesPerChunk = 4;
constexpr float kPi = 3.14159265358979323846F;

struct Complex {
  float real;
  float imag;
};

template <size_t Size>
std::array<float, Size> make_hann_window() {
  std::array<float, Size> window{};
  for (size_t index = 0; index < Size; ++index) {
    const float angle =
        2.0F * kPi * static_cast<float>(index) / static_cast<float>(Size);
    window[index] = 0.5F - 0.5F * std::cos(angle);
  }
  return window;
}

template <size_t Size>
const std::array<float, Size>& hann_window() {
  static const std::array<float, Size> window = make_hann_window<Size>();
  return window;
}

template <size_t Size>
std::array<int, Size> make_bit_reverse() {
  std::array<int, Size> indices{};
  int bits = 0;
  for (size_t size = Size; size > 1; size >>= 1) {
    ++bits;
  }
  for (int index = 0; index < static_cast<int>(Size); ++index) {
    int reversed = 0;
    int value = index;
    for (int bit = 0; bit < bits; ++bit) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    indices[static_cast<size_t>(index)] = reversed;
  }
  return indices;
}

template <size_t Size>
const std::array<int, Size>& bit_reverse_indices() {
  static const std::array<int, Size> indices = make_bit_reverse<Size>();
  return indices;
}

template <size_t Size>
std::array<Complex, Size / 2> make_fft_twiddles() {
  std::array<Complex, Size / 2> twiddles{};
  for (size_t index = 0; index < Size / 2; ++index) {
    const float angle =
        -2.0F * kPi * static_cast<float>(index) / static_cast<float>(Size);
    twiddles[index] = Complex{std::cos(angle), std::sin(angle)};
  }
  return twiddles;
}

template <size_t Size>
const std::array<Complex, Size / 2>& fft_twiddles() {
  static const std::array<Complex, Size / 2> twiddles = make_fft_twiddles<Size>();
  return twiddles;
}

template <size_t FftSize>
std::array<Complex, FftSize / 2> make_real_fft_split_twiddles() {
  std::array<Complex, FftSize / 2> twiddles{};
  for (size_t bin = 0; bin < FftSize / 2; ++bin) {
    const float angle =
        -2.0F * kPi * static_cast<float>(bin) / static_cast<float>(FftSize);
    twiddles[bin] = Complex{std::cos(angle), std::sin(angle)};
  }
  return twiddles;
}

template <size_t FftSize>
const std::array<Complex, FftSize / 2>& real_fft_split_twiddles() {
  static const std::array<Complex, FftSize / 2> twiddles =
      make_real_fft_split_twiddles<FftSize>();
  return twiddles;
}

template <size_t Size>
void fft(std::array<Complex, Size>& values) {
  static_assert(Size > 1 && (Size & (Size - 1)) == 0);

  const auto& indices = bit_reverse_indices<Size>();
  const auto& twiddles = fft_twiddles<Size>();
  for (int index = 0; index < static_cast<int>(Size); ++index) {
    const int reversed = indices[static_cast<size_t>(index)];
    if (reversed > index) {
      std::swap(values[static_cast<size_t>(index)],
                values[static_cast<size_t>(reversed)]);
    }
  }

  for (int length = 2; length <= static_cast<int>(Size); length <<= 1) {
    for (int start = 0; start < static_cast<int>(Size); start += length) {
      const int half = length >> 1;
      for (int index = 0; index < half; ++index) {
        Complex& even = values[static_cast<size_t>(start + index)];
        Complex& odd = values[static_cast<size_t>(start + index + half)];
        const Complex twiddle =
            twiddles[static_cast<size_t>(index) * Size /
                     static_cast<size_t>(length)];
        const float odd_real = odd.real * twiddle.real - odd.imag * twiddle.imag;
        const float odd_imag = odd.real * twiddle.imag + odd.imag * twiddle.real;
        const float even_real = even.real;
        const float even_imag = even.imag;
        even.real = even_real + odd_real;
        even.imag = even_imag + odd_imag;
        odd.real = even_real - odd_real;
        odd.imag = even_imag - odd_imag;
      }
    }
  }
}

template <size_t FftSize>
void real_fft_magnitudes(
    const float* input,
    const std::array<float, FftSize>& window,
    std::array<float, FftSize / 2 + 1>& magnitudes) {
  constexpr size_t half_fft_size = FftSize / 2;
  std::array<Complex, half_fft_size> values{};
  const auto& twiddles = real_fft_split_twiddles<FftSize>();
  for (size_t index = 0; index < half_fft_size; ++index) {
    values[index].real = input[2 * index] * window[2 * index];
    values[index].imag = input[2 * index + 1] * window[2 * index + 1];
  }

  fft(values);

  magnitudes[0] = std::abs(values[0].real + values[0].imag);
  magnitudes[half_fft_size] = std::abs(values[0].real - values[0].imag);

  for (size_t bin = 1; bin < half_fft_size; ++bin) {
    const Complex a = values[bin];
    const Complex mirror = values[half_fft_size - bin];
    const Complex b{mirror.real, -mirror.imag};
    const float even_real = 0.5F * (a.real + b.real);
    const float even_imag = 0.5F * (a.imag + b.imag);
    const float odd_real = 0.5F * (a.imag - b.imag);
    const float odd_imag = 0.5F * (b.real - a.real);
    const Complex twiddle = twiddles[bin];
    const float real =
        even_real + twiddle.real * odd_real - twiddle.imag * odd_imag;
    const float imag =
        even_imag + twiddle.real * odd_imag + twiddle.imag * odd_real;
    magnitudes[bin] = std::sqrt(real * real + imag * imag);
  }
}

template <size_t FftSize, size_t ChunkSamples, size_t ContextSamples>
void compute_frontend(Ort::KernelContext& kernel_context,
                      const Ort::ConstValue& input_value,
                      const Ort::ConstValue& left_context_value,
                      int64_t input_len) {
  static_assert(ChunkSamples == 2 * FftSize);
  static_assert(ContextSamples == ChunkSamples / 8);

  constexpr size_t fft_bins = FftSize / 2 + 1;
  constexpr size_t frame_step = FftSize / 2;
  constexpr size_t segment_samples = ContextSamples + ChunkSamples + ContextSamples;

  const int64_t num_chunks =
      (input_len + static_cast<int64_t>(ChunkSamples) - 1) /
      static_cast<int64_t>(ChunkSamples);
  const std::array<int64_t, 3> features_shape{
      num_chunks, static_cast<int64_t>(fft_bins),
      static_cast<int64_t>(kFramesPerChunk)};
  const std::array<int64_t, 2> left_shape_out{
      1, static_cast<int64_t>(ContextSamples)};

  Ort::UnownedValue features_value = kernel_context.GetOutput(
      0, features_shape.data(), features_shape.size());
  Ort::UnownedValue left_output_value = kernel_context.GetOutput(
      1, left_shape_out.data(), left_shape_out.size());

  const float* input = input_value.GetTensorData<float>();
  const float* left_context = left_context_value.GetTensorData<float>();
  float* features = features_value.GetTensorMutableData<float>();
  float* left_output = left_output_value.GetTensorMutableData<float>();

  const auto& window = hann_window<FftSize>();
  std::array<float, segment_samples> segment{};
  std::array<float, fft_bins> magnitudes{};

  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    const int64_t chunk_start = chunk * static_cast<int64_t>(ChunkSamples);
    if (chunk == 0) {
      std::copy_n(left_context, ContextSamples, segment.begin());
    } else {
      std::copy_n(input + chunk_start - ContextSamples, ContextSamples,
                  segment.begin());
    }

    const int64_t available_samples = std::min(
        static_cast<int64_t>(ChunkSamples), input_len - chunk_start);
    std::copy_n(input + chunk_start, available_samples,
                segment.begin() + ContextSamples);
    std::fill(segment.begin() + ContextSamples + available_samples,
              segment.begin() + ContextSamples + ChunkSamples, 0.0F);

    for (size_t index = 0; index < ContextSamples; ++index) {
      segment[ContextSamples + ChunkSamples + index] =
          segment[ContextSamples + ChunkSamples - 2 - index];
    }

    float* chunk_features =
        features + chunk * static_cast<int64_t>(fft_bins * kFramesPerChunk);
    for (size_t frame = 0; frame < kFramesPerChunk; ++frame) {
      real_fft_magnitudes<FftSize>(
          segment.data() + frame * frame_step, window, magnitudes);
      for (size_t bin = 0; bin < fft_bins; ++bin) {
        chunk_features[bin * kFramesPerChunk + frame] = magnitudes[bin];
      }
    }
  }

  const int64_t last_chunk_start =
      (num_chunks - 1) * static_cast<int64_t>(ChunkSamples);
  const int64_t tail_start =
      last_chunk_start + static_cast<int64_t>(ChunkSamples - ContextSamples);
  const int64_t available_tail = std::max<int64_t>(
      0, std::min(static_cast<int64_t>(ContextSamples), input_len - tail_start));
  std::copy_n(input + std::min(tail_start, input_len), available_tail, left_output);
  std::fill(left_output + available_tail, left_output + ContextSamples, 0.0F);
}

struct VadFrontendKernel {
  VadFrontendKernel(const OrtApi&, const OrtKernelInfo*) {}

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext kernel_context(context);
    const Ort::ConstValue input_value = kernel_context.GetInput(0);
    const Ort::ConstValue left_context_value = kernel_context.GetInput(1);

    const auto input_shape = input_value.GetTensorTypeAndShapeInfo().GetShape();
    const auto left_shape = left_context_value.GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 2 || input_shape[0] != 1) {
      throw std::runtime_error("VadFrontend expects input_vad shape (1, num_samples).");
    }
    if (left_shape.size() != 2 || left_shape[0] != 1 ||
        (left_shape[1] != 32 && left_shape[1] != 64)) {
      throw std::runtime_error(
          "VadFrontend expects cached_left_context shape (1, 32) or (1, 64).");
    }

    const int64_t input_len = input_shape[1];
    if (input_len < 1) {
      throw std::runtime_error("VadFrontend expects at least one input sample.");
    }

    if (left_shape[1] == 64) {
      compute_frontend<256, 512, 64>(
          kernel_context, input_value, left_context_value, input_len);
    } else {
      compute_frontend<128, 256, 32>(
          kernel_context, input_value, left_context_value, input_len);
    }
  }
};

struct VadFrontendOp : Ort::CustomOpBase<VadFrontendOp, VadFrontendKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new VadFrontendKernel(api, info);
  }

  const char* GetName() const {
    return "VadFrontend";
  }

  size_t GetInputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

struct VadCustomOps {
  VadCustomOps() : domain("com.soundsgoodai") {
    domain.Add(&op);
  }

  VadFrontendOp op;
  Ort::CustomOpDomain domain;
};

VadCustomOps& vad_custom_ops() {
  static VadCustomOps custom_ops;
  return custom_ops;
}

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::InitApi(api->GetApi(ORT_API_VERSION));

  try {
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(vad_custom_ops().domain);
  } catch (const std::exception& error) {
    Ort::Status status{error};
    return status.release();
  }

  return nullptr;
}
