#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "silero.h"
#include "wav.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kChunkDurationMs = 32;
constexpr std::size_t kChunkSamples = kSampleRate * kChunkDurationMs / 1000;

struct Measurement {
  double best_seconds;
  double median_seconds;
  double rtfx;
};

Measurement Measure(silero::VadIterator& vad, std::vector<float>& audio,
                    double duration_seconds, int warmup, int repeats) {
  for (int repeat = 0; repeat < warmup; ++repeat) {
    vad.SpeechProbs(audio);
  }

  std::vector<double> elapsed_values;
  elapsed_values.reserve(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    vad.SpeechProbs(audio);
    const auto end = std::chrono::steady_clock::now();
    elapsed_values.push_back(
        std::chrono::duration<double>(end - start).count());
  }

  std::sort(elapsed_values.begin(), elapsed_values.end());
  const double best = elapsed_values.front();
  const std::size_t middle = elapsed_values.size() / 2;
  const double median = elapsed_values.size() % 2 == 0
                            ? (elapsed_values[middle - 1] +
                               elapsed_values[middle]) /
                                  2.0
                            : elapsed_values[middle];
  return {best, median, duration_seconds / median};
}

void PrintUsage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " OFFICIAL_ONNX OUTPUT_TSV WARMUP REPEATS WAV_PATH"
               " WAV_OFFSET_SEC DURATION_MS [DURATION_MS ...]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 8) {
    PrintUsage(argv[0]);
    return 2;
  }

  try {
    const std::string model_path = argv[1];
    const std::string output_path = argv[2];
    const int warmup = std::stoi(argv[3]);
    const int repeats = std::stoi(argv[4]);
    const std::string wav_path = argv[5];
    const double wav_offset_seconds = std::stod(argv[6]);
    if (warmup < 0 || repeats <= 0) {
      throw std::invalid_argument(
          "warmup must be non-negative and repeats must be positive");
    }
    if (!std::isfinite(wav_offset_seconds) || wav_offset_seconds < 0.0) {
      throw std::invalid_argument("WAV offset must be finite and non-negative");
    }

    std::vector<int> durations_ms;
    durations_ms.reserve(argc - 7);
    for (int argument = 7; argument < argc; ++argument) {
      const int duration_ms = std::stoi(argv[argument]);
      if (duration_ms <= 0 || duration_ms % kChunkDurationMs != 0) {
        throw std::invalid_argument(
            "duration must be a positive multiple of 32 ms, got " +
            std::to_string(duration_ms));
      }
      durations_ms.push_back(duration_ms);
    }

    wav::WavReader wav_reader(wav_path);
    if (wav_reader.num_channel() != 1 ||
        wav_reader.sample_rate() != kSampleRate ||
        wav_reader.bits_per_sample() != 16) {
      throw std::runtime_error("Expected mono 16-bit PCM WAV at 16 kHz");
    }

    silero::VadIterator vad(model_path);
    vad.sample_rate = kSampleRate;
    vad.SetVariables();

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("Cannot open output TSV: " + output_path);
    }
    output << "language\tbenchmark\taudio_duration_ms\taudio_duration_sec\t"
              "best_sec\tmedian_sec\trtfx\tspeedup\toutputs\tonnxruntime\t"
              "compiler\tbuild\twarmup\trepeats\n";

    const std::size_t offset_samples = static_cast<std::size_t>(
        std::llround(wav_offset_seconds * kSampleRate));
    for (const int duration_ms : durations_ms) {
      const std::size_t output_count = duration_ms / kChunkDurationMs;
      const std::size_t sample_count = output_count * kChunkSamples;
      if (offset_samples + sample_count >
          static_cast<std::size_t>(wav_reader.num_samples())) {
        throw std::runtime_error(
            "WAV does not contain the requested " +
            std::to_string(duration_ms) + " ms excerpt at offset " +
            std::to_string(wav_offset_seconds) + " s");
      }
      std::vector<float> audio(wav_reader.data() + offset_samples,
                               wav_reader.data() + offset_samples + sample_count);
      const double duration_seconds =
          static_cast<double>(sample_count) / kSampleRate;
      const Measurement measurement =
          Measure(vad, audio, duration_seconds, warmup, repeats);

      output << "cpp\tofficial_silero_onnx\t" << duration_ms << '\t'
             << std::fixed << std::setprecision(6) << duration_seconds << '\t'
             << std::setprecision(9)
             << measurement.best_seconds << '\t' << measurement.median_seconds
             << '\t' << std::setprecision(6) << measurement.rtfx << "\t1\t"
             << output_count << '\t'
             << OrtGetApiBase()->GetVersionString() << '\t' << __VERSION__
             << "\t-std=c++20 -O3 -DNDEBUG\t" << warmup << '\t' << repeats
             << '\n';
    }
    if (!output) {
      throw std::runtime_error("Failed to write output TSV: " + output_path);
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
