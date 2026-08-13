#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace vc2hq_stage_profile {

// Plain namespace-scope flag (initialised before main) so enabled() is a single
// load. A function-local static would carry MSVC's thread-safe-init guard
// (_Init_thread_header) on every hot-path Scope call.
static const bool vc2hq_profile_enabled = (std::getenv("VC2HQ_PROFILE") != nullptr);

enum Stage {
  INPUT_HORIZONTAL,
  VERTICAL_WAVELET,
  HORIZONTAL_WAVELET,
  QUANTISE_SEARCH,
  QUANTISE_ENCODE,
  SERIALISE,
  STAGE_COUNT
};

inline std::atomic<long long> *elapsed_ns() {
  static std::atomic<long long> values[STAGE_COUNT]{};
  return values;
}

inline std::atomic<unsigned long long> *calls() {
  static std::atomic<unsigned long long> values[STAGE_COUNT]{};
  return values;
}

inline bool enabled() {
  return vc2hq_profile_enabled;
}

inline void report() {
  static const char *names[STAGE_COUNT] = {
    "input+horizontal-L0",
    "vertical-wavelet",
    "horizontal-wavelet-L1+",
    "quantiser-search",
    "quantise+encode",
    "serialise"
  };
  if (!enabled())
    return;
  long long total = 0;
  for (int i = 0; i < STAGE_COUNT; ++i)
    total += elapsed_ns()[i].load(std::memory_order_relaxed);
  std::fprintf(stderr, "VC2HQ_STAGE_PROFILE,encoder,total_thread_ms,%.3f\n", total / 1.0e6);
  for (int i = 0; i < STAGE_COUNT; ++i) {
    const long long ns = elapsed_ns()[i].load(std::memory_order_relaxed);
    const double percent = total ? 100.0 * static_cast<double>(ns) / total : 0.0;
    std::fprintf(stderr, "VC2HQ_STAGE_PROFILE,encoder,%s,%.3f,%.2f,%llu\n",
                 names[i], ns / 1.0e6, percent,
                 calls()[i].load(std::memory_order_relaxed));
  }
}

inline void ensure_report_registered() {
  static const bool registered = []() { std::atexit(report); return true; }();
  (void)registered;
}

class Scope {
public:
  explicit Scope(Stage stage)
    : mStage(stage), mActive(enabled()), mStart() {
    if (mActive) {
      ensure_report_registered();
      mStart = std::chrono::steady_clock::now();
    }
  }

  ~Scope() {
    if (!mActive)
      return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - mStart).count();
    elapsed_ns()[mStage].fetch_add(ns, std::memory_order_relaxed);
    calls()[mStage].fetch_add(1, std::memory_order_relaxed);
  }

private:
  Stage mStage;
  bool mActive;
  std::chrono::steady_clock::time_point mStart;
};

} // namespace vc2hq_stage_profile
