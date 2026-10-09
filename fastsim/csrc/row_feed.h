// A growing array of records shared by a producer (the event loop) and consumers (encoder
// threads). Records [0, avail) are final; `base` is where they currently are (a buffer that grows
// moves, but the producer keeps the old buffers alive until the consumers are done, so a base
// read together with an avail is always valid for it). `done` is set, after the last records are
// published, when no more will come.
#pragma once
#include <atomic>
#include <cstdint>

namespace fastsim {

struct RowFeed {
  std::atomic<const uint8_t*> base{nullptr};
  std::atomic<int64_t> avail{0};
  std::atomic<bool> done{false};
  void publish(const void* b, int64_t n) {
    base.store((const uint8_t*)b, std::memory_order_relaxed);
    avail.store(n, std::memory_order_release);
  }
};

}  // namespace fastsim
