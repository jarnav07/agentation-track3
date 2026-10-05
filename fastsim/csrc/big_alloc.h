// Allocator for the large per-run buffers (trace/ledger records and output columns).
//
// Allocations of 2 MiB and up are their own anonymous mappings (what malloc does at these sizes
// anyway, made explicit so the engine's pre-fault helper can work on whole pages of them). Not
// MADV_HUGEPAGE: huge-page faults can stall on compaction for milliseconds, which showed up as a
// long tail in run times.
//
// construct() with no arguments default-initialises (no zero fill for scalars): every column is
// written in full right after resize(), and fresh mappings are zero already.
#pragma once
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

namespace fastsim {

inline size_t big_len(size_t bytes) { return (bytes + 4095) & ~(size_t)4095; }
inline void* big_map(size_t bytes) {
  void* p = mmap(nullptr, big_len(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) throw std::bad_alloc();
  return p;
}
inline void big_unmap(void* p, size_t bytes) { munmap(p, big_len(bytes)); }

template <typename T>
struct BigAlloc {
  using value_type = T;
  static constexpr size_t kMapMin = (size_t)2 << 20;
  BigAlloc() noexcept = default;
  template <typename U>
  BigAlloc(const BigAlloc<U>&) noexcept {}
  T* allocate(size_t n) {
    const size_t bytes = n * sizeof(T);
    if (bytes >= kMapMin) return static_cast<T*>(big_map(bytes));
    void* p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, size_t n) noexcept {
    const size_t bytes = n * sizeof(T);
    if (bytes >= kMapMin) big_unmap(p, bytes);
    else std::free(p);
  }
  template <typename U>
  void construct(U* p) noexcept {
    ::new ((void*)p) U;
  }
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new ((void*)p) U(std::forward<Args>(args)...);
  }
  template <typename U>
  bool operator==(const BigAlloc<U>&) const noexcept { return true; }
  template <typename U>
  bool operator!=(const BigAlloc<U>&) const noexcept { return false; }
};

template <typename T>
using bvec = std::vector<T, BigAlloc<T>>;

// Append-only record buffer for trivially copyable T, with an always-inlined append (the engine
// appends a record per event; std::vector's emplace_back is not reliably inlined for these
// structs). Same subset of the vector interface the engine uses.
template <typename T>
class RecBuf {
 public:
  RecBuf() = default;
  RecBuf(const RecBuf&) = delete;
  RecBuf& operator=(const RecBuf&) = delete;
  ~RecBuf() { if (p_) BigAlloc<T>().deallocate(p_, cap_); }
  void reserve(size_t n) { if (n > cap_) regrow(n); }
  __attribute__((always_inline)) void push_back(const T& v) {
    if (__builtin_expect(n_ == cap_, 0)) regrow(cap_ ? 2 * cap_ : 64);
    p_[n_++] = v;
  }
  size_t size() const { return n_; }
  size_t capacity() const { return cap_; }
  T* data() { return p_; }
  const T* data() const { return p_; }
  T& operator[](size_t i) { return p_[i]; }
  const T& operator[](size_t i) const { return p_[i]; }

 private:
  __attribute__((noinline)) void regrow(size_t n) {
    T* q = BigAlloc<T>().allocate(n);
    if (n_) std::memcpy((void*)q, (const void*)p_, n_ * sizeof(T));
    if (p_) BigAlloc<T>().deallocate(p_, cap_);
    p_ = q;
    cap_ = n;
  }
  T* p_ = nullptr;
  size_t n_ = 0, cap_ = 0;
};

}  // namespace fastsim
