// Fast parquet writer for the two fixed output schemas, producing the same bytes as
// parquet::arrow::WriteTable with pyarrow.parquet.write_table's defaults (see pqfast.cpp). It has
// no Arrow / libparquet dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "row_feed.h"

namespace fastsim {

// A column, read in place: value i is at data + i * stride (stride 0 = packed), so columns can
// be fields of an array of records.
struct PqColumn {
  enum Kind { I64, I32, STR } kind;
  const void* data = nullptr;  // I64: int64; I32: int32; STR: codes (index into names) of code_bytes
  size_t stride = 0;
  int code_bytes = 4;  // STR: 1 or 4
  const char* const* names = nullptr;
  int n_names = 0;
  const uint8_t* valid = nullptr;  // nonzero = valid; nullptr = no nulls
  size_t valid_stride = 1;
  // derived I64 columns (data unused): SEQ: value = row index; DIFF: cond ? a - b : 0, with a, b,
  // cond read like data
  enum Derive { NONE, SEQ, DIFF } derive = NONE;
  const void* a = nullptr;
  const void* b = nullptr;
  const uint8_t* cond = nullptr;
};

// Which file: selects the column names and the constant footer parts (pq_footer_consts.h).
enum class PqSchema { TRACE, LEDGER };

// Growable byte buffer without zero-fill.
class ByteBuf {
 public:
  ByteBuf() = default;
  ByteBuf(const ByteBuf&) = delete;
  ByteBuf& operator=(const ByteBuf&) = delete;
  ByteBuf(ByteBuf&& o) noexcept : p_(o.p_), n_(o.n_), cap_(o.cap_) { o.p_ = nullptr; o.n_ = o.cap_ = 0; }
  ByteBuf& operator=(ByteBuf&& o) noexcept {
    if (this != &o) { std::free(p_); p_ = o.p_; n_ = o.n_; cap_ = o.cap_; o.p_ = nullptr; o.n_ = o.cap_ = 0; }
    return *this;
  }
  ~ByteBuf() { std::free(p_); }
  void reserve(size_t c) {
    if (c <= cap_) return;
    p_ = (uint8_t*)std::realloc(p_, c);
    if (!p_) throw std::bad_alloc();
    cap_ = c;
  }
  void append(const void* d, size_t k) {
    if (n_ + k > cap_) reserve(std::max(n_ + k, cap_ * 2 + 256));
    std::memcpy(p_ + n_, d, k);
    n_ += k;
  }
  void push(uint8_t b) {
    if (n_ == cap_) reserve(cap_ * 2 + 256);
    p_[n_++] = b;
  }
  uint8_t* data() { return p_; }
  const uint8_t* data() const { return p_; }
  size_t size() const { return n_; }
  void resize(size_t k) { reserve(k); n_ = k; }  // new bytes uninitialised
  void clear() { n_ = 0; }

 private:
  uint8_t* p_ = nullptr;
  size_t n_ = 0, cap_ = 0;
};

// An encoded file, held as a list of segments (page headers and metadata in one buffer, page
// bodies where the encoders left them), so it is written with writev and hashed without first
// being copied into one contiguous image.
class PqImage {
 public:
  PqImage();
  ~PqImage();
  PqImage(const PqImage&) = delete;
  PqImage& operator=(const PqImage&) = delete;
  size_t size() const;
  void copy_to(ByteBuf& out) const;  // contiguous image (tests)
  // Writes the file (created or truncated) and its SHA-256 hex digest (if sha != nullptr).
  bool write_file(const std::string& path, std::string* sha, std::string& err) const;
  struct Impl;
  Impl* impl;
};

// Serialises a table of `n` rows (columns in schema order, all nullable as in writer.cpp),
// encoding column chunks on up to `threads` threads. Returns false with `err` set if anything is
// unexpected; the caller then falls back to the Python implementation.
bool pq_write_fast(PqSchema schema, const std::vector<PqColumn>& cols, int64_t n, PqImage* out, std::string& err,
                   int threads = 1);

// Encodes a table whose rows are records in a growing array (RowFeed) while it grows, on
// `threads` threads started by the constructor: complete batches of each column are encoded as
// soon as they are published. Columns are given as for pq_write_fast, but with pointers that are
// offsets from the records' base. finish() waits for feed.done and the last rows, then lays the
// file out; the bytes are those pq_write_fast would produce for the final table.
class PqStream {
 public:
  PqStream(PqSchema schema, const std::vector<PqColumn>& rel_cols, const RowFeed& feed, int threads);
  // one feed per column (same records; a column may be published later than the others, but all
  // end at the same count)
  PqStream(PqSchema schema, const std::vector<PqColumn>& rel_cols, const std::vector<const RowFeed*>& feeds,
           int threads);
  ~PqStream();  // joins (every feed must be done by then)
  PqStream(const PqStream&) = delete;
  PqStream& operator=(const PqStream&) = delete;
  bool finish(PqImage* out, std::string& err);
  struct Impl;
  Impl* impl;
};

}  // namespace fastsim
