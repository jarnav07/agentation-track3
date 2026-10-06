// RLE / bit-packed hybrid encoder with exactly the run decisions of arrow::util::RleEncoder
// (cpp/src/arrow/util/rle_encoding.h, apache-arrow-15.0.2), which parquet-cpp uses for dictionary
// indices and definition levels: groups of 8 values; a repeated run once 8 equal values are seen;
// literal runs of at most 63 groups; the final partial group zero-padded. Same state machine, but
// fully inline and writing into a growable byte vector. tests/test_rle_fast.cpp checks it against
// arrow's encoder byte for byte.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace fastsim {

class RleFast {
 public:
  RleFast(std::vector<uint8_t>& out, int bit_width) : out_(out), bw_(bit_width) {}

  inline void put(uint64_t value) {
    if (value == current_) {
      if (++repeat_ > 8) return;
    } else {
      if (repeat_ >= 8) flush_repeated();
      repeat_ = 1;
      current_ = value;
    }
    buf_[nbuf_] = value;
    if (++nbuf_ == 8) flush_buffered(false);
  }

  // Flushes pending values; returns the encoded length (RleEncoder::Flush).
  int flush() {
    if (literal_ > 0 || repeat_ > 0 || nbuf_ > 0) {
      const bool all_repeat = literal_ == 0 && (repeat_ == nbuf_ || nbuf_ == 0);
      if (repeat_ > 0 && all_repeat) {
        flush_repeated();
      } else {
        for (; nbuf_ != 0 && nbuf_ < 8; ++nbuf_) buf_[nbuf_] = 0;
        literal_ += nbuf_;
        flush_literal(true);
        repeat_ = 0;
      }
    }
    return (int)out_.size();
  }

 private:
  std::vector<uint8_t>& out_;
  const int bw_;
  uint64_t current_ = 0;
  int repeat_ = 0, nbuf_ = 0, literal_ = 0;
  uint64_t buf_[8];
  long indicator_ = -1;  // offset of the reserved literal indicator byte

  void flush_repeated() {
    uint32_t v = (uint32_t)repeat_ << 1;
    while (v & 0xFFFFFF80u) {
      out_.push_back((uint8_t)((v & 0x7F) | 0x80));
      v >>= 7;
    }
    out_.push_back((uint8_t)v);
    const int nbytes = (bw_ + 7) / 8;
    uint64_t cv = current_;
    for (int i = 0; i < nbytes; i++) {
      out_.push_back((uint8_t)cv);
      cv >>= 8;
    }
    nbuf_ = 0;
    repeat_ = 0;
  }

  // Bit-pack the buffered values (always whole groups of 8 here, so byte aligned).
  void flush_literal(bool update_indicator) {
    if (indicator_ < 0) {
      indicator_ = (long)out_.size();
      out_.push_back(0);
    }
    if (nbuf_) {
      const size_t o = out_.size();
      out_.resize(o + (size_t)bw_);  // 8 values * bw bits
      uint8_t* p = out_.data() + o;
      // BitWriter::PutValue: little-endian 64-bit accumulator, spilled 8 bytes at a time
      uint64_t acc = 0;
      int bits = 0;
      for (int i = 0; i < nbuf_; i++) {
        const uint64_t v = buf_[i];
        acc |= v << bits;
        bits += bw_;
        if (bits >= 64) {
          std::memcpy(p, &acc, 8);
          p += 8;
          bits -= 64;
          acc = (bw_ - bits == 64) ? 0 : (v >> (bw_ - bits));
        }
      }
      std::memcpy(p, &acc, (size_t)((bits + 7) / 8));
      nbuf_ = 0;
    }
    if (update_indicator) {
      out_[(size_t)indicator_] = (uint8_t)(((literal_ / 8) << 1) | 1);
      indicator_ = -1;
      literal_ = 0;
    }
  }

  void flush_buffered(bool done) {
    if (repeat_ >= 8) {
      nbuf_ = 0;
      if (literal_ != 0) flush_literal(true);
      return;
    }
    literal_ += nbuf_;
    if (literal_ / 8 + 1 >= (1 << 6)) flush_literal(true);
    else flush_literal(done);
    repeat_ = 0;
  }
};

}  // namespace fastsim
