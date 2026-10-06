// Byte-for-byte check of csrc/rle_fast.h against arrow::util::RleEncoder (pyarrow's header).
//   g++ -O2 -std=c++17 -I<pyarrow>/include fastsim/tests/test_rle_fast.cpp -o /tmp/t && /tmp/t
#include <arrow/util/rle_encoding.h>

#include <cstdio>
#include <random>
#include <vector>

#include "../csrc/rle_fast.h"

int main() {
  std::mt19937_64 g(7);
  long bad = 0, cases = 0;
  for (int trial = 0; trial < 300000; trial++) {
    const int bw = (int)(g() % 21);  // indices up to 2^20 entries; bw 0..20
    const int n = (int)(g() % 3000);
    const int mode = (int)(g() % 4);
    std::vector<uint64_t> v(n);
    const uint64_t maxv = bw == 0 ? 0 : ((1ULL << bw) - 1);
    uint64_t cur = 0;
    for (int i = 0; i < n; i++) {
      if (mode == 0) v[i] = maxv ? g() % (maxv + 1) : 0;                       // random
      else if (mode == 1) { if (g() % 10 == 0) cur = maxv ? g() % (maxv + 1) : 0; v[i] = cur; }  // runs
      else if (mode == 2) v[i] = maxv ? std::min<uint64_t>(maxv, (uint64_t)i / (1 + g() % 3)) : 0;  // increasing
      else { if (g() % 3 == 0) cur = maxv ? g() % (maxv + 1) : 0; v[i] = (g() % 4 == 0) ? (maxv ? g() % (maxv + 1) : 0) : cur; }
    }
    const int cap = arrow::util::RleEncoder::MaxBufferSize(bw, n) + arrow::util::RleEncoder::MinBufferSize(bw);
    std::vector<uint8_t> ref(cap);
    arrow::util::RleEncoder e(ref.data(), cap, bw);
    for (uint64_t x : v) e.Put(x);
    const int len = e.Flush();
    std::vector<uint8_t> out;
    fastsim::RleFast f(out, bw);
    for (uint64_t x : v) f.put(x);
    const int len2 = f.flush();
    cases++;
    if (len != len2 || std::memcmp(ref.data(), out.data(), (size_t)len) != 0) {
      if (bad++ < 5) std::printf("mismatch bw=%d n=%d mode=%d len %d vs %d\n", bw, n, mode, len, len2);
    }
  }
  std::printf("rle_fast: %ld cases, %ld mismatches\n", cases, bad);
  return bad != 0;
}
