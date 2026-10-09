// Checks that the vendored snappy 1.1.10 (csrc/third_party/snappy) compresses byte-for-byte like the
// SNAPPY codec in the pyarrow 15.0.2 wheel, on structured random buffers and on any files given
// on the command line (e.g. trace.parquet outputs, which hold real pages).
//   g++ -O2 -std=c++17 -DHAVE_CONFIG_H -Icsrc/third_party/snappy -I$PA/include tests/test_snappy.cpp \
//       csrc/third_party/snappy/snappy*.cc -L$PA -l:libarrow.so.1500 -Wl,-rpath,$PA
#include <arrow/util/compression.h>
#include <snappy.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

static bool same(const std::vector<uint8_t>& in, arrow::util::Codec* codec) {
  std::vector<uint8_t> a(codec->MaxCompressedLen(in.size(), in.data()));
  const int64_t na = *codec->Compress(in.size(), in.data(), a.size(), a.data());
  std::vector<char> b(snappy::MaxCompressedLength(in.size()));
  size_t nb = 0;
  snappy::RawCompress((const char*)in.data(), in.size(), b.data(), &nb);
  return (size_t)na == nb && std::equal(a.begin(), a.begin() + na, (const uint8_t*)b.data());
}

int main(int argc, char** argv) {
  auto codec = *arrow::util::Codec::Create(arrow::Compression::SNAPPY);
  std::mt19937_64 rng(7);
  int bad = 0, n = 0;
  for (int t = 0; t < 3000; t++) {
    size_t len = rng() % (t < 2000 ? 5000 : 3000000);
    int alpha = 1 + rng() % 256;
    std::vector<uint8_t> in(len);
    for (size_t i = 0; i < len; i++) {
      if (i > 16 && rng() % 4 == 0) {  // back-references of varied length / distance
        size_t d = 1 + rng() % std::min<size_t>(i, 70000), l = 1 + rng() % 80;
        for (size_t k = 0; k < l && i < len; k++, i++) in[i] = in[i - d];
        if (i < len) in[i] = rng() % alpha;
      } else {
        in[i] = rng() % alpha;
      }
    }
    bad += !same(in, codec.get());
    n++;
  }
  for (int i = 1; i < argc; i++) {
    std::ifstream f(argv[i], std::ios::binary);
    std::vector<uint8_t> in((std::istreambuf_iterator<char>(f)), {});
    for (size_t off = 0; off < in.size(); off += 4093) {  // sliding windows of the real bytes
      std::vector<uint8_t> w(in.begin() + off, in.begin() + std::min(in.size(), off + 1 + (off * 7919) % 2000000));
      bad += !same(w, codec.get());
      n++;
    }
  }
  std::printf("%d cases, %d mismatches\n", n, bad);
  return bad != 0;
}
