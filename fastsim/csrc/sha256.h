// SHA-256 (FIPS 180-4), for the trace_sha256 / message_trace_sha256 fields of events.json.
#pragma once
#include <cpuid.h>
#include <dlfcn.h>
#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace fastsim {

// SHA-256 compression with the x86 SHA extensions (SHA-NI), used when the CPU has them; the
// portable block() below otherwise. Same function, so the same digest. After the public-domain
// reference by Jeffrey Walton / Sean Gulley (Intel).
__attribute__((target("sha,sse4.1"))) inline void sha256_ni_blocks(uint32_t state[8], const uint8_t* data,
                                                                     size_t nblocks) {
  const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
  __m128i TMP = _mm_loadu_si128((const __m128i*)&state[0]);
  __m128i STATE1 = _mm_loadu_si128((const __m128i*)&state[4]);
  TMP = _mm_shuffle_epi32(TMP, 0xB1);           // CDAB
  STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);     // EFGH
  __m128i STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);  // ABEF
  STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);       // CDGH
  while (nblocks--) {
    const __m128i ABEF_SAVE = STATE0, CDGH_SAVE = STATE1;
    __m128i MSG, MSG0, MSG1, MSG2, MSG3;
    // rounds 0-3
    MSG = _mm_loadu_si128((const __m128i*)(data + 0));
    MSG0 = _mm_shuffle_epi8(MSG, MASK);
    MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x71374491428A2F98ULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    // rounds 4-7
    MSG1 = _mm_loadu_si128((const __m128i*)(data + 16));
    MSG1 = _mm_shuffle_epi8(MSG1, MASK);
    MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0xAB1C5ED5923F82A4ULL, 0x59F111F13956C25BULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);
    // rounds 8-11
    MSG2 = _mm_loadu_si128((const __m128i*)(data + 32));
    MSG2 = _mm_shuffle_epi8(MSG2, MASK);
    MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x550C7DC3243185BEULL, 0x12835B01D807AA98ULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);
    // rounds 12-15
    MSG3 = _mm_loadu_si128((const __m128i*)(data + 48));
    MSG3 = _mm_shuffle_epi8(MSG3, MASK);
    MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC19BF1749BDC06A7ULL, 0x80DEB1FE72BE5D74ULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
    MSG0 = _mm_add_epi32(MSG0, TMP);
    MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);
#define FASTSIM_SHA_QROUND(MA, MB, MC, MD, K1, K0)                     \
    MSG = _mm_add_epi32(MA, _mm_set_epi64x(K1, K0));                    \
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);                \
    TMP = _mm_alignr_epi8(MA, MD, 4);                                   \
    MB = _mm_add_epi32(MB, TMP);                                        \
    MB = _mm_sha256msg2_epu32(MB, MA);                                  \
    MSG = _mm_shuffle_epi32(MSG, 0x0E);                                 \
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);                \
    MD = _mm_sha256msg1_epu32(MD, MA);
    // rounds 16-51 (message schedule in flight)
    FASTSIM_SHA_QROUND(MSG0, MSG1, MSG2, MSG3, 0x240CA1CC0FC19DC6ULL, 0xEFBE4786E49B69C1ULL)
    FASTSIM_SHA_QROUND(MSG1, MSG2, MSG3, MSG0, 0x76F988DA5CB0A9DCULL, 0x4A7484AA2DE92C6FULL)
    FASTSIM_SHA_QROUND(MSG2, MSG3, MSG0, MSG1, 0xBF597FC7B00327C8ULL, 0xA831C66D983E5152ULL)
    FASTSIM_SHA_QROUND(MSG3, MSG0, MSG1, MSG2, 0x1429296706CA6351ULL, 0xD5A79147C6E00BF3ULL)
    FASTSIM_SHA_QROUND(MSG0, MSG1, MSG2, MSG3, 0x53380D134D2C6DFCULL, 0x2E1B213827B70A85ULL)
    FASTSIM_SHA_QROUND(MSG1, MSG2, MSG3, MSG0, 0x92722C8581C2C92EULL, 0x766A0ABB650A7354ULL)
    FASTSIM_SHA_QROUND(MSG2, MSG3, MSG0, MSG1, 0xC76C51A3C24B8B70ULL, 0xA81A664BA2BFE8A1ULL)
    FASTSIM_SHA_QROUND(MSG3, MSG0, MSG1, MSG2, 0x106AA070F40E3585ULL, 0xD6990624D192E819ULL)
    FASTSIM_SHA_QROUND(MSG0, MSG1, MSG2, MSG3, 0x34B0BCB52748774CULL, 0x1E376C0819A4C116ULL)
#undef FASTSIM_SHA_QROUND
    // rounds 52-55
    MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x682E6FF35B9CCA4FULL, 0x4ED8AA4A391C0CB3ULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
    MSG2 = _mm_add_epi32(MSG2, TMP);
    MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    // rounds 56-59
    MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x8CC7020884C87814ULL, 0x78A5636F748F82EEULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
    MSG3 = _mm_add_epi32(MSG3, TMP);
    MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    // rounds 60-63
    MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xA4506CEB90BEFFFAULL));
    STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
    MSG = _mm_shuffle_epi32(MSG, 0x0E);
    STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
    STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
    STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);
    data += 64;
  }
  TMP = _mm_shuffle_epi32(STATE0, 0x1B);        // FEBA
  STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);     // DCHG
  STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);  // DCBA
  STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);     // HGFE
  _mm_storeu_si128((__m128i*)&state[0], STATE0);
  _mm_storeu_si128((__m128i*)&state[4], STATE1);
}

inline bool cpu_has_sha_ni() {
  static const bool has = [] {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 19))) return false;  // SSE4.1
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, a, b, c, d);
    return (b & (1u << 29)) != 0 && !std::getenv("FASTSIM_NO_SHANI");      // SHA
  }();
  return has;
}

class Sha256 {
 public:
  Sha256() { reset(); }
  void reset() {
    static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof(h_));
    len_ = 0;
    fill_ = 0;
  }
  void update(const uint8_t* p, size_t n) {
    len_ += n;
    if (fill_ == 0 && n >= 64) {  // whole blocks straight from the input
      const size_t nb = n / 64;
      blocks(p, nb);
      p += nb * 64;
      n -= nb * 64;
    }
    while (n) {
      size_t take = 64 - fill_ < n ? 64 - fill_ : n;
      std::memcpy(buf_ + fill_, p, take);
      fill_ += take;
      p += take;
      n -= take;
      if (fill_ == 64) {
        blocks(buf_, 1);
        fill_ = 0;
      }
    }
  }
  std::string hexdigest() {
    uint64_t bits = len_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (fill_ != 56) update(&zero, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lenb, 8);
    char out[65];
    for (int i = 0; i < 8; i++) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
    return std::string(out, 64);
  }

 private:
  uint32_t h_[8];
  uint64_t len_;
  uint8_t buf_[64];
  size_t fill_;
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void blocks(const uint8_t* p, size_t nb) {
    if (cpu_has_sha_ni()) {
      sha256_ni_blocks(h_, p, nb);
      return;
    }
    for (size_t i = 0; i < nb; i++) block(p + 64 * i);
  }
  void block(const uint8_t* b) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 | (uint32_t)b[4 * i + 2] << 8 | b[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], bb = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; i++) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + k[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t mj = (a & bb) ^ (a & c) ^ (bb & c);
      uint32_t t2 = S0 + mj;
      h = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += bb; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }
};

// SHA-256 of a memory buffer as hex. Uses the image's OpenSSL (libcrypto.so.3, which Python's
// hashlib already depends on) when it can be loaded -- its hand-tuned SHA-256 (AVX2 / SHA-NI) is
// faster than the code above -- and the code above otherwise. Same function, same digest.
inline std::string sha256_hex(const uint8_t* data, size_t n) {
  typedef const void* (*md_fn)();
  typedef int (*digest_fn)(const void*, size_t, unsigned char*, unsigned int*, const void*, void*);
  static md_fn evp_sha256 = nullptr;
  static digest_fn evp_digest = nullptr;
  static const bool have = [] {
    if (std::getenv("FASTSIM_NO_OPENSSL")) return false;
    void* h = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
    evp_sha256 = (md_fn)dlsym(h, "EVP_sha256");
    evp_digest = (digest_fn)dlsym(h, "EVP_Digest");
    return evp_sha256 && evp_digest;
  }();
  if (have) {
    unsigned char md[32];
    unsigned int len = 0;
    if (evp_digest(data, n, md, &len, evp_sha256(), nullptr) == 1 && len == 32) {
      char out[65];
      for (int i = 0; i < 32; i++) std::snprintf(out + 2 * i, 3, "%02x", md[i]);
      return std::string(out, 64);
    }
  }
  Sha256 s;
  s.update(data, n);
  return s.hexdigest();
}

inline std::string sha256_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return "";
  Sha256 s;
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.update(buf, n);
  std::fclose(f);
  return s.hexdigest();
}

}  // namespace fastsim
