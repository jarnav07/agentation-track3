// Bit-exact re-implementation of the NumPy legacy RandomState stream (MT19937 + the "legacy"
// distributions that RandomState methods use). State is imported verbatim from
// RandomState.get_state(), so seeding is NumPy's own; only the draws are done here.
//
// Every routine mirrors numpy/random/src/{mt19937,legacy,distributions} and
// numpy/random/_bounded_integers for the call shapes the baseline uses. tests/test_rng.py checks
// each against NumPy over millions of draws.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// glibc's exp/log cores without the errno wrapper (same ifunc-selected routine as exp()/log(), so
// bit-identical results; tests/test_rng.py and the byte-equality oracle check the streams).
__asm__(".symver fs_exp_finite,__exp_finite@GLIBC_2.15");
__asm__(".symver fs_log_finite,__log_finite@GLIBC_2.15");
extern "C" double fs_exp_finite(double);
extern "C" double fs_log_finite(double);

namespace fastsim {

inline double rng_exp(double x) { return fs_exp_finite(x); }
inline double rng_log(double x) { return fs_log_finite(x); }

struct MT19937 {
  static constexpr int N = 624;
  static constexpr int M = 397;
  uint32_t key[N];
  uint32_t out[N];  // tempered key (valid for the current block; see temper_all)
  int pos = N;
  int has_gauss = 0;
  double gauss = 0.0;

  void refill() {
    constexpr uint32_t UPPER = 0x80000000U, LOWER = 0x7fffffffU, MATRIX_A = 0x9908b0dfU;
    int i;
    uint32_t y;
    for (i = 0; i < N - M; i++) {
      y = (key[i] & UPPER) | (key[i + 1] & LOWER);
      key[i] = key[i + M] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }
    for (; i < N - 1; i++) {
      y = (key[i] & UPPER) | (key[i + 1] & LOWER);
      key[i] = key[i + (M - N)] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }
    y = (key[N - 1] & UPPER) | (key[0] & LOWER);
    key[N - 1] = key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    temper_all();
    pos = 0;
  }
  // Temper the whole block at once (a vectorisable loop) so a draw is a load. Must be called after
  // the key is set from outside with pos < N.
  void temper_all() {
    for (int i = 0; i < N; i++) {
      uint32_t y = key[i];
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680U;
      y ^= (y << 15) & 0xefc60000U;
      y ^= (y >> 18);
      out[i] = y;
    }
  }

  inline uint32_t next32() {
    if (__builtin_expect(pos == N, 0)) refill();
    return out[pos++];
  }

  inline uint64_t next64() {
    uint64_t hi = next32();
    return (hi << 32) | next32();
  }

  // mt19937_next_double / legacy_double
  inline double next_double() {
    int32_t a = next32() >> 5, b = next32() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  // legacy_gauss (polar Box-Muller with one cached value)
  double std_gauss() {
    if (has_gauss) {
      const double tmp = gauss;
      gauss = 0;
      has_gauss = 0;
      return tmp;
    }
    double f, x1, x2, r2;
    do {
      x1 = 2.0 * next_double() - 1.0;
      x2 = 2.0 * next_double() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    f = std::sqrt(-2.0 * rng_log(r2) / r2);
    gauss = f * x1;
    has_gauss = 1;
    return f * x2;
  }

  // The std_gauss() output stream in bulk: appends outputs in stream order to `out`, at least
  // `want` of them and at most want + 2*kBulk + 1, and leaves the generator exactly where that many
  // sequential std_gauss() calls would (has_gauss cleared: whole pairs are always emitted). The
  // polar method consumes 4 words per attempt whatever the outcome, so a batch of attempts can be
  // evaluated with straight-line (vectorisable) code and only the accepted pairs kept, in order.
  static constexpr int kBulk = 32;
  size_t gauss_bulk(double* out, size_t want) {
    size_t n = 0;
    if (has_gauss) {
      out[n++] = gauss;
      gauss = 0;
      has_gauss = 0;
    }
    uint32_t u[4 * kBulk];
    double x1[kBulk], x2[kBulk], r2[kBulk];
    while (n < want) {
      for (int i = 0; i < 4 * kBulk; i++) u[i] = next32();
      for (int i = 0; i < kBulk; i++) {
        const int32_t a1 = (int32_t)(u[4 * i] >> 5), b1 = (int32_t)(u[4 * i + 1] >> 6);
        const int32_t a2 = (int32_t)(u[4 * i + 2] >> 5), b2 = (int32_t)(u[4 * i + 3] >> 6);
        x1[i] = 2.0 * ((a1 * 67108864.0 + b1) / 9007199254740992.0) - 1.0;
        x2[i] = 2.0 * ((a2 * 67108864.0 + b2) / 9007199254740992.0) - 1.0;
        r2[i] = x1[i] * x1[i] + x2[i] * x2[i];
      }
      for (int i = 0; i < kBulk; i++) {
        if (r2[i] >= 1.0 || r2[i] == 0.0) continue;
        const double f = std::sqrt(-2.0 * rng_log(r2[i]) / r2[i]);
        out[n++] = f * x2[i];  // returned first
        out[n++] = f * x1[i];  // the cached half
      }
    }
    return n;
  }

  // RandomState.normal(loc, scale)
  inline double normal(double loc, double scale) { return loc + scale * std_gauss(); }
  // RandomState.lognormal(mean, sigma)
  inline double lognormal(double mean, double sigma) { return rng_exp(normal(mean, sigma)); }
  // RandomState.uniform(low, high): low + (high - low) * U   (range computed first, as NumPy does)
  inline double uniform(double low, double high) {
    const double range = high - low;
    return low + range * next_double();
  }
  // legacy_standard_exponential: -log(1 - U)
  inline double std_exponential() { return -rng_log(1.0 - next_double()); }
  // RandomState.exponential(scale)
  inline double exponential(double scale) { return scale * std_exponential(); }
  // RandomState.pareto(a): exp(E / a) - 1
  inline double pareto(double a) { return rng_exp(std_exponential() / a) - 1; }

  // RandomState.randint(low, high) for the default int64 dtype: masked rejection sampling on the
  // inclusive range rng = high - 1 - low. rng == 0 consumes nothing.
  int64_t randint(int64_t low, int64_t high) {
    const uint64_t rng = (uint64_t)(high - 1 - low);
    if (rng == 0) return low;
    if (rng <= 0xFFFFFFFFULL) {
      if (rng == 0xFFFFFFFFULL) return low + (int64_t)next32();
      uint32_t mask = (uint32_t)rng;
      mask |= mask >> 1;
      mask |= mask >> 2;
      mask |= mask >> 4;
      mask |= mask >> 8;
      mask |= mask >> 16;
      uint32_t val;
      while ((val = (next32() & mask)) > (uint32_t)rng) {
      }
      return low + (int64_t)val;
    }
    if (rng == 0xFFFFFFFFFFFFFFFFULL) return low + (int64_t)next64();
    uint64_t mask = rng;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    uint64_t val;
    while ((val = (next64() & mask)) > rng) {
    }
    return low + (int64_t)val;
  }
};

}  // namespace fastsim
