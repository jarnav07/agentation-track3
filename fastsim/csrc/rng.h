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

namespace fastsim {

struct MT19937 {
  static constexpr int N = 624;
  static constexpr int M = 397;
  uint32_t key[N];
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
    pos = 0;
  }

  inline uint32_t next32() {
    if (pos == N) refill();
    uint32_t y = key[pos++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
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
    f = std::sqrt(-2.0 * std::log(r2) / r2);
    gauss = f * x1;
    has_gauss = 1;
    return f * x2;
  }

  // RandomState.normal(loc, scale)
  inline double normal(double loc, double scale) { return loc + scale * std_gauss(); }
  // RandomState.lognormal(mean, sigma)
  inline double lognormal(double mean, double sigma) { return std::exp(normal(mean, sigma)); }
  // RandomState.uniform(low, high): low + (high - low) * U   (range computed first, as NumPy does)
  inline double uniform(double low, double high) {
    const double range = high - low;
    return low + range * next_double();
  }
  // legacy_standard_exponential: -log(1 - U)
  inline double std_exponential() { return -std::log(1.0 - next_double()); }
  // RandomState.exponential(scale)
  inline double exponential(double scale) { return scale * std_exponential(); }
  // RandomState.pareto(a): exp(E / a) - 1
  inline double pareto(double a) { return std::exp(std_exponential() / a) - 1; }

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
