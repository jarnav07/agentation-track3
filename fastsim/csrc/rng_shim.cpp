// ctypes shim exposing rng.h for the RNG exactness tests (not used at run time).
#include "rng.h"
using fastsim::MT19937;
extern "C" {
// kind: 0 next_double, 1 normal(p0,p1), 2 lognormal(p0,p1), 3 uniform(p0,p1), 4 pareto(p0),
//       5 exponential(p0), 6 randint(int(p0), int(p1)), 7 next32
void draw(const uint32_t* key, int pos, int has_gauss, double gauss, int kind, double p0, double p1,
          int n, double* out, uint32_t* key_out, int* pos_out) {
  MT19937 mt;
  std::memcpy(mt.key, key, sizeof(mt.key));
  mt.pos = pos; mt.has_gauss = has_gauss; mt.gauss = gauss;
  for (int i = 0; i < n; i++) {
    switch (kind) {
      case 0: out[i] = mt.next_double(); break;
      case 1: out[i] = mt.normal(p0, p1); break;
      case 2: out[i] = mt.lognormal(p0, p1); break;
      case 3: out[i] = mt.uniform(p0, p1); break;
      case 4: out[i] = mt.pareto(p0); break;
      case 5: out[i] = mt.exponential(p0); break;
      case 6: out[i] = (double)mt.randint((int64_t)p0, (int64_t)p1); break;
      case 7: out[i] = (double)mt.next32(); break;
    }
  }
  std::memcpy(key_out, mt.key, sizeof(mt.key));
  *pos_out = mt.pos;
}
}
