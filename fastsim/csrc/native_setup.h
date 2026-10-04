// Scenario JSON -> engine Spec without Python: the native mirror of fastsim/scenario.py (which
// mirrors baselines/abides_fork/config.py). Seeding is done here too, reproducing NumPy's legacy
// seeding (init_genrand; verified equal to np.random.seed / RandomState(int)) and draw order.
//
// Anything outside the shapes this mirror is sure about returns false with a reason, and the
// caller hands the whole run to the Python implementation instead — which then behaves exactly as
// the baseline would (including raising the same errors for malformed scenarios).
#pragma once
#include <string>

#include "engine.h"
#include "json.h"

namespace fastsim {

struct ScenarioMeta {
  std::string scenario_id;
  int64_t seed = 0;
};

// Optional seed override (the --seed flag) is applied by the caller to the JSON before this.
bool build_native_spec(const json::Value& scenario, Spec& spec, ScenarioMeta& meta, std::string& why);

// np.log on a float64 scalar, as NumPy 1.26 computes it on this CPU: Intel SVML (__svml_log8) when
// the CPU has AVX512_SKX (NumPy's dispatch condition for its SVML loops), else libm log.
double numpy_log(double x);

}  // namespace fastsim
