#include "native_setup.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace fastsim {

// ------------------------------------------------------------------ np.log replication
extern "C" __m512d __svml_log8(__m512d);

__attribute__((target("avx512f"))) static double svml_log_scalar(double x) {
  // npyv_load_till_f64(src, len=1, fill=1.0): lane 0 = x, other lanes 1.0
  __m512d v = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x);
  __m512d r = __svml_log8(v);
  double out[8];
  _mm512_storeu_pd(out, r);
  return out[0];
}

static bool cpu_has_avx512_skx() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512cd") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq");
}

double numpy_log(double x) {
  static const bool skx = cpu_has_avx512_skx();
  return skx ? svml_log_scalar(x) : std::log(x);
}

// ------------------------------------------------------------------ Python conversions
namespace {

using json::Value;

struct Conv {
  std::string& why;
  bool ok = true;
  void fail(const std::string& msg) {
    if (ok) why = msg;
    ok = false;
  }
  // float(v)  (v absent -> default)
  double f(const Value* v, double dflt, const char* name) {
    if (!v) return dflt;
    if (v->type == json::FLOAT) return v->d;
    if (v->type == json::INT && !v->big_int) return (double)v->i;
    fail(std::string("non-numeric float field ") + name);
    return 0.0;
  }
  // int(v): int stays, float truncates toward zero
  int64_t i(const Value* v, int64_t dflt, const char* name) {
    if (!v) return dflt;
    if (v->type == json::INT && !v->big_int) return v->i;
    if (v->type == json::FLOAT && std::isfinite(v->d) && std::fabs(v->d) < 9.2e18) return (int64_t)std::trunc(v->d);
    fail(std::string("non-integral int field ") + name);
    return 0;
  }
  // int(max(lo, v))  (Python max returns the first maximal argument)
  int64_t imax(int64_t lo, const Value* v, int64_t dflt, const char* name) {
    if (!v) return dflt > lo ? dflt : lo;
    if (v->type == json::INT && !v->big_int) return v->i > lo ? v->i : lo;
    if (v->type == json::FLOAT && std::isfinite(v->d) && std::fabs(v->d) < 9.2e18)
      return v->d > (double)lo ? (int64_t)std::trunc(v->d) : lo;
    fail(std::string("bad numeric field ") + name);
    return lo;
  }
  const Value* obj(const Value* v, const char* name, bool required) {
    if (!v) {
      if (required) fail(std::string("missing ") + name);
      return nullptr;
    }
    if (v->type != json::OBJECT) {
      fail(std::string("non-object ") + name);
      return nullptr;
    }
    return v;
  }
};

constexpr int64_t DATE_NS = 1612483200000000000LL;    // pd.to_datetime("20210205").value
constexpr int64_t OPEN_OFFSET_NS = 34200000000000LL;  // 09:30:00
constexpr int64_t ONE_SECOND_NS = 1000000000LL;

void init_genrand(MT19937& m, uint32_t s) {
  for (int k = 0; k < MT19937::N; k++) {
    m.key[k] = s;
    s = 1812433253U * (s ^ (s >> 30)) + (uint32_t)(k + 1);
  }
  m.pos = MT19937::N;
  m.has_gauss = 0;
  m.gauss = 0.0;
}

// np.random.RandomState(seed=np.random.randint(0, 2**32, dtype="uint64"))
MT19937 draw_random_state(MT19937& global) {
  MT19937 m;
  init_genrand(m, global.next32());
  return m;
}

const Value empty_object = [] {
  Value v;
  v.type = json::OBJECT;
  return v;
}();

}  // namespace

bool build_native_spec(const json::Value& sc, Spec& s, ScenarioMeta& meta, std::string& why) {
  Conv C{why};
  if (sc.type != json::OBJECT) { why = "scenario is not an object"; return false; }

  const Value* sid = sc.get("scenario_id");
  if (!sid || sid->type != json::STRING) { why = "scenario_id is not a string"; return false; }
  meta.scenario_id = sid->s;
  const Value* seedv = sc.get("seed");
  if (!seedv) { why = "missing seed"; return false; }
  int64_t seed = C.i(seedv, 0, "seed");
  if (!C.ok) return false;
  if (seed < 0 || seed > 0xFFFFFFFFLL) { why = "seed outside [0, 2**32)"; return false; }
  meta.seed = seed;

  MT19937 global;
  init_genrand(global, (uint32_t)seed);  // np.random.seed(seed)

  const Value* ex = C.obj(sc.get("exchange_config"), "exchange_config", true);
  const Value* oc = C.obj(sc.get("oracle_config"), "oracle_config", true);
  if (!C.ok) return false;
  bool proto = ex->get("protocol_enforcement") && ex->get("protocol_enforcement")->truthy();
  s.stp = STP_NONE;
  if (proto && ex->get("stp_policy") && ex->get("stp_policy")->truthy()) {
    const Value* sp = ex->get("stp_policy");
    if (sp->type != json::STRING) { why = "non-string stp_policy"; return false; }
    s.stp = sp->s == "cancel_oldest" ? STP_OLDEST : STP_NEWEST;
  }
  s.pipeline_delay = proto ? C.i(ex->get("ack_delay_ns"), 0, "ack_delay_ns") : 0;
  s.ex_comp_delay = proto ? C.i(ex->get("compute_delay_ns"), 0, "compute_delay_ns") : 0;
  const Value* op = oc->get("params");
  if (op) op = C.obj(op, "oracle_config.params", false);
  if (!op) op = &empty_object;
  if (!C.ok) return false;
  int64_t reference_price = C.i(op->get("initial_price"), 100000, "initial_price");
  const Value* hz = sc.get("horizon_ns");
  if (!hz) { why = "missing horizon_ns"; return false; }
  int64_t horizon = C.i(hz, 0, "horizon_ns");
  if (!C.ok) return false;

  s.start_time = DATE_NS;
  s.mkt_open = DATE_NS + OPEN_OFFSET_NS;
  s.mkt_close = s.mkt_open + horizon;
  s.stop_time = s.mkt_close + ONE_SECOND_NS;

  double kappa_per_s = C.f(op->get("kappa"), 0.0, "kappa");
  s.kappa = kappa_per_s > 0 ? kappa_per_s / 1e9 : 1.67e-16;
  double rate_per_s = C.f(op->get("jump_intensity"), 0.0, "jump_intensity");
  double lam = rate_per_s > 0 ? rate_per_s / 1e9 : 2.77778e-18;
  double ms_mean = C.f(op->get("jump_sigma"), 0.0, "jump_sigma");
  if (ms_mean == 0.0) ms_mean = 1000.0;  // float(...) or 1000.0
  s.fund_vol = C.f(op->get("sigma"), 5e-5, "sigma");
  s.r_bar = reference_price;
  s.ms_scale = 1.0 / lam;
  s.ms_mean = ms_mean;
  s.ms_sd = std::sqrt(50000.0);
  if (!C.ok) return false;

  // oracle RandomState, then the oracle constructor's global exponential + its own normal/randint
  MT19937 oracle_rs = draw_random_state(global);
  const Value* sj = op->get("scheduled_jump");
  if (sj && sj->truthy()) {
    if (sj->type != json::OBJECT) { why = "non-object scheduled_jump"; return false; }
    const Value* t = sj->get("time_ns");
    const Value* m = sj->get("magnitude");
    if (!t || !m) { why = "incomplete scheduled_jump"; return false; }
    Jump j{s.mkt_open + C.i(t, 0, "scheduled_jump.time_ns"), C.i(m, 0, "scheduled_jump.magnitude"), false};
    if (!C.ok) return false;
    s.jumps.push_back(j);
  }
  double ms_time_delta = global.exponential(1.0 / lam);
  s.first_mst = (double)s.mkt_open + ms_time_delta;
  double msv = oracle_rs.normal(ms_mean, std::sqrt(50000.0));
  if (oracle_rs.randint(0, 2) != 0) msv = -msv;
  s.first_msv = msv;

  MT19937 exchange_rs = draw_random_state(global);  // never used; consumed for order
  (void)exchange_rs;

  const Value* acs = sc.get("agent_configs");
  if (!acs || acs->type != json::ARRAY) { why = "agent_configs is not a list"; return false; }
  s.agents.clear();
  s.agents.push_back(AgentParams{});  // slot 0 = exchange
  std::vector<MT19937> agent_rngs;
  for (const Value& cfg : acs->arr) {
    if (cfg.type != json::OBJECT) { why = "agent_config is not an object"; return false; }
    const Value* at = cfg.get("agent_type");
    if (!at || at->type != json::STRING) { why = "agent_type is not a string"; return false; }
    const Value* params = cfg.get("params");
    if (params) params = C.obj(params, "params", false);
    if (!params) params = &empty_object;
    const Value* cnt = cfg.get("count");
    if (!cnt) { why = "missing count"; return false; }
    int64_t count = C.i(cnt, 0, "count");
    if (!C.ok) return false;

    AgentParams p{};
    // _interval_ns, then ScheduledAgent: int(max(1, interval))
    int64_t interval;
    if (params->has("rebalance_interval_ns")) {
      interval = C.i(params->get("rebalance_interval_ns"), 0, "rebalance_interval_ns");
    } else {
      double h = C.f(params->get("arrival_rate_hz"), 1.0, "arrival_rate_hz");
      double q = h > 0 ? 1e9 / h : 1e9;
      if (!(std::isfinite(q) && std::fabs(q) < 9.2e18)) { why = "interval out of range"; return false; }
      interval = (int64_t)std::trunc(q);
    }
    p.interval_ns = interval > 1 ? interval : 1;
    const std::string& type = at->s;
    if (type == "NoiseTrader") {
      p.kind = A_NOISE;
      p.order_size_mean = C.f(params->get("order_size_mean"), 10, "order_size_mean");
      p.order_size_std = C.f(params->get("order_size_std"), 2, "order_size_std");
      p.price_offset_ticks = C.i(params->get("price_offset_ticks"), 5, "price_offset_ticks");
      p.reference_price = reference_price;
      if (p.price_offset_ticks + 1 <= 0) { why = "price_offset_ticks < 0"; return false; }
    } else if (type == "MarketMaker") {
      p.kind = A_MM;
      p.spread_ticks = C.imax(2, params->get("spread_ticks"), 2, "spread_ticks");
      p.depth_levels = C.imax(1, params->get("depth_levels"), 3, "depth_levels");
      p.size_per_level = C.imax(1, params->get("size_per_level"), 10, "size_per_level");
      p.reference_price = reference_price;
    } else if (type == "ValueTrader") {
      p.kind = A_VALUE;
      p.order_size_mean = C.f(params->get("order_size_mean"), 25, "order_size_mean");
      p.threshold_ticks = C.i(params->get("threshold_ticks"), 2, "threshold_ticks");
      p.sigma_n = 1000.0;
      int64_t r = (int64_t)std::nearbyint(p.order_size_mean);
      p.size = r > 1 ? r : 1;
    } else if (type == "MomentumTrader") {
      p.kind = A_MOMENTUM;
      p.order_size_mean = C.f(params->get("order_size_mean"), 15, "order_size_mean");
      p.threshold_ticks = C.i(params->get("threshold_ticks"), 2, "threshold_ticks");
      p.lookback = C.imax(1, params->get("lookback"), 5, "lookback");
      int64_t r = (int64_t)std::nearbyint(p.order_size_mean);
      p.size = r > 1 ? r : 1;
    } else {
      why = "unsupported agent_type " + type;
      return false;
    }
    if (!C.ok) return false;
    if ((p.kind == A_VALUE || p.kind == A_MOMENTUM) && !std::isfinite(p.order_size_mean)) {
      why = "non-finite order_size_mean";
      return false;
    }
    for (int64_t k = 0; k < count; k++) {
      s.agents.push_back(p);
      agent_rngs.push_back(draw_random_state(global));
    }
  }
  s.n_agents = (int32_t)s.agents.size();

  const Value* lc = sc.get("latency_config");
  if (!lc || !lc->truthy()) { why = "no latency_config (line-distance model)"; return false; }
  if (lc->type != json::OBJECT) { why = "non-object latency_config"; return false; }
  const Value* lp = lc->get("params");
  if (lp) lp = C.obj(lp, "latency_config.params", false);
  if (!lp) lp = &empty_object;
  const Value* mv = lc->get("model");
  std::string model = "deterministic";
  if (mv) {
    if (mv->type != json::STRING) { why = "non-string latency model"; return false; }
    model = mv->s;
  }
  s.lat_model = model == "log_normal" ? L_LOGNORMAL : model == "uniform" ? L_UNIFORM : model == "pareto" ? L_PARETO : L_DET;
  s.lat_mean = C.f(lp->get("mean_ns"), 0.0, "mean_ns");
  s.lat_sigma = C.f(lp->get("sigma"), 0.0, "sigma");
  s.lat_min = C.f(lp->get("min_ns"), 0.0, "min_ns");
  s.lat_max = C.f(lp->get("max_ns"), 1e12, "max_ns");
  s.lat_alpha = C.f(lp->get("alpha"), 1.5, "alpha");
  if (!C.ok) return false;
  s.lat_mu = s.lat_mean > 0 ? numpy_log(s.lat_mean) : 0.0;
  MT19937 latency_rs = draw_random_state(global);
  MT19937 kernel_rs = draw_random_state(global);  // consumed for order; never used
  (void)kernel_rs;

  s.rngs.clear();
  s.rngs.reserve(agent_rngs.size() + 3);
  s.rngs.push_back(global);
  s.rngs.push_back(oracle_rs);
  s.rngs.push_back(latency_rs);
  for (auto& r : agent_rngs) s.rngs.push_back(r);
  return true;
}

}  // namespace fastsim
