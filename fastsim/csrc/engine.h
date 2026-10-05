// Compiled event engine: a direct port of fastsim/pyengine.py (which is the readable statement of
// the ABIDES baseline's semantics). Every branch below has its counterpart there; keep the two in
// step. Inputs arrive as a packed spec built in Python (scenario.py does all pre-loop RNG work with
// NumPy); outputs are columnar buffers in final row order.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "big_alloc.h"
#include "rng.h"

namespace fastsim {

enum Kind : int8_t {
  K_WAKEUP = 0,
  K_CLOSE_PRICE_REQ = 1,
  K_HOURS_REQ = 2,
  K_HOURS = 3,
  K_QUERY_SPREAD = 4,
  K_SPREAD_RESP = 5,
  K_LIMIT = 6,
  K_CANCEL = 7,
  K_ACCEPTED = 8,
  K_EXECUTED = 9,
  K_CANCELLED = 10,
  K_MKT_CLOSED = 11,
  K_CLOSE_PRICE = 12,
};

enum AgentKind : int32_t { A_NOISE = 0, A_MM = 1, A_VALUE = 2, A_MOMENTUM = 3 };
enum LatModel : int32_t { L_DET = 0, L_LOGNORMAL = 1, L_UNIFORM = 2, L_PARETO = 3, L_MATRIX = 4 };
enum Stp : int32_t { STP_NONE = 0, STP_OLDEST = 1, STP_NEWEST = 2 };

struct AgentParams {
  int32_t kind;
  int64_t interval_ns;
  double order_size_mean, order_size_std;
  int64_t price_offset_ticks, reference_price;
  int64_t spread_ticks, depth_levels, size_per_level;
  int64_t threshold_ticks;
  double sigma_n;
  int64_t lookback, size;
};

struct Jump {
  int64_t time_ns, magnitude;
  bool consumed;
};

struct Spec {
  int64_t start_time, stop_time, mkt_open, mkt_close;
  int64_t pipeline_delay, ex_comp_delay;
  int32_t stp;
  int32_t n_agents;  // including the exchange
  // latency
  int32_t lat_model;
  double lat_mean, lat_sigma, lat_min, lat_max, lat_alpha, lat_mu;
  std::vector<int64_t> lat_matrix;
  // oracle
  int64_t r_bar;
  double kappa, fund_vol, ms_scale, ms_mean, ms_sd;
  double first_mst, first_msv;
  std::vector<Jump> jumps;
  std::vector<AgentParams> agents;  // index = agent id; [0] unused
  // RNG streams: [0] global, [1] oracle, [2] latency, [3 + i] agent i+1
  std::vector<MT19937> rngs;
  // Spare cores the engine may use for helper work (page pre-faulting); results never depend on it.
  int helper_threads = 0;
};

struct Output {
  // trace (final order)
  bvec<int64_t> t_ns, price, size, order_id;
  bvec<int32_t> agent_id, msg_code, side_code;
  // ledger (seq order)
  bvec<int64_t> l_t_recv, l_t_send, l_latency, l_msg_id, l_order_id, l_causal;
  bvec<int32_t> l_src, l_dst, l_kind;
  bvec<uint8_t> l_t_send_valid, l_order_valid, l_causal_valid;
  std::string error;
};

bool parse_spec(const char* buf, size_t len, Spec& spec, std::string& err);
// sim_sec (optional): seconds spent in the event loop alone, excluding the trace assembly that
// follows it -- the same boundary the baseline times (abides.run, before extract_trace).
bool run_engine(Spec& spec, Output& out, double* sim_sec = nullptr);

}  // namespace fastsim
