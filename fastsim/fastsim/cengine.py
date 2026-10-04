"""Pack an :class:`EngineSpec` for the compiled engine, run it, and write the outputs.

The packed layout is parsed by ``csrc/engine.cpp::parse_spec``; keep the two in step. RNG streams
travel as their full NumPy ``get_state()`` (key, pos, cached gaussian), so the compiled engine
continues exactly the streams ``scenario.build_spec`` seeded.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from . import _engine
from .output import write_ledger, write_trace
from .pyengine import KIND_NAMES
from .assemble import MSG_NAMES
from .scenario import LAT_MATRIX, EngineSpec

_MAGIC = 0x46535631  # "FSV1"


def _rng_bytes(rs: np.random.RandomState) -> bytes:
    name, key, pos, has_gauss, gauss = rs.get_state()
    assert name == "MT19937"
    return np.ascontiguousarray(key, dtype="<u4").tobytes() + struct.pack("<iid", pos, has_gauss, gauss)


def pack_spec(spec: EngineSpec) -> bytes:
    stp = 0 if not spec.stp_policy else (1 if spec.stp_policy == "cancel_oldest" else 2)
    lat = spec.latency
    parts = [
        struct.pack("<I", _MAGIC),
        struct.pack("<6q", spec.start_time, spec.stop_time, spec.mkt_open, spec.mkt_close,
                    spec.pipeline_delay, spec.exchange_computation_delay),
        struct.pack("<ii", stp, spec.n_agents),
        struct.pack("<i6d", lat.model, lat.mean_ns, lat.sigma, lat.min_ns, lat.max_ns, lat.alpha, lat.mu),
    ]
    if lat.model == LAT_MATRIX:
        parts.append(np.ascontiguousarray(lat.matrix, dtype="<i8").tobytes())
    o = spec.oracle
    parts.append(struct.pack(
        "<q7d", o.r_bar, o.kappa, o.fund_vol, 1.0 / o.megashock_lambda_a, o.megashock_mean,
        math.sqrt(o.megashock_var), float(o.first_megashock_time), float(o.first_megashock_value)))
    parts.append(struct.pack("<i", len(o.scheduled_jumps)))
    for j in o.scheduled_jumps:
        parts.append(struct.pack("<qqi", j["time_ns"], j["magnitude"], 1 if j.get("_consumed") else 0))
    for a in spec.agents:
        parts.append(struct.pack(
            "<iqddqqqqqqdqq", a.kind, a.interval_ns, a.order_size_mean, a.order_size_std,
            a.price_offset_ticks, a.reference_price, a.spread_ticks, a.depth_levels, a.size_per_level,
            a.threshold_ticks, a.sigma_n, a.lookback, a.size))
    rngs = [spec.global_rs, o.random_state, lat.random_state] + [a.random_state for a in spec.agents]
    parts.append(struct.pack("<i", len(rngs)))
    parts.extend(_rng_bytes(rs) for rs in rngs)
    return b"".join(parts)


def run_and_write(spec: EngineSpec, trace_path, ledger_path):
    """Returns (n_events, n_messages, engine_seconds)."""
    import time

    blob = pack_spec(spec)
    t0 = time.perf_counter()
    cols = _engine.run(blob)
    engine_sec = time.perf_counter() - t0

    def i64(k):
        return np.frombuffer(cols[k], dtype=np.int64)

    def i32(k):
        return np.frombuffer(cols[k], dtype=np.int32)

    def b8(k):
        return np.frombuffer(cols[k], dtype=np.uint8).astype(bool)

    write_trace(trace_path, i64("t_ns"), i32("agent_id"), i32("msg_code"), MSG_NAMES,
                i32("side_code"), i64("price"), i64("size"), i64("order_id"))
    n_led = len(cols["l_t_recv"]) // 8
    write_ledger(ledger_path, np.arange(n_led, dtype=np.int64), i64("l_t_recv"), i64("l_t_send"),
                 b8("l_t_send_valid"), i64("l_latency"), i32("l_src"), i32("l_dst"), i64("l_msg_id"),
                 i32("l_kind"), KIND_NAMES, i64("l_order_id"), b8("l_order_valid"), i64("l_causal"),
                 b8("l_causal_valid"))
    return len(cols["t_ns"]) // 8, n_led, engine_sec
