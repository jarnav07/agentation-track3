"""Turn the engine's event records into the baseline's exact row order (``abides_fork/trace.py``).

trace.parquet
    Order-lifecycle rows are recorded in processing order, which is the baseline's per-agent log
    order with nondecreasing time. The baseline stable-sorts the concatenated agent logs by
    ``EventTime`` and then the whole frame by ``(t_ns, order_id)``; rows sharing ``(t_ns, order_id)``
    always belong to one agent (the order's owner), so the result equals a stable sort of the
    processing-order rows by ``(t_ns, order_id)``. The last execution per order id is
    ``ORDER_FILLED``, earlier ones ``PARTIAL_FILL``.

    Quote rows: the exchange's BEST_BID / BEST_ASK emissions, keeping the last per ``(t_ns, side)``,
    ordered by each key's first appearance, ``order_id = -1``, ``agent_id = 0``. They sort ahead of
    order rows at the same ``t_ns`` because ``-1`` is below every order id.

message_trace.parquet
    One row per delivered message, in delivery (``seq``) order, already produced by the engine.
"""

from __future__ import annotations

import numpy as np

from .output import write_ledger, write_trace
from .pyengine import KIND_NAMES, T_ACCEPTED, T_CANCELLED, T_EXEC, T_SUBMITTED

# trace msg_type codes
MSG_NAMES = (
    "ORDER_SUBMITTED",
    "ORDER_ACCEPTED",
    "ORDER_FILLED",
    "PARTIAL_FILL",
    "ORDER_CANCELLED",
    "QUOTE_UPDATE",
)
M_SUBMITTED, M_ACCEPTED, M_FILLED, M_PARTIAL, M_CANCELLED, M_QUOTE = range(6)
_TTYPE_TO_MSG = {T_SUBMITTED: M_SUBMITTED, T_ACCEPTED: M_ACCEPTED, T_CANCELLED: M_CANCELLED}


def trace_columns(order_rows: list[tuple], quotes: list[tuple]) -> dict[str, np.ndarray]:
    n = len(order_rows)
    # order rows: (t, agent, ttype, is_bid, price, qty, oid)
    last_exec: dict[int, int] = {}
    for i, r in enumerate(order_rows):
        if r[2] == T_EXEC:
            last_exec[r[6]] = i
    final = set(last_exec.values())
    msg = np.empty(n, dtype=np.int32)
    for i, r in enumerate(order_rows):
        tt = r[2]
        msg[i] = (M_FILLED if i in final else M_PARTIAL) if tt == T_EXEC else _TTYPE_TO_MSG[tt]
    if n:
        t = np.fromiter((r[0] for r in order_rows), dtype=np.int64, count=n)
        agent = np.fromiter((r[1] for r in order_rows), dtype=np.int32, count=n)
        side = np.fromiter((0 if r[3] else 1 for r in order_rows), dtype=np.int32, count=n)
        price = np.fromiter((r[4] for r in order_rows), dtype=np.int64, count=n)
        size = np.fromiter((r[5] for r in order_rows), dtype=np.int64, count=n)
        oid = np.fromiter((r[6] for r in order_rows), dtype=np.int64, count=n)
    else:
        t = agent = side = price = size = oid = np.empty(0, dtype=np.int64)

    # quotes: keep last per (t, side), ranked by first appearance
    q_first: dict[tuple, int] = {}
    q_last: dict[tuple, tuple] = {}
    for qt, is_bid, qp, qs in quotes:
        key = (qt, is_bid)
        if key not in q_first:
            q_first[key] = len(q_first)
        q_last[key] = (qp, qs)
    qkeys = sorted(q_first, key=q_first.__getitem__)
    m = len(qkeys)
    qt_arr = np.fromiter((k[0] for k in qkeys), dtype=np.int64, count=m)
    qside = np.fromiter((0 if k[1] else 1 for k in qkeys), dtype=np.int32, count=m)
    qprice = np.fromiter((q_last[k][0] for k in qkeys), dtype=np.int64, count=m)
    qsize = np.fromiter((q_last[k][1] for k in qkeys), dtype=np.int64, count=m)

    all_t = np.concatenate([t.astype(np.int64), qt_arr])
    all_oid = np.concatenate([oid.astype(np.int64), np.full(m, -1, dtype=np.int64)])
    # stable sort by (t, oid): lexsort is stable; last key is primary
    order = np.lexsort((all_oid, all_t))
    return {
        "t_ns": all_t[order],
        "agent_id": np.concatenate([agent.astype(np.int32), np.zeros(m, dtype=np.int32)])[order],
        "msg_code": np.concatenate([msg, np.full(m, M_QUOTE, dtype=np.int32)])[order],
        "side_code": np.concatenate([side.astype(np.int32), qside])[order],
        "price": np.concatenate([price.astype(np.int64), qprice])[order],
        "size": np.concatenate([size.astype(np.int64), qsize])[order],
        "order_id": all_oid[order],
    }


def write_outputs_py(res, trace_path, ledger_path) -> tuple[int, int]:
    cols = trace_columns(res.order_rows, res.quotes)
    write_trace(trace_path, cols["t_ns"], cols["agent_id"], cols["msg_code"], MSG_NAMES,
                cols["side_code"], cols["price"], cols["size"], cols["order_id"])
    led = res.ledger
    n = len(led)
    # (mid, src, dst, t_send, t_recv, latency, kind, order_id, causal)
    t_send_valid = np.fromiter((r[3] is not None for r in led), dtype=bool, count=n)
    order_valid = np.fromiter((r[7] is not None for r in led), dtype=bool, count=n)
    causal_valid = np.fromiter((r[8] is not None for r in led), dtype=bool, count=n)
    write_ledger(
        ledger_path,
        np.arange(n, dtype=np.int64),
        np.fromiter((r[4] for r in led), dtype=np.int64, count=n),
        np.fromiter((r[3] if r[3] is not None else 0 for r in led), dtype=np.int64, count=n),
        t_send_valid,
        np.fromiter((r[5] for r in led), dtype=np.int64, count=n),
        np.fromiter((r[1] for r in led), dtype=np.int32, count=n),
        np.fromiter((r[2] for r in led), dtype=np.int32, count=n),
        np.fromiter((r[0] for r in led), dtype=np.int64, count=n),
        np.fromiter((r[6] for r in led), dtype=np.int32, count=n),
        KIND_NAMES,
        np.fromiter((r[7] if r[7] is not None else 0 for r in led), dtype=np.int64, count=n),
        order_valid,
        np.fromiter((r[8] if r[8] is not None else 0 for r in led), dtype=np.int64, count=n),
        causal_valid,
    )
    return len(cols["t_ns"]), n
