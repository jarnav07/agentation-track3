"""Columnar outputs → ``trace.parquet`` / ``message_trace.parquet``, byte-identical to the baseline.

The baseline writes through ``pandas.DataFrame.to_parquet(compression="snappy", index=False)``
(pandas 1.5.3 → pyarrow 15.0.2). That is ``pyarrow.parquet.write_table`` with default options on a
table whose schema carries a ``pandas`` metadata blob. Neither the blob nor the writer options
depend on the data, so building the same Arrow table directly — with the same metadata string —
produces the same bytes without importing pandas (≈0.2 s of start-up). The pyarrow version must
match the baseline's (15.0.2): it is written into the footer.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PYARROW_VERSION = "15.0.2"
PANDAS_VERSION = "1.5.3"


def _pandas_meta(columns: list[tuple[str, str, str]]) -> bytes:
    meta = {
        "index_columns": [],
        "column_indexes": [],
        "columns": [
            {"name": n, "field_name": n, "pandas_type": pt, "numpy_type": nt, "metadata": None}
            for n, pt, nt in columns
        ],
        "creator": {"library": "pyarrow", "version": PYARROW_VERSION},
        "pandas_version": PANDAS_VERSION,
    }
    return json.dumps(meta).encode()


TRACE_SCHEMA = pa.schema(
    [
        ("t_ns", pa.int64()),
        ("agent_id", pa.int32()),
        ("msg_type", pa.string()),
        ("side", pa.string()),
        ("price", pa.int64()),
        ("size", pa.int64()),
        ("order_id", pa.int64()),
    ],
    metadata={
        b"pandas": _pandas_meta(
            [
                ("t_ns", "int64", "int64"),
                ("agent_id", "int32", "int32"),
                ("msg_type", "unicode", "string"),
                ("side", "unicode", "string"),
                ("price", "int64", "int64"),
                ("size", "int64", "int64"),
                ("order_id", "int64", "int64"),
            ]
        )
    },
)

LEDGER_SCHEMA = pa.schema(
    [
        ("seq", pa.int64()),
        ("t_recv_ns", pa.int64()),
        ("t_send_ns", pa.int64()),
        ("latency_ns", pa.int64()),
        ("src_id", pa.int32()),
        ("dst_id", pa.int32()),
        ("message_id", pa.int64()),
        ("msg_type", pa.string()),
        ("order_id", pa.int64()),
        ("causal_parent", pa.int64()),
    ],
    metadata={
        b"pandas": _pandas_meta(
            [
                ("seq", "int64", "int64"),
                ("t_recv_ns", "int64", "int64"),
                ("t_send_ns", "int64", "Int64"),
                ("latency_ns", "int64", "int64"),
                ("src_id", "int32", "int32"),
                ("dst_id", "int32", "int32"),
                ("message_id", "int64", "int64"),
                ("msg_type", "unicode", "string"),
                ("order_id", "int64", "Int64"),
                ("causal_parent", "int64", "Int64"),
            ]
        )
    },
)


def _dict_strings(codes: np.ndarray, names: tuple[str, ...]) -> pa.Array:
    """A plain string array from small-int codes into ``names``; code -1 -> null.

    Built directly from offset / data buffers (vectorised byte gather), which avoids a dictionary
    cast and with it the import of ``pyarrow.compute`` (~35 ms of start-up).
    """
    codes = np.asarray(codes, dtype=np.int32)
    n = len(codes)
    enc = [nm.encode() for nm in names]
    width = max((len(b) for b in enc), default=1) or 1
    table = np.zeros((len(enc) + 1, width), dtype=np.uint8)  # last row: null -> empty
    lens = np.zeros(len(enc) + 1, dtype=np.int32)
    for i, b in enumerate(enc):
        table[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        lens[i] = len(b)
    null = codes < 0
    idx = np.where(null, len(enc), codes)
    row_lens = lens[idx]
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(row_lens, out=offsets[1:])
    mask = np.arange(width, dtype=np.int32)[None, :] < row_lens[:, None]
    data = table[idx][mask]
    validity = None
    null_count = int(null.sum())
    if null_count:
        validity = pa.py_buffer(np.packbits(~null, bitorder="little"))
    return pa.StringArray.from_buffers(
        n, pa.py_buffer(offsets), pa.py_buffer(data), validity, null_count
    )


def write_trace(path, t_ns, agent_id, msg_code, msg_names, side_code, price, size, order_id) -> None:
    """``side_code``: 0 = BID, 1 = ASK, -1 = null."""
    table = pa.Table.from_arrays(
        [
            pa.array(np.asarray(t_ns, dtype=np.int64)),
            pa.array(np.asarray(agent_id, dtype=np.int32)),
            _dict_strings(np.asarray(msg_code, dtype=np.int32), msg_names),
            _dict_strings(np.asarray(side_code, dtype=np.int32), ("BID", "ASK")),
            pa.array(np.asarray(price, dtype=np.int64)),
            pa.array(np.asarray(size, dtype=np.int64)),
            pa.array(np.asarray(order_id, dtype=np.int64)),
        ],
        schema=TRACE_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def _nullable_i64(values: np.ndarray, valid: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return pa.array(values)
    return pa.array(values, mask=~valid)


def write_ledger(path, seq, t_recv, t_send, t_send_valid, latency, src, dst, message_id,
                 kind_code, kind_names, order_id, order_valid, causal, causal_valid) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array(np.asarray(seq, dtype=np.int64)),
            pa.array(np.asarray(t_recv, dtype=np.int64)),
            _nullable_i64(t_send, t_send_valid),
            pa.array(np.asarray(latency, dtype=np.int64)),
            pa.array(np.asarray(src, dtype=np.int32)),
            pa.array(np.asarray(dst, dtype=np.int32)),
            pa.array(np.asarray(message_id, dtype=np.int64)),
            _dict_strings(np.asarray(kind_code, dtype=np.int32), kind_names),
            _nullable_i64(order_id, order_valid),
            _nullable_i64(causal, causal_valid),
        ],
        schema=LEDGER_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")
