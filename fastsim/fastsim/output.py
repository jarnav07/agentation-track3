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
    """A plain (non-dictionary) string array from small-int codes; code -1 → null."""
    return pa.DictionaryArray.from_arrays(
        pa.array(codes, type=pa.int32(), mask=codes < 0) if (codes < 0).any() else pa.array(codes, type=pa.int32()),
        pa.array(list(names), type=pa.string()),
    ).cast(pa.string())


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
