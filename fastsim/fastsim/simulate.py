"""``simulate`` — the Track 3 single-scenario verb for the fast engine.

    simulate --config /input/scenario.json --out /output/trace.parquet [--seed N]

Writes ``trace.parquet``, ``message_trace.parquet`` and ``events.json`` next to ``--out``, with the
same contents as the ABIDES baseline adapter (``baselines/abides_fork/simulate.py``).

``wall_clock_sec`` keeps the baseline's boundary: it times the simulation itself (the baseline
times ``abides.run``), not interpreter start-up or parquet writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import resource
import sys
import time
from typing import Any, Optional

from .scenario import build_spec
from .scenario_io import read_scenario

ENGINE = os.environ.get("FASTSIM_ENGINE", "auto")  # "auto" | "c" | "py"


def _use_compiled() -> bool:
    if ENGINE == "py":
        return False
    try:
        from . import _engine  # noqa: F401
    except ImportError:
        if ENGINE == "c":
            raise
        return False
    return True


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_scenario(scenario: dict[str, Any], out_path: pathlib.Path) -> dict[str, Any]:
    """Run one scenario and write its outputs. Returns the events.json dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msg_out = out_path.parent / "message_trace.parquet"
    spec = build_spec(scenario)

    if _use_compiled():
        from .cengine import run_and_write

        n_events, n_messages, wall_clock_sec = run_and_write(spec, out_path, msg_out)
    else:
        from .assemble import write_outputs_py
        from .pyengine import PyEngine

        t0 = time.perf_counter()
        res = PyEngine(spec).run()
        wall_clock_sec = time.perf_counter() - t0
        n_events, n_messages = write_outputs_py(res, out_path, msg_out)

    events = {
        "scenario_id": str(scenario["scenario_id"]),
        "seed": int(scenario["seed"]),
        "n_events": int(n_events),
        "wall_clock_sec": float(wall_clock_sec),
        "events_per_sec": float(n_events / wall_clock_sec) if wall_clock_sec > 0 else 0.0,
        "trace_sha256": _sha256(out_path),
        "n_messages": int(n_messages),
        "message_trace_sha256": _sha256(msg_out),
        "peak_memory_bytes": _peak_rss_bytes(),
        "gpu_seconds": 0.0,
    }
    (out_path.parent / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    return events


def simulate(config_path, out_path, seed: Optional[int] = None) -> dict[str, Any]:
    scenario = json.loads(read_scenario(config_path))
    if seed is not None:
        scenario = {**scenario, "seed": int(seed)}
    return run_scenario(scenario, pathlib.Path(out_path))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="simulate")
    ap.add_argument("verb", nargs="?", default="simulate", choices=["simulate"])
    ap.add_argument("--config", required=True, help="path to scenario.json")
    ap.add_argument("--out", required=True, help="output path for trace.parquet")
    ap.add_argument("--seed", type=int, default=None, help="override scenario seed")
    args = ap.parse_args(argv)
    events = simulate(args.config, args.out, args.seed)
    sys.stdout.write(json.dumps(events) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
