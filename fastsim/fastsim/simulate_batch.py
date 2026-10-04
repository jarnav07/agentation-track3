"""``simulate-batch`` — the batched multi-scenario verb (BatchMarketSim / family GB).

    simulate-batch --batch-dir /input/scenarios --out-dir /output

Each ``<sub>.json`` is an independent scenario. Every sub is run exactly as ``simulate`` would run it
alone (fresh counters, fresh seeded streams — the engine holds no global state), so each sub's
output is byte-identical to its isolated run: the isolation gate's property.

Subs are distributed over a process pool (one worker per available CPU, capped by the number of
subs). The pool changes only *when* a sub runs, never what it computes. Outputs are written by the
worker that ran the sub; ``batch_events.json`` lists subs in sorted filename order, as the baseline
does.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

from .simulate import simulate


def _run_one(args: tuple[str, str]) -> dict[str, Any]:
    sub_path, out_path = args
    return {**simulate(sub_path, out_path), "_pid": os.getpid()}


def _workers(n_subs: int) -> int:
    env = os.environ.get("FASTSIM_BATCH_WORKERS")
    if env:
        return max(1, min(int(env), n_subs))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    # Respect a cgroup CPU quota (docker --cpus) when it is tighter than the affinity mask.
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            q = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                cpus = min(cpus, max(1, q // p))
        except (OSError, ValueError):
            pass
    return max(1, min(cpus, n_subs))


def simulate_batch(batch_dir, out_dir) -> dict[str, Any]:
    batch_dir = pathlib.Path(batch_dir)
    out_dir = pathlib.Path(out_dir)
    subs = sorted(p for p in batch_dir.glob("*.json"))
    if not subs:
        raise SystemExit(f"simulate-batch: no sub-scenarios (*.json) found in {batch_dir}")
    jobs = [(str(p), str(out_dir / p.stem / "trace.parquet")) for p in subs]

    t0 = time.perf_counter()
    workers = _workers(len(jobs))
    if workers == 1:
        results = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, jobs))
    wall_clock_sec = time.perf_counter() - t0

    per_scenario = []
    total_events = 0
    peak_by_pid: dict[int, int] = {}
    gpu_seconds = 0.0
    for sub_path, ev in zip(subs, results):
        total_events += int(ev["n_events"])
        # Workers run concurrently: the batch's peak is the sum of each worker process's peak.
        pid = int(ev["_pid"])
        peak_by_pid[pid] = max(peak_by_pid.get(pid, 0), int(ev.get("peak_memory_bytes", 0)))
        gpu_seconds += float(ev.get("gpu_seconds", 0.0))
        per_scenario.append(
            {"sub": sub_path.stem, "n_events": int(ev["n_events"]), "trace_sha256": ev["trace_sha256"]}
        )
    batch_events = {
        "n_scenarios": len(subs),
        "total_events": total_events,
        "wall_clock_sec": float(wall_clock_sec),
        "events_per_sec": float(total_events / wall_clock_sec) if wall_clock_sec > 0 else 0.0,
        "peak_memory_bytes": sum(peak_by_pid.values()),
        "gpu_seconds": gpu_seconds,
        "per_scenario": per_scenario,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_events.json").write_text(json.dumps(batch_events, indent=2) + "\n")
    return batch_events


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="simulate-batch")
    ap.add_argument("verb", nargs="?", default="simulate-batch", choices=["simulate-batch"])
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    batch_events = simulate_batch(args.batch_dir, args.out_dir)
    sys.stdout.write(json.dumps(batch_events) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
