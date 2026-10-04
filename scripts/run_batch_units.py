"""Run every public batch unit (``t3-gbatch-*``) through ``simulate-batch`` and score it locally.

``run_regression.py`` covers the 65 single-scenario units only. This script fills the gap for the
six batch units: it mounts each unit's ``scenarios/`` read-only, runs the candidate image with the
``simulate-batch`` verb under ``--network none``, then applies the same public gates the g3 path
uses: ``batch.score_isolation`` (per-sub Tier-A + message-ledger checks against
``checks/reference_data``) and ``batch.check_aggregate`` (``batch_events.json`` arithmetic).

Usage::

    python scripts/run_batch_units.py --image <img> --out-dir run_outputs/batch/<label>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from qfbench2_track_simulation.batch import check_aggregate, load_subs, score_isolation

REPO = Path(__file__).resolve().parent.parent


def _family_tier(unit: Path) -> tuple[int, str | None, dict[str, float], int]:
    card = tomllib.loads((unit / "card.toml").read_text())
    params = card["scoring"]["params"]
    # Batch units are throughput-scale (family 6, Tier A).
    return (
        6,
        params.get("semantic_tier"),
        dict(params.get("stylized_fact_ceilings", {})),
        int(params.get("timestamp_tolerance_ns", 1000)),
    )


def run_unit(image: str, unit: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "16g",
        "-v", f"{unit.resolve()}:/input:ro", "-v", f"{out.resolve()}:/output",
        image, "simulate-batch", "--batch-dir", "/input/scenarios", "--out-dir", "/output",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    host_wall = time.perf_counter() - t0
    rec: dict = {"unit": unit.name, "returncode": proc.returncode, "host_wall_sec": host_wall}
    if proc.returncode != 0:
        rec["stderr"] = proc.stderr[-2000:]
        rec["pass"] = False
        return rec
    family, tier, ceilings, tol = _family_tier(unit)
    ok_iso, failures = score_isolation(unit, out, family, tier, ceilings, timestamp_tolerance_ns=tol)
    ok_agg, agg = check_aggregate(out, load_subs(unit))
    be = json.loads((out / "batch_events.json").read_text())
    rec.update(
        isolation_ok=ok_iso,
        isolation_failures=failures,
        aggregate_ok=ok_agg,
        aggregate=agg,
        events_per_sec=be.get("events_per_sec"),
        total_events=be.get("total_events"),
        wall_clock_sec=be.get("wall_clock_sec"),
    )
    rec["pass"] = bool(ok_iso and ok_agg)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--units-dir", default=str(REPO / "units"))
    args = ap.parse_args(argv)
    units = sorted(
        p for p in Path(args.units_dir).iterdir()
        if (p / "batch.json").is_file() and (p / "scenarios").is_dir()
    )
    results = []
    for unit in units:
        rec = run_unit(args.image, unit, Path(args.out_dir) / unit.name)
        results.append(rec)
        print(
            f"[{'PASS' if rec['pass'] else 'FAIL'}] {unit.name} "
            f"eps={rec.get('events_per_sec')} host_wall={rec['host_wall_sec']:.2f}s"
            + ("" if rec["pass"] else f" {rec.get('isolation_failures') or rec.get('aggregate') or rec.get('stderr')}"),
            flush=True,
        )
    summary = {
        "n_units": len(results),
        "passed": sum(r["pass"] for r in results),
        "results": results,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "batch_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"batch units passed: {summary['passed']}/{summary['n_units']}")
    return 0 if summary["passed"] == summary["n_units"] else 1


if __name__ == "__main__":
    sys.exit(main())
