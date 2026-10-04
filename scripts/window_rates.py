"""Final-style throughput proxy: events / Docker container window, per public unit.

Final ranks the arithmetic mean over units of (parquet row count / the Docker daemon's window
``State.StartedAt -> State.FinishedAt``). This measures that quantity locally for an image, one
container at a time (no concurrent load), over every single-scenario public unit (each unit's own
seed). It is a local guide, not an official number.

    python scripts/window_rates.py --image track3-fastsim:latest --out run_outputs/window_fastsim.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent


def _ts(s: str) -> dt.datetime:
    # 2026-10-04T19:01:26.123456789Z -> microsecond precision
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"
    return dt.datetime.fromisoformat(s)


def window(image: str, unit: Path, out: Path) -> float:
    cid = subprocess.run(
        ["docker", "create", "--network", "none", "--cpus", "4", "--memory", "16g",
         "-v", f"{unit.resolve()}:/input:ro", "-v", f"{out.resolve()}:/output", image,
         "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet"],
        capture_output=True, text=True, check=True).stdout.strip()
    try:
        subprocess.run(["docker", "start", "-a", cid], capture_output=True, check=True)
        info = json.loads(subprocess.run(["docker", "inspect", cid], capture_output=True, text=True,
                                         check=True).stdout)[0]["State"]
        if info["ExitCode"] != 0:
            raise RuntimeError(f"{unit.name}: exit {info['ExitCode']}")
        return (_ts(info["FinishedAt"]) - _ts(info["StartedAt"])).total_seconds()
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for unit in sorted((REPO / "units").iterdir()):
        if not (unit / "scenario.json").is_file():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            w = window(args.image, unit, Path(tmp))
            n = pq.ParquetFile(Path(tmp) / "trace.parquet").metadata.num_rows
        rows.append({"unit": unit.name, "n_events": n, "window_sec": w, "rate": n / w})
        print(f"{unit.name:40s} {n:8d} ev {w * 1000:9.1f} ms {n / w:12.0f} ev/s", flush=True)
    rates = [r["rate"] for r in rows]
    summary = {"image": args.image, "n_units": len(rows), "mean_rate": statistics.mean(rates),
               "median_rate": statistics.median(rates), "units": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"mean {summary['mean_rate']:.0f} ev/s, median {summary['median_rate']:.0f} ev/s over {len(rows)} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
