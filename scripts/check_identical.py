"""Byte-equality oracle: does a candidate image reproduce the ABIDES baseline EXACTLY?

This is deliberately stricter than the scorer. ``run_regression.py`` admits a Tier-B unit on a
statistical comparison and a Tier-A unit on coverage + Kendall-tau; this script demands that
``trace.parquet`` and ``message_trace.parquet`` are byte-identical to what the baseline image writes
for the same scenario and seed. It is the safety net for the sealed scenarios, which we cannot see:
if every public scenario is byte-identical at its own seed *and* at fresh seeds, the candidate is
behaving as the baseline does rather than merely passing the public references.

For every (scenario, seed) pair it:

1. runs the candidate image (``--network none``) with a seed-patched copy of the scenario;
2. obtains the baseline output for the same pair, from a cache keyed by baseline image id, or by
   running the baseline image once;
3. compares the two files by SHA-256 and, where they differ, row by row, reporting the schema
   difference, the row counts and the first diverging row from both sides;
4. checks ``events.json``: the six core fields plus the two telemetry fields are present,
   ``n_events`` equals the parquet row count, ``events_per_sec`` is within 5% of
   ``n_events / wall_clock_sec`` and ``trace_sha256`` matches the file.

``--batch`` additionally runs every batch unit through ``simulate-batch`` and compares each sub's
outputs with the baseline's.

Usage::

    python scripts/check_identical.py --candidate-image track3-fast:latest            # 65 x own seed
    python scripts/check_identical.py --candidate-image track3-fast:latest --extra-seeds 2 --batch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
RUN_TIMEOUT = 1800  # seconds per container run; --timeout overrides
FILES = ("trace.parquet", "message_trace.parquet")
EVENTS_FIELDS = (
    "scenario_id", "seed", "n_events", "wall_clock_sec", "events_per_sec", "trace_sha256",
    "peak_memory_bytes", "gpu_seconds",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seeds(base_seed: int, n: int) -> list[int]:
    """Fresh seeds, derived the way throughput/timer.py derives its seed family."""
    return [
        int(hashlib.sha256(f"{base_seed}:{i}".encode()).hexdigest(), 16) & 0x7FFF_FFFF
        for i in range(1, n + 1)
    ]


def image_id(image: str) -> str:
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().replace("sha256:", "")[:16]


def docker_simulate(image: str, scenario: dict[str, Any], out_dir: Path) -> tuple[int, str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "scenario.json"
        inp.write_text(json.dumps(scenario, indent=2))
        name = f"check-identical-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "4", "--memory", "16g",
            "-v", f"{tmp}:/input:ro", "-v", f"{out_dir.resolve()}:/output",
            image, "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet",
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return 124, f"timed out after {RUN_TIMEOUT}s", time.perf_counter() - t0
        return proc.returncode, proc.stderr[-3000:], time.perf_counter() - t0


def docker_simulate_batch(image: str, unit: Path, out_dir: Path) -> tuple[int, str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "16g",
        "-v", f"{unit.resolve()}:/input:ro", "-v", f"{out_dir.resolve()}:/output",
        image, "simulate-batch", "--batch-dir", "/input/scenarios", "--out-dir", "/output",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stderr[-3000:], time.perf_counter() - t0


def local_simulate(cmd: str, scenario: dict[str, Any], out_dir: Path) -> tuple[int, str, float]:
    """Run a candidate command on the host (no Docker) — fast iteration, same comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "scenario.json"
        inp.write_text(json.dumps(scenario, indent=2))
        argv = shlex.split(cmd) + ["simulate", "--config", str(inp), "--out", str(out_dir / "trace.parquet")]
        t0 = time.perf_counter()
        proc = subprocess.run(argv, capture_output=True, text=True)
        return proc.returncode, proc.stderr[-3000:], time.perf_counter() - t0


def first_divergence(cand: Path, base: Path) -> str:
    """Explain how two parquet files differ: schema, row count, first diverging row."""
    tc, tb = pq.read_table(cand), pq.read_table(base)
    msgs = []
    if tc.schema != tb.schema:
        msgs.append(f"schema differs:\n  cand={tc.schema}\n  base={tb.schema}")
    if tc.num_rows != tb.num_rows:
        msgs.append(f"row count differs: cand={tc.num_rows} base={tb.num_rows}")
    common_cols = [c for c in tb.column_names if c in tc.column_names]
    n = min(tc.num_rows, tb.num_rows)
    first = None
    for col in common_cols:
        a = tc.column(col).to_pylist()[:n]
        b = tb.column(col).to_pylist()[:n]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                if first is None or i < first[0]:
                    first = (i, col)
                break
    if first is not None:
        i, col = first
        lo = max(0, i - 2)
        rc = tc.slice(lo, 5).to_pylist()
        rb = tb.slice(lo, 5).to_pylist()
        msgs.append(f"first diverging row {i} (column {col!r}); rows {lo}..{lo + 4}:")
        for k, (x, y) in enumerate(zip(rc, rb)):
            mark = ">>" if lo + k == i else "  "
            msgs.append(f"  {mark} cand {x}\n  {mark} base {y}")
    elif tc.num_rows != tb.num_rows:
        msgs.append(f"identical over the first {n} rows; one side has extra rows")
    elif not msgs:
        msgs.append("values identical; bytes differ (encoding / metadata / compression)")
        if tc.schema.metadata != tb.schema.metadata:
            msgs.append(f"  schema metadata differs:\n  cand={tc.schema.metadata}\n  base={tb.schema.metadata}")
    return "\n".join(msgs)


def check_events(out_dir: Path) -> list[str]:
    problems = []
    ev_path = out_dir / "events.json"
    if not ev_path.exists():
        return ["missing events.json"]
    ev = json.loads(ev_path.read_text())
    missing = [k for k in EVENTS_FIELDS if k not in ev]
    if missing:
        problems.append(f"events.json missing {missing}")
        return problems
    rows = pq.ParquetFile(out_dir / "trace.parquet").metadata.num_rows
    if int(ev["n_events"]) != rows:
        problems.append(f"n_events {ev['n_events']} != trace rows {rows}")
    wc, eps = float(ev["wall_clock_sec"]), float(ev["events_per_sec"])
    if wc <= 0 or abs(eps - rows / wc) / (rows / wc) > 0.05:
        problems.append(f"events_per_sec {eps} inconsistent with n_events/wall_clock {rows / wc if wc > 0 else 'inf'}")
    if ev["trace_sha256"] != sha256(out_dir / "trace.parquet"):
        problems.append("trace_sha256 does not match trace.parquet")
    return problems


def compare_dirs(cand: Path, base: Path) -> tuple[bool, list[str]]:
    ok, notes = True, []
    for name in FILES:
        c, b = cand / name, base / name
        if not b.exists():
            notes.append(f"{name}: baseline missing (?)")
            ok = False
            continue
        if not c.exists():
            notes.append(f"{name}: candidate missing")
            ok = False
            continue
        if sha256(c) != sha256(b):
            ok = False
            notes.append(f"{name}: SHA differs\n{first_divergence(c, b)}")
    return ok, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-image", help="candidate Docker image")
    ap.add_argument("--candidate-local", help="host command prefix instead of an image, e.g. "
                    "'python -m fastsim.simulate' (the verb and --config/--out are appended)")
    ap.add_argument("--baseline-image", default="track3-abides-baseline:latest")
    ap.add_argument("--scenarios-dir", default=str(REPO / "regression_suite" / "scenarios"))
    ap.add_argument("--only", nargs="*", help="restrict to scenario file stems")
    ap.add_argument("--extra-seeds", type=int, default=0, help="fresh seeds per scenario beyond its own")
    ap.add_argument("--no-own-seed", action="store_true", help="skip each scenario's own seed")
    ap.add_argument("--batch", action="store_true", help="also compare every batch unit per sub")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per container run")
    ap.add_argument("--out-dir", default=str(REPO / "run_outputs" / "identical"))
    ap.add_argument("--cache-dir", default=str(REPO / "run_outputs" / "baseline_cache"))
    args = ap.parse_args(argv)
    global RUN_TIMEOUT
    RUN_TIMEOUT = args.timeout
    if bool(args.candidate_image) == bool(args.candidate_local):
        ap.error("give exactly one of --candidate-image / --candidate-local")
    if args.candidate_local and args.batch:
        ap.error("--batch needs --candidate-image")

    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    cache_root = Path(args.cache_dir) / image_id(args.baseline_image)

    jobs: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(Path(args.scenarios_dir).glob("*.json")):
        if path.name == "index.json" or (args.only and path.stem not in args.only):
            continue
        scenario = json.loads(path.read_text())
        seeds = [] if args.no_own_seed else [int(scenario["seed"])]
        seeds += derive_seeds(int(scenario["seed"]), args.extra_seeds)
        for seed in seeds:
            jobs.append((f"{path.stem}@{seed}", {**scenario, "seed": seed}))

    def run_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        label, scenario = job
        base_dir = cache_root / label
        if not (base_dir / "trace.parquet").exists():
            rc, err, _ = docker_simulate(args.baseline_image, scenario, base_dir)
            if rc != 0:
                return {"label": label, "ok": False, "notes": [f"baseline failed rc={rc}: {err}"]}
        cand_dir = out_root / label
        if args.candidate_local:
            rc, err, wall = local_simulate(args.candidate_local, scenario, cand_dir)
        else:
            rc, err, wall = docker_simulate(args.candidate_image, scenario, cand_dir)
        if rc != 0:
            return {"label": label, "ok": False, "notes": [f"candidate failed rc={rc}: {err}"]}
        ok, notes = compare_dirs(cand_dir, base_dir)
        ev_problems = check_events(cand_dir)
        if ev_problems:
            ok = False
            notes += ev_problems
        return {"label": label, "ok": ok, "notes": notes, "host_wall_sec": wall}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(run_job, jobs):
            results.append(res)
            print(f"[{'SAME' if res['ok'] else 'DIFF'}] {res['label']}", flush=True)
            for note in res["notes"]:
                print("    " + note.replace("\n", "\n    "), flush=True)

    if args.batch:
        for unit in sorted((REPO / "units").iterdir()):
            if not ((unit / "batch.json").is_file() and (unit / "scenarios").is_dir()):
                continue
            label = f"batch:{unit.name}"
            base_dir = cache_root / label.replace(":", "_")
            if not (base_dir / "batch_events.json").exists():
                rc, err, _ = docker_simulate_batch(args.baseline_image, unit, base_dir)
                if rc != 0:
                    results.append({"label": label, "ok": False, "notes": [f"baseline failed: {err}"]})
                    continue
            cand_dir = out_root / label.replace(":", "_")
            rc, err, _ = docker_simulate_batch(args.candidate_image, unit, cand_dir)
            ok, notes = (rc == 0), ([] if rc == 0 else [f"candidate failed rc={rc}: {err}"])
            if rc == 0:
                for sub in sorted(p.stem for p in (unit / "scenarios").glob("*.json")):
                    s_ok, s_notes = compare_dirs(cand_dir / sub, base_dir / sub)
                    ev_problems = check_events(cand_dir / sub)
                    ok = ok and s_ok and not ev_problems
                    notes += [f"{sub}: {n}" for n in s_notes + ev_problems]
                be = json.loads((cand_dir / "batch_events.json").read_text())
                for k in ("n_scenarios", "total_events", "wall_clock_sec", "events_per_sec", "per_scenario"):
                    if k not in be:
                        ok = False
                        notes.append(f"batch_events.json missing {k}")
            results.append({"label": label, "ok": ok, "notes": notes})
            print(f"[{'SAME' if ok else 'DIFF'}] {label}", flush=True)
            for note in notes:
                print("    " + note.replace("\n", "\n    "), flush=True)

    n_ok = sum(r["ok"] for r in results)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "identical_summary.json").write_text(
        json.dumps({"identical": n_ok, "total": len(results), "results": results}, indent=2)
    )
    print(f"\nbyte-identical: {n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
