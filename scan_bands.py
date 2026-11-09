"""Select each operation's bands and store them where SABLE's
bench_suitesparse.py replays them: results_vdia<op>/<matrix>.yaml beside this
script, one file per matrix the operation's extractor accepts (a stale file of
a matrix it now declines is removed). One band search per matrix serves both
operations.

Without matrix names it scans the pool: square SuiteSparse matrices with
1e5-1.5e7 nonzeros, one per name, whose local Matrix Market file
(SABLE_LOCAL_SUITESPARSE_DIR/<name>/<name>.mtx) is not complex. A summary
line per matrix goes to band_scan.jsonl; matrices already there are skipped,
so an interrupted scan resumes.

  scan_bands.py [--workers N] [matrix ...]
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import ssgetpy
import yaml
from scipy.io import mmread
from scipy.sparse import csr_matrix

from find_vdia import (
    SPMM_MIN_COVERAGE,
    SPMV_MIN_COVERAGE,
    SearchTimeout,
    band_candidates,
    select_bands,
    spmm_slot_cost,
    spmv_slot_cost,
)

FILEPATH = pathlib.Path(__file__).resolve().parent
PARAMETERS = {"spmv": (spmv_slot_cost, SPMV_MIN_COVERAGE), "spmm": (spmm_slot_cost, SPMM_MIN_COVERAGE)}
POOL_NNZ = (100_000, 15_000_000)
BANDS_DIR = FILEPATH
SUMMARY = BANDS_DIR / "band_scan.jsonl"


def local_dir() -> pathlib.Path:
    path = os.environ.get("SABLE_LOCAL_SUITESPARSE_DIR")
    if not path:
        sys.exit("Set SABLE_LOCAL_SUITESPARSE_DIR to the local SuiteSparse copy (<dir>/<name>/<name>.mtx)")
    return pathlib.Path(path)


def mtx_path(name: str) -> pathlib.Path:
    return local_dir() / name / f"{name}.mtx"


def header(path: pathlib.Path) -> tuple[str, int, int]:
    """(field, rows, cols) of a Matrix Market file."""
    with open(path) as f:
        field = f.readline().split()[3].lower()
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, cols = map(int, line.split()[:2])
    return field, rows, cols


def pool() -> list[str]:
    names = sorted({m.name for m in ssgetpy.search(nzbounds=POOL_NNZ, limit=100_000) if m.rows == m.cols})
    out = []
    for name in names:
        path = mtx_path(name)
        if path.exists():
            field, rows, cols = header(path)
            if field != "complex" and rows == cols:
                out.append(name)
    return out


def scan(name: str) -> dict:
    start = time.perf_counter()
    A = csr_matrix(mmread(mtx_path(name)))
    A.eliminate_zeros()
    A.sort_indices()
    record = {"matrix": name, "nnz": int(A.nnz)}
    try:
        candidates = band_candidates(A)
    except SearchTimeout:
        candidates = None
    search_seconds = time.perf_counter() - start
    for op, (slot_cost, min_coverage) in PARAMETERS.items():
        info = {}
        if candidates is None:
            regions = []
            info.update(segments=0, coverage=0.0, density=0.0, min_coverage=min_coverage,
                        seconds=round(search_seconds, 1), choice="timeout")
        else:
            regions = select_bands(A, slot_cost, min_coverage, candidates=candidates, info=info)
        path = BANDS_DIR / f"results_vdia{op}" / f"{name}.yaml"
        if regions:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.safe_dump({"order": ["vdia"], "band_extractor": op, "band_selection": info,
                                "extraction_seconds": {"vdia": round(search_seconds, 3)},
                                "bands": [r.to_dict() for r in regions], "blocks": []}, f, sort_keys=False)
        elif path.exists():
            path.unlink()
        record[op] = info
    record["seconds"] = round(time.perf_counter() - start, 1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrices", nargs="*", help="Matrices to scan (default: the whole pool)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    names = args.matrices or pool()
    done = set()
    if SUMMARY.exists() and not args.matrices:
        done = {json.loads(line)["matrix"] for line in open(SUMMARY) if line.strip()}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} matrices, {len(todo)} to scan", flush=True)
    with mp.Pool(args.workers, maxtasksperchild=1) as workers, open(SUMMARY, "a") as summary:
        for i, record in enumerate(workers.imap_unordered(scan, todo), 1):
            summary.write(json.dumps(record) + "\n")
            summary.flush()
            picks = [op for op in PARAMETERS if record[op]["choice"] == "bands"]
            print(f"{i}/{len(todo)} {record['matrix']}: {', '.join(picks) or record['spmv']['choice']} "
                  f"({record['seconds']} s)", flush=True)
    for op in PARAMETERS:
        print(f"{op}: {len(list((BANDS_DIR / f'results_vdia{op}').glob('*.yaml')))} matrices with bands")


if __name__ == "__main__":
    main()
