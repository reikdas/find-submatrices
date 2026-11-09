"""
Find VDIA (Variable-width DIAgonal) regions in sparse matrices.

A VDIA region is a union of row-segments centred on a common diagonal offset d.
For each segment s we store (row_start_s, row_end_s, L_s, U_s); a nonzero at
(i, j) with i in segment s belongs to the segment iff  d - L_s <= j - i <= d + U_s.

The algorithm is:

1.  Per-row ribbon fitting is a GREEDY expansion that admits an offset
    into the ribbon only if the per-row ribbon density stays >= MIN_DENSITY.
    Outliers are left for a sparser catch-all format.

2.  No hard cap on bandwidth: the search range scales with the matrix.

3.  Segmentation grows a segment while *segment* density (with L, U taken
    as the rolling max over rows in the segment) stays >= MIN_DENSITY.
    Segment nnz is recomputed exactly against the final (L, U).

4.  Candidate central diagonals are ranked by mass; after a region is
    accepted its rows are masked out so overlapping rediscoveries at
    neighbouring offsets cannot happen.

5.  Acceptance is expressed in purely geometric terms (density and fill
    fraction) because we assume indirections are staged away, so the
    underlying storage-word count does not reflect real SpMV cost:

        vdia_area        = sum_s (L_s + U_s + 1) * len_s
        dia_ribbon_area  = (L_max + U_max + 1) * (row_end - row_start)
        dia_covered_area = (L_max + U_max + 1) * sum_s len_s
        density          = total_nnz / vdia_area
        fill_ribbon      = vdia_area / dia_ribbon_area
        fill_covered     = vdia_area / dia_covered_area

    A region is accepted when

        density    >= MIN_DENSITY
        fill_ribbon <= 1 - MIN_SAVINGS

    `savings_vs_dia_ribbon = 1 - fill_ribbon` splits into a *gap* component
    (1 - fill_covered is zero when the covered rows ARE the ribbon) and a
    *bandwidth-variation* component (1 - vdia_area/dia_covered_area).
"""

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import ssgetpy
from scipy.io import mmread
from scipy.sparse import csr_matrix
from ssgetpy import fetch, search

MIN_DENSITY = 0.5
MIN_SEGMENT_LEN = 20
MIN_REGION_LEN = 50
MIN_REGION_AREA = 2500
MIN_SAVINGS = 0.10
MIN_ROW_NNZ = 2


class SearchTimeout(Exception):
    """find_vdia_regions ran past its deadline."""


@dataclass
class Segment:
    row_start: int
    row_end: int
    lower_bw: int
    upper_bw: int
    nnz: int

    @property
    def length(self) -> int:
        return self.row_end - self.row_start

    @property
    def area(self) -> int:
        return (self.lower_bw + self.upper_bw + 1) * self.length

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rows': [self.row_start, self.row_end],
            'bandwidth': [self.lower_bw, self.upper_bw],
            'nnz': self.nnz,
            'area': self.area,
        }


@dataclass
class VdiaRegion:
    diag_offset: int
    segments: List[Segment]
    dia_ribbon_area: int
    dia_covered_area: int
    # Seconds since the start of find_vdia_regions() at which this region was
    # accepted; lets us evaluate what a shorter search budget would have found.
    found_at_seconds: float = 0.0

    @property
    def row_start(self) -> int:
        return self.segments[0].row_start

    @property
    def row_end(self) -> int:
        return self.segments[-1].row_end

    @property
    def total_nnz(self) -> int:
        return sum(s.nnz for s in self.segments)

    @property
    def vdia_area(self) -> int:
        return sum(s.area for s in self.segments)

    @property
    def density(self) -> float:
        return self.total_nnz / self.vdia_area if self.vdia_area > 0 else 0.0

    @property
    def fill_ribbon(self) -> float:
        if self.dia_ribbon_area == 0:
            return 0.0
        return self.vdia_area / self.dia_ribbon_area

    @property
    def fill_covered(self) -> float:
        if self.dia_covered_area == 0:
            return 0.0
        return self.vdia_area / self.dia_covered_area

    @property
    def savings_vs_dia_ribbon(self) -> float:
        return 1.0 - self.fill_ribbon

    @property
    def savings_vs_dia_covered(self) -> float:
        return 1.0 - self.fill_covered

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rows': [self.row_start, self.row_end],
            'diag_offset': self.diag_offset,
            'segments': [s.to_dict() for s in self.segments],
            'total_nnz': self.total_nnz,
            'vdia_area': self.vdia_area,
            'dia_ribbon_area': self.dia_ribbon_area,
            'dia_covered_area': self.dia_covered_area,
            'density': round(self.density, 6),
            'fill_ribbon': round(self.fill_ribbon, 6),
            'fill_covered': round(self.fill_covered, 6),
            'savings_vs_dia_ribbon': round(self.savings_vs_dia_ribbon, 4),
            'savings_vs_dia_covered': round(self.savings_vs_dia_covered, 4),
            'found_at_seconds': round(self.found_at_seconds, 3),
        }


def compute_row_ribbon(offsets: np.ndarray, max_bw: int,
                       min_density: float = MIN_DENSITY,
                       min_k: int = MIN_ROW_NNZ) -> Tuple[int, int, int]:
    """Tightest (lower, upper, k) around offset 0 whose density stays >= min_density.

    `offsets` are column offsets relative to the diagonal centre for a single row;
    values outside [-max_bw, max_bw] are ignored.  We greedily add offsets ordered
    by |offset|; the last (l, u, k) with k >= min_k and density >= min_density
    is returned (or (0, 0, 0) if none qualifies).
    """
    if offsets.size == 0:
        return 0, 0, 0
    mask = np.abs(offsets) <= max_bw
    valid = offsets[mask]
    if valid.size == 0:
        return 0, 0, 0

    order = np.argsort(np.abs(valid), kind='stable')
    sorted_offs = valid[order]

    l = u = k = 0
    best = (0, 0, 0)
    for off in sorted_offs:
        new_l = max(l, int(-off)) if off < 0 else l
        new_u = max(u, int(off)) if off > 0 else u
        new_k = k + 1
        if new_k / (new_l + new_u + 1) >= min_density:
            l, u, k = new_l, new_u, new_k
            if k >= min_k:
                best = (l, u, k)
        else:
            break
    return best


def compute_row_ribbons_for_diag(csr: csr_matrix, d: int, max_bw: int,
                                 row_used: np.ndarray,
                                 min_density: float = MIN_DENSITY
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """For each row i valid under offset d, return (l, u, k).

    Returns (lower, upper, k, row_min, row_max) where the arrays are indexed
    0..row_max-row_min-1 (local indices).  Rows already in `row_used` or rows
    with no nonzero near the centre get (0, 0, 0).
    """
    n_rows, n_cols = csr.shape
    indptr = csr.indptr
    indices = csr.indices

    if d >= 0:
        row_min, row_max = 0, min(n_rows, n_cols - d)
    else:
        row_min, row_max = max(0, -d), min(n_rows, n_cols - d)
    num = row_max - row_min
    if num <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), row_min, row_max

    lower = np.zeros(num, dtype=np.int32)
    upper = np.zeros(num, dtype=np.int32)
    ks = np.zeros(num, dtype=np.int32)

    for local_i in range(num):
        i = row_min + local_i
        if row_used[i]:
            continue
        start, end = indptr[i], indptr[i + 1]
        if start >= end:
            continue
        center = i + d
        cols = indices[start:end]
        offsets = cols - center
        l, u, k = compute_row_ribbon(offsets, max_bw, min_density)
        if k >= MIN_ROW_NNZ:
            lower[local_i] = l
            upper[local_i] = u
            ks[local_i] = k
    return lower, upper, ks, row_min, row_max


def _count_nnz_in_ribbon(indptr: np.ndarray, indices: np.ndarray,
                         row_start: int, row_end: int,
                         d: int, L: int, U: int, n_cols: int) -> int:
    """Count nonzeros inside the [d-L, d+U] ribbon over [row_start, row_end)."""
    total = 0
    for i in range(row_start, row_end):
        center = i + d
        col_lo = max(0, center - L)
        col_hi = min(n_cols - 1, center + U)
        if col_hi < col_lo:
            continue
        a = indptr[i]
        b = indptr[i + 1]
        if a == b:
            continue
        row_cols = indices[a:b]
        lo_idx = np.searchsorted(row_cols, col_lo, side='left')
        hi_idx = np.searchsorted(row_cols, col_hi, side='right')
        total += int(hi_idx - lo_idx)
    return total


def grow_segments(lower: np.ndarray, upper: np.ndarray, ks: np.ndarray,
                  csr: csr_matrix, d: int, row_min: int,
                  min_density: float = MIN_DENSITY,
                  min_segment_len: int = MIN_SEGMENT_LEN) -> List[Segment]:
    """Sweep rows left-to-right, building segments whose true (recomputed)
    density on the evolving (L, U) bounding ribbon stays >= min_density.
    A segment starts at the first band-good row after a non-band-good run,
    and ends when either (a) the current row is not band-good, or (b) adding
    it would drop segment density below min_density.
    """
    n = lower.size
    if n == 0:
        return []
    indptr = csr.indptr
    indices = csr.indices
    n_cols = csr.shape[1]

    segments: List[Segment] = []
    i = 0
    while i < n:
        if ks[i] == 0:
            i += 1
            continue
        seg_start_local = i
        seg_L = int(lower[i])
        seg_U = int(upper[i])
        global_start = row_min + seg_start_local
        seg_nnz = _count_nnz_in_ribbon(indptr, indices,
                                       global_start, global_start + 1,
                                       d, seg_L, seg_U, n_cols)
        j = i + 1
        while j < n and ks[j] > 0:
            cand_L = max(seg_L, int(lower[j]))
            cand_U = max(seg_U, int(upper[j]))
            global_j = row_min + j
            add_nnz = _count_nnz_in_ribbon(indptr, indices,
                                           global_j, global_j + 1,
                                           d, cand_L, cand_U, n_cols)
            if cand_L > seg_L or cand_U > seg_U:
                extra = _count_nnz_in_ribbon(
                    indptr, indices, global_start, global_j,
                    d, cand_L, cand_U, n_cols)
                cand_nnz = extra + add_nnz
            else:
                cand_nnz = seg_nnz + add_nnz
            cand_len = j - seg_start_local + 1
            cand_area = (cand_L + cand_U + 1) * cand_len
            if cand_nnz / cand_area >= min_density:
                seg_L, seg_U, seg_nnz = cand_L, cand_U, cand_nnz
                j += 1
            else:
                break
        seg_len = j - seg_start_local
        if seg_len >= min_segment_len and seg_nnz > 0:
            segments.append(Segment(
                row_start=row_min + seg_start_local,
                row_end=row_min + j,
                lower_bw=seg_L,
                upper_bw=seg_U,
                nnz=seg_nnz,
            ))
        i = j
    return segments


def evaluate_region(d: int, segments: List[Segment]) -> Optional[VdiaRegion]:
    """Wrap segments into a VdiaRegion with derived area metrics."""
    if not segments:
        return None
    row_start = segments[0].row_start
    row_end = segments[-1].row_end
    total_rows_ribbon = row_end - row_start
    covered_rows = sum(s.length for s in segments)
    L_max = max(s.lower_bw for s in segments)
    U_max = max(s.upper_bw for s in segments)
    width_max = L_max + U_max + 1

    return VdiaRegion(
        diag_offset=d,
        segments=segments,
        dia_ribbon_area=width_max * total_rows_ribbon,
        dia_covered_area=width_max * covered_rows,
    )


# ---------------------------------------------------------------------------
# Compiled search (optional)
# ---------------------------------------------------------------------------
# The search above is pure Python per row and per candidate diagonal, which
# takes minutes on a 400k-nonzero matrix and days at 15M. When numba is
# installed, find_vdia_regions runs these ports of compute_row_ribbon,
# compute_row_ribbons_for_diag, _count_nnz_in_ribbon and grow_segments
# instead; they return the same regions (the per-row walk takes offsets in
# the order the stable argsort of |offset| gives, ties negative first). They
# need sorted column indices, like _count_nnz_in_ribbon; a matrix without
# them uses the Python path. Set USE_COMPILED = False to force that path.

USE_COMPILED = True

try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover - numba is optional
    _njit = None

if _njit is not None:
    @_njit(cache=True)
    def _nb_row_ribbon(cols, a, b, center, max_bw, min_density, min_k):
        lo, hi = a, b
        while lo < hi:
            mid = (lo + hi) // 2
            if cols[mid] < center:
                lo = mid + 1
            else:
                hi = mid
        li, ri = lo - 1, lo
        l = u = k = 0
        bl = bu = bk = 0
        while li >= a or ri < b:
            if li >= a and ri < b:
                offl = cols[li] - center
                offr = cols[ri] - center
                if -offl <= offr:
                    off = offl
                    li -= 1
                else:
                    off = offr
                    ri += 1
            elif li >= a:
                off = cols[li] - center
                li -= 1
            else:
                off = cols[ri] - center
                ri += 1
            if abs(off) > max_bw:
                break
            nl = max(l, -off) if off < 0 else l
            nu = max(u, off) if off > 0 else u
            nk = k + 1
            if nk / (nl + nu + 1) >= min_density:
                l, u, k = nl, nu, nk
                if k >= min_k:
                    bl, bu, bk = l, u, k
            else:
                break
        return bl, bu, bk

    @_njit(cache=True)
    def _nb_ribbons(indptr, indices, n_rows, n_cols, d, max_bw, row_used, min_density, min_k):
        if d >= 0:
            row_min, row_max = 0, min(n_rows, n_cols - d)
        else:
            row_min, row_max = max(0, -d), min(n_rows, n_cols - d)
        num = max(row_max - row_min, 0)
        lower = np.zeros(num, np.int32)
        upper = np.zeros(num, np.int32)
        ks = np.zeros(num, np.int32)
        for li in range(num):
            i = row_min + li
            if row_used[i]:
                continue
            s, e = indptr[i], indptr[i + 1]
            if s >= e:
                continue
            l, u, k = _nb_row_ribbon(indices, s, e, i + d, max_bw, min_density, min_k)
            if k >= min_k:
                lower[li], upper[li], ks[li] = l, u, k
        return lower, upper, ks, row_min, row_max

    @_njit(cache=True)
    def _nb_count(indptr, indices, r0, r1, d, L, U, n_cols):
        total = 0
        for i in range(r0, r1):
            c = i + d
            clo, chi = max(0, c - L), min(n_cols - 1, c + U)
            if chi < clo:
                continue
            a, b = indptr[i], indptr[i + 1]
            if a == b:
                continue
            x, y = a, b
            while x < y:
                mid = (x + y) // 2
                if indices[mid] < clo:
                    x = mid + 1
                else:
                    y = mid
            first = x
            x, y = a, b
            while x < y:
                mid = (x + y) // 2
                if indices[mid] <= chi:
                    x = mid + 1
                else:
                    y = mid
            total += x - first
        return total

    @_njit(cache=True)
    def _nb_grow(lower, upper, ks, indptr, indices, n_cols, d, row_min, min_density, min_segment_len):
        n = lower.size
        out = []
        i = 0
        while i < n:
            if ks[i] == 0:
                i += 1
                continue
            s0 = i
            L, U = int(lower[i]), int(upper[i])
            g0 = row_min + s0
            nnz = _nb_count(indptr, indices, g0, g0 + 1, d, L, U, n_cols)
            j = i + 1
            while j < n and ks[j] > 0:
                cL, cU = max(L, int(lower[j])), max(U, int(upper[j]))
                gj = row_min + j
                add = _nb_count(indptr, indices, gj, gj + 1, d, cL, cU, n_cols)
                if cL > L or cU > U:
                    cn = _nb_count(indptr, indices, g0, gj, d, cL, cU, n_cols) + add
                else:
                    cn = nnz + add
                if cn / ((cL + cU + 1) * (j - s0 + 1)) >= min_density:
                    L, U, nnz = cL, cU, cn
                    j += 1
                else:
                    break
            if j - s0 >= min_segment_len and nnz > 0:
                out.append((row_min + s0, row_min + j, L, U, nnz))
            i = j
        return out


def compute_diag_mass(csr: csr_matrix, max_diag: int) -> Dict[int, int]:
    """Return {d: nnz on diagonal d} for |d| <= max_diag."""
    coo = csr.tocoo()
    offs = coo.col.astype(np.int64) - coo.row.astype(np.int64)
    mask = np.abs(offs) <= max_diag
    offs = offs[mask]
    if offs.size == 0:
        return {}
    uniq, counts = np.unique(offs, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def find_vdia_regions(csr: csr_matrix,
                      min_density: float = MIN_DENSITY,
                      min_segment_len: int = MIN_SEGMENT_LEN,
                      min_region_len: int = MIN_REGION_LEN,
                      min_region_area: int = MIN_REGION_AREA,
                      min_savings: float = MIN_SAVINGS,
                      max_diag: Optional[int] = None,
                      max_bw: Optional[int] = None,
                      verbose: bool = False,
                      deadline: Optional[float] = None) -> List[VdiaRegion]:
    """Main entry point: find all accepted VDIA regions in a matrix. With a
    `deadline` (a time.perf_counter() value), raise SearchTimeout once the
    search of the candidate diagonals passes it."""
    search_start = time.perf_counter()
    n_rows, n_cols = csr.shape
    small_dim = min(n_rows, n_cols)
    if max_diag is None:
        max_diag = max(50, min(500, small_dim // 4))
    if max_bw is None:
        max_bw = max(50, min(500, small_dim // 4))

    diag_mass = compute_diag_mass(csr, max_diag)
    min_mass = max(MIN_ROW_NNZ * min_region_len // 4, 20)
    candidates = [d for d, m in diag_mass.items() if m >= min_mass]
    candidates.sort(key=lambda d: -diag_mass[d])

    row_used = np.zeros(n_rows, dtype=bool)
    accepted: List[VdiaRegion] = []
    compiled = USE_COMPILED and _njit is not None and csr.has_sorted_indices
    if compiled:
        indptr = csr.indptr.astype(np.int64)
        indices = csr.indices.astype(np.int64)

    for d in candidates:
        if deadline is not None and time.perf_counter() > deadline:
            raise SearchTimeout
        if compiled:
            lower, upper, ks, row_min, row_max = _nb_ribbons(
                indptr, indices, n_rows, n_cols, d, max_bw, row_used, float(min_density), MIN_ROW_NNZ)
        else:
            lower, upper, ks, row_min, row_max = compute_row_ribbons_for_diag(
                csr, d, max_bw, row_used, min_density)
        if row_max - row_min < min_region_len:
            continue
        if ks.sum() == 0:
            continue
        if compiled:
            segments = [Segment(row_start=a, row_end=b, lower_bw=L, upper_bw=U, nnz=z)
                        for a, b, L, U, z in _nb_grow(lower, upper, ks, indptr, indices, n_cols, d, row_min,
                                                      float(min_density), min_segment_len)]
        else:
            segments = grow_segments(lower, upper, ks, csr, d, row_min,
                                      min_density, min_segment_len)
        if not segments:
            continue
        region = evaluate_region(d, segments)
        if region is None:
            continue
        if region.vdia_area < min_region_area:
            continue
        if region.density < min_density:
            continue
        if region.fill_ribbon > 1.0 - min_savings:
            continue
        region.found_at_seconds = time.perf_counter() - search_start
        accepted.append(region)
        for s in segments:
            row_used[s.row_start:s.row_end] = True
        if verbose:
            print(f"  accept d={d}: rows=[{region.row_start},{region.row_end}] "
                  f"nsegs={len(segments)} Lmax={max(s.lower_bw for s in segments)} "
                  f"Umax={max(s.upper_bw for s in segments)} "
                  f"density={region.density:.3f} "
                  f"fill_ribbon={region.fill_ribbon:.3f} "
                  f"fill_covered={region.fill_covered:.3f} "
                  f"save_vs_DIA_ribbon={region.savings_vs_dia_ribbon:.3f} "
                  f"save_vs_DIA_covered={region.savings_vs_dia_covered:.3f}")
    accepted.sort(key=lambda r: -(r.dia_ribbon_area - r.vdia_area))
    return accepted


# ---------------------------------------------------------------------------
# Band selection for the VDIA kernels
# ---------------------------------------------------------------------------
# find_vdia_regions accepts a band by density alone. What a band is worth
# depends on the kernel that computes it: a stored slot of a band of w
# diagonals costs that kernel as much time as CSR spends on slot_cost(w)
# nonzeros, so a segment of area a holding z nonzeros saves
# z - slot_cost(w) * a nonzeros' worth of CSR time. select_bands makes one
# selection for every kernel:
#   candidates: every segment find_vdia_regions grows at each density floor
#     in BAND_FLOORS, over every diagonal offset and width (min_savings 0, no
#     minimum region area);
#   selection: the candidates in decreasing savings while the savings are
#     positive, skipping any that shares rows with one already taken;
#   acceptance: the selected segments hold at least min_coverage of the
#     matrix's nonzeros, and the search finished within timeout_seconds.
# The operations differ only in slot_cost and min_coverage:
#   SpMV (NaiveVDIASpmv): a slot saves only a column index, so slot_cost is
#     SPMV_SLOT_COST at every width; min_coverage SPMV_MIN_COVERAGE.
#   SpMM (NaiveVDIASpmm, 512 columns): spmm_slot_cost, measured on synthetic
#     bands: 3 diagonals 0.90, 7 0.82, 21 0.59, 61 0.42, 111 0.38 (log-linear
#     in between, 0.95 below 3); min_coverage SPMM_MIN_COVERAGE.
# Both coverage cutoffs and the SpMV slot cost carry over from the earlier
# per-operation selectors. Every input is static but the time limit: on the
# 856 real square SuiteSparse matrices with 1e5-1.5e7 nonzeros, SpMV gets
# bands on 43 and SpMM on 49, and the slowest of these finishes the search in
# 34 s, while 29 matrices without bands (graphs and large FEM matrices with
# 1e4-2e5 candidate diagonals) take longer than BAND_TIMEOUT_SECONDS.

SPMM_SLOT_COST_POINTS = ((3, 0.90), (7, 0.82), (21, 0.59), (61, 0.42), (111, 0.38))
SPMV_SLOT_COST = 0.90
SPMV_MIN_COVERAGE = 0.35
SPMM_MIN_COVERAGE = 0.60
BAND_FLOORS = (0.90, 0.75, 0.60, 0.45)
BAND_TIMEOUT_SECONDS = 600.0


def spmm_slot_cost(ndiags: int) -> float:
    """Break-even density of a band of `ndiags` diagonals for the tiled SpMM kernel."""
    pts = SPMM_SLOT_COST_POINTS
    if ndiags < pts[0][0]:
        return 0.95
    if ndiags >= pts[-1][0]:
        return pts[-1][1]
    for (w0, c0), (w1, c1) in zip(pts, pts[1:]):
        if w0 <= ndiags <= w1:
            f = (np.log(ndiags) - np.log(w0)) / (np.log(w1) - np.log(w0))
            return float(c0 + f * (c1 - c0))
    return pts[-1][1]


def spmv_slot_cost(ndiags: int) -> float:
    """Break-even density of a band of `ndiags` diagonals for the register-blocked SpMV kernel."""
    return SPMV_SLOT_COST


def band_candidates(csr: csr_matrix,
                    floors: Tuple[float, ...] = BAND_FLOORS,
                    timeout_seconds: Optional[float] = BAND_TIMEOUT_SECONDS) -> List[Tuple[int, Segment]]:
    """(diagonal offset, segment) for every segment find_vdia_regions grows at
    each density floor in `floors` over every offset and width. Raises
    SearchTimeout when the search runs longer than `timeout_seconds` (None:
    no limit)."""
    span = max(min(csr.shape) - 1, 1)
    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds
    out = []
    for floor in floors:
        for region in find_vdia_regions(csr, min_density=floor, min_segment_len=MIN_SEGMENT_LEN, min_region_area=0,
                                        min_savings=0.0, max_diag=span, max_bw=span, deadline=deadline):
            out += [(region.diag_offset, seg) for seg in region.segments]
    return out


def select_bands(csr: csr_matrix,
                 slot_cost: Callable[[int], float],
                 min_coverage: float,
                 floors: Tuple[float, ...] = BAND_FLOORS,
                 timeout_seconds: Optional[float] = BAND_TIMEOUT_SECONDS,
                 candidates: Optional[List[Tuple[int, Segment]]] = None,
                 info: Optional[Dict[str, Any]] = None) -> List[VdiaRegion]:
    """Bands for a VDIA kernel whose stored slot of a band of w diagonals costs
    slot_cost(w) CSR nonzeros: row-disjoint regions of one segment each, in
    row order, or [] when they would hold less than `min_coverage` of the
    nonzeros or the search runs longer than `timeout_seconds` (None: no
    limit). `candidates`, from band_candidates(csr, floors), lets several
    parameter sets share one search. `info`, when given, receives the segment
    count, coverage, density, search seconds and the choice (bands, none or
    timeout)."""
    nnz = int(csr.nnz)
    start = time.perf_counter()
    if candidates is None:
        try:
            candidates = band_candidates(csr, floors, timeout_seconds)
        except SearchTimeout:
            if info is not None:
                info.update(segments=0, coverage=0.0, density=0.0, min_coverage=min_coverage,
                            seconds=round(time.perf_counter() - start, 1), choice="timeout")
            return []
    scored = []
    for diag_offset, seg in candidates:
        width = seg.lower_bw + seg.upper_bw + 1
        saved = seg.nnz - slot_cost(width) * width * seg.length
        scored.append((saved, seg.row_start, seg.length, diag_offset - seg.lower_bw, width, diag_offset, seg))
    scored.sort(key=lambda c: c[:5], reverse=True)
    used = np.zeros(csr.shape[0], dtype=bool)
    picked: List[Tuple[int, Segment]] = []
    for saved, *_, diag_offset, seg in scored:
        if saved <= 0:
            break
        if not used[seg.row_start:seg.row_end].any():
            used[seg.row_start:seg.row_end] = True
            picked.append((diag_offset, seg))
    claimed = sum(seg.nnz for _, seg in picked)
    area = sum(seg.area for _, seg in picked)
    coverage = claimed / nnz if nnz else 0.0
    accept = bool(picked) and coverage >= min_coverage
    if info is not None:
        info.update(segments=len(picked), coverage=round(coverage, 4),
                    density=round(claimed / area, 4) if area else 0.0, min_coverage=min_coverage,
                    seconds=round(time.perf_counter() - start, 1), choice="bands" if accept else "none")
    if not accept:
        return []
    return [evaluate_region(d, [seg]) for d, seg in sorted(picked, key=lambda p: p[1].row_start)]


def get_matrix_info(name: str):
    found = search(name_or_id=name)
    filtered = [m for m in found if m.name == name]
    return filtered[0] if len(filtered) == 1 else None


def get_paths(info):
    lp = info.localpath()
    tar = lp[0] if isinstance(lp, tuple) else lp
    subdir = os.path.join(os.path.dirname(tar), info.name)
    mtx = os.path.join(subdir, f"{info.name}.mtx")
    return tar, subdir, mtx


def cleanup(tar: str, subdir: str) -> None:
    if os.path.exists(tar):
        os.remove(tar)
    if os.path.exists(subdir):
        shutil.rmtree(subdir)


def load_matrix(name: str) -> Optional[csr_matrix]:
    info = get_matrix_info(name)
    if info is None:
        return None
    fetch(name)
    tar, subdir, mtx = get_paths(info)
    if not os.path.exists(mtx):
        cleanup(tar, subdir)
        return None
    A = mmread(mtx)
    csr = csr_matrix(A)
    return csr


def process_matrix(name: str, verbose: bool = True,
                   timeout_seconds: int = 600,
                   min_density: float = MIN_DENSITY,
                   spmm: bool = False,
                   spmv: bool = False) -> Tuple[str, Optional[Dict]]:
    start = time.time()
    csr = load_matrix(name)
    if csr is None:
        return name, None
    if verbose:
        print(f"[{name}] loaded {csr.shape[0]}x{csr.shape[1]} nnz={csr.nnz}")
    if spmm:
        regions = select_bands(csr, spmm_slot_cost, SPMM_MIN_COVERAGE)
    elif spmv:
        regions = select_bands(csr, spmv_slot_cost, SPMV_MIN_COVERAGE)
    else:
        regions = find_vdia_regions(csr, min_density=min_density, verbose=verbose)
    elapsed = time.time() - start
    timeout = elapsed > timeout_seconds
    return name, {
        'timeout': timeout,
        'has_bands': len(regions) > 0,
        'bands': [r.to_dict() for r in regions],
        'matrix_shape': list(csr.shape),
        'matrix_nnz': int(csr.nnz),
        'elapsed_seconds': round(elapsed, 2),
    }


def save_results(name: str, result: Dict, output_dir: str = "results_bands") -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.yaml")
    with open(path, 'w') as f:
        f.write(f"timeout: {str(result.get('timeout', False)).lower()}\n")
        f.write(f"matrix_shape: [{result.get('matrix_shape', [0,0])[0]}, "
                f"{result.get('matrix_shape', [0,0])[1]}]\n")
        f.write(f"matrix_nnz: {result.get('matrix_nnz', 0)}\n")
        f.write(f"elapsed_seconds: {result.get('elapsed_seconds', 0)}\n")
        bands = result.get('bands', [])
        if not bands:
            f.write("bands: []\n")
            return
        f.write("bands:\n")
        for b in bands:
            f.write(f"  - rows: [{b['rows'][0]}, {b['rows'][1]}]\n")
            f.write(f"    diag_offset: {b['diag_offset']}\n")
            f.write(f"    total_nnz: {b['total_nnz']}\n")
            f.write(f"    vdia_area: {b['vdia_area']}\n")
            f.write(f"    dia_ribbon_area: {b['dia_ribbon_area']}\n")
            f.write(f"    dia_covered_area: {b['dia_covered_area']}\n")
            f.write(f"    density: {b['density']}\n")
            f.write(f"    fill_ribbon: {b['fill_ribbon']}\n")
            f.write(f"    fill_covered: {b['fill_covered']}\n")
            f.write(f"    savings_vs_dia_ribbon: {b['savings_vs_dia_ribbon']}\n")
            f.write(f"    savings_vs_dia_covered: {b['savings_vs_dia_covered']}\n")
            f.write(f"    found_at_seconds: {b.get('found_at_seconds', 0.0)}\n")
            f.write(f"    segments:\n")
            for s in b['segments']:
                f.write(f"      - rows: [{s['rows'][0]}, {s['rows'][1]}]\n")
                f.write(f"        bandwidth: [{s['bandwidth'][0]}, {s['bandwidth'][1]}]\n")
                f.write(f"        nnz: {s['nnz']}\n")
                f.write(f"        area: {s['area']}\n")


def _worker(name: str) -> Tuple[str, Optional[Dict]]:
    return process_matrix(name, verbose=False)


def run_batch(names: List[str], num_workers: int = 4,
              output_dir: str = "results_bands") -> None:
    ctx = get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        for name, result in pool.imap_unordered(_worker, names):
            if result is None:
                print(f"[{name}] failed")
                continue
            save_results(name, result, output_dir)
            n = len(result.get('bands', []))
            print(f"[{name}] wrote {n} VDIA region(s) "
                  f"(elapsed {result.get('elapsed_seconds', 0)}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find VDIA regions in sparse matrices")
    parser.add_argument("matrices", nargs="*", default=[],
                        help="Matrix names to process")
    parser.add_argument("--min-density", type=float, default=MIN_DENSITY,
                        help=f"Minimum density threshold (default: {MIN_DENSITY})")
    parser.add_argument("--output-dir", default="results_bands",
                        help="Output directory for YAML results (default: results_bands)")
    parser.add_argument("--spmm", action="store_true",
                        help="Select bands for the tiled SpMM kernel (select_bands with spmm_slot_cost) "
                             "instead of accepting them by density; --min-density is then unused")
    parser.add_argument("--spmv", action="store_true",
                        help="Select bands for the register-blocked SpMV kernel (select_bands with "
                             "spmv_slot_cost); --min-density is then unused")
    args = parser.parse_args()

    matrices = args.matrices
    if not matrices:
        matrices = ["bcsstk13", "bcsstk28", "nd3k", "heart1", "lowThrust_3"]

    for m in matrices:
        rn, result = process_matrix(m, verbose=True, min_density=args.min_density, spmm=args.spmm, spmv=args.spmv)
        if result is not None:
            save_results(rn, result, output_dir=args.output_dir)
            print(f"[{rn}] wrote {len(result['bands'])} region(s)")
        else:
            print(f"[{m}] failed")
