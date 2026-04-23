"""List matrices and categorise them by which sparse format(s) they benefit from.

Categories:
  VDIA + VBR  — has >=1 VDIA region AND >=1 VBR block, with the two formats
                covering complementary (non-overlapping) parts of the matrix.
  VDIA-only   — has >=1 accepted VDIA region, no VBR blocks.
  VBR-only    — has >=1 VBR block, no VDIA regions.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
VBR_DIR = os.path.join(ROOT, "results")
VDIA_DIR = os.path.join(ROOT, "results_bands")


def _load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_vdia(name: str):
    p = os.path.join(VDIA_DIR, f"{name}.yaml")
    if not os.path.exists(p):
        return None
    return _load(p)


def _load_vbr(name: str):
    p = os.path.join(VBR_DIR, f"{name}.yaml")
    if not os.path.exists(p):
        return None
    return _load(p)


def _segment_rects(segments: List[Dict], diag_offset: int,
                   n_cols: int) -> List[Tuple[int, int, int, int]]:
    rects = []
    for s in segments:
        r0, r1 = s['rows']
        l, u = s['bandwidth']
        col_lo = max(0, r0 + diag_offset - l)
        col_hi = min(n_cols - 1, (r1 - 1) + diag_offset + u)
        rects.append((r0, r1, col_lo, col_hi + 1))
    return rects


def _rects_overlap(a, b) -> int:
    r0a, r1a, c0a, c1a = a
    r0b, r1b, c0b, c1b = b
    dr = max(0, min(r1a, r1b) - max(r0a, r0b))
    dc = max(0, min(c1a, c1b) - max(c0a, c0b))
    return dr * dc


def _bw_varies(segments: List[Dict]) -> bool:
    if len(segments) < 2:
        return False
    return len({tuple(s['bandwidth']) for s in segments}) > 1


def _rect_area(r) -> int:
    return (r[1] - r[0]) * (r[3] - r[2])


def _vbr_block_nnz(block: Dict) -> int:
    area = (block['rows'][1] - block['rows'][0]) * (block['cols'][1] - block['cols'][0])
    return int(round(area * block.get('density', 1.0)))


def analyse(name: str) -> Dict[str, Any] | None:
    vdia_data = _load_vdia(name)
    vbr_data = _load_vbr(name)

    bands = (vdia_data or {}).get('bands') or []
    blocks = (vbr_data or {}).get('blocks') or []
    if not bands and not blocks:
        return None

    shape = ((vdia_data or {}).get('matrix_shape')
             or (vbr_data or {}).get('matrix_shape'))
    total_nnz = ((vdia_data or {}).get('matrix_nnz')
                 or (vbr_data or {}).get('matrix_nnz', 0))
    n_cols = shape[1] if shape else 0

    vdia_nnz = sum(b['total_nnz'] for b in bands)
    vdia_saving = sum(b['dia_ribbon_area'] - b['vdia_area'] for b in bands)
    bw_varies = any(_bw_varies(b['segments']) for b in bands)

    vbr_nnz = sum(_vbr_block_nnz(b) for b in blocks)
    vbr_rects = [
        (b['rows'][0], b['rows'][1], b['cols'][0], b['cols'][1])
        for b in blocks
    ]
    vbr_largest = max((_rect_area(r) for r in vbr_rects), default=0)

    disjoint_score = None
    overlap_area = 0
    if bands and blocks and n_cols:
        vdia_rects = []
        for b in bands:
            vdia_rects.extend(_segment_rects(b['segments'], b['diag_offset'], n_cols))
        vdia_area = sum(_rect_area(r) for r in vdia_rects)
        vbr_area = sum(_rect_area(r) for r in vbr_rects)
        for a in vdia_rects:
            for b in vbr_rects:
                overlap_area += _rects_overlap(a, b)
        disjoint_score = 1.0 - overlap_area / max(1, vbr_area + vdia_area)

    has_vdia = len(bands) > 0
    has_vbr = len(blocks) > 0
    if has_vdia and has_vbr:
        category = 'VDIA+VBR'
    elif has_vdia:
        category = 'VDIA-only'
    else:
        category = 'VBR-only'

    return {
        'matrix': name,
        'category': category,
        'shape': shape,
        'nnz': total_nnz,
        'n_regions': len(bands),
        'n_segments': sum(len(b['segments']) for b in bands),
        'vdia_nnz': vdia_nnz,
        'vdia_saving': vdia_saving,
        'bw_varies': bw_varies,
        'n_vbr_blocks': len(blocks),
        'vbr_nnz': vbr_nnz,
        'vbr_largest': vbr_largest,
        'overlap_area': overlap_area,
        'disjoint_score': disjoint_score,
    }


def _fmt(r: Dict) -> str:
    parts = [f"  {r['matrix']:30s}"]
    if r['n_regions']:
        parts.append(
            f"VDIA: {r['n_regions']} region(s), {r['n_segments']} seg, "
            f"nnz={r['vdia_nnz']}, bw_varies={'Y' if r['bw_varies'] else 'N'}, "
            f"saving={r['vdia_saving']}")
    if r['n_vbr_blocks']:
        parts.append(
            f"VBR: {r['n_vbr_blocks']} block(s), largest={r['vbr_largest']}")
    if r['disjoint_score'] is not None:
        parts.append(f"disjoint={r['disjoint_score']:.3f}")
    return '  '.join(parts)


def main():
    vdia_names = {f[:-5] for f in os.listdir(VDIA_DIR) if f.endswith('.yaml')}
    vbr_names = {f[:-5] for f in os.listdir(VBR_DIR) if f.endswith('.yaml')}
    all_names = sorted(vdia_names | vbr_names)

    rows = []
    for name in all_names:
        info = analyse(name)
        if info:
            rows.append(info)

    vdia_vbr = sorted(
        [r for r in rows if r['category'] == 'VDIA+VBR'],
        key=lambda r: (r['bw_varies'], r['disjoint_score'] or 0,
                       r['vdia_saving'] + r['vbr_largest']),
        reverse=True,
    )
    vdia_only = sorted(
        [r for r in rows if r['category'] == 'VDIA-only'],
        key=lambda r: -r['vdia_saving'],
    )
    vbr_only = sorted(
        [r for r in rows if r['category'] == 'VBR-only'],
        key=lambda r: -r['vbr_largest'],
    )

    print(f"Total matrices analysed: {len(rows)}\n")

    print(f"=== VDIA + VBR ({len(vdia_vbr)}) ===")
    for r in vdia_vbr:
        print(_fmt(r))

    print(f"\n=== VDIA-only ({len(vdia_only)}) ===")
    for r in vdia_only:
        print(_fmt(r))

    print(f"\n=== VBR-only ({len(vbr_only)}) ===")
    for r in vbr_only:
        print(_fmt(r))

    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        import json
        with open(os.path.join(ROOT, "vbr_plus_vdia.json"), 'w') as f:
            json.dump({'vdia_vbr': vdia_vbr, 'vdia_only': vdia_only,
                       'vbr_only': vbr_only}, f, indent=2)
        print("\nWrote vbr_plus_vdia.json")


if __name__ == "__main__":
    main()
