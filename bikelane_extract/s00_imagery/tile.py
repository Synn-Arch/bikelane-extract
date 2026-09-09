"""tile — mosaic GeoTIFF → 1024 px JPG tiles (25 % overlap) + Tile_Mappings.csv

Ported from pipeline.ipynb. Tiles with ≥ empty_threshold nodata are dropped
(0.5; the earlier value 0 silently dropped 70 % of Boston). Filenames are
tile_px<X>_py<Y>.jpg with X/Y the global mosaic pixel offset; the CSV maps
each to the CRS coordinate of its top-left corner. Every later stage relies
on this file and naming.
"""
from __future__ import annotations

import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..config import Config


def _tile(args):
    import cv2
    import rasterio
    from rasterio.windows import Window
    path, out_dir, x, y, size, transform, empty_th = args
    with rasterio.open(path) as src:
        t = src.read(window=Window(x, y, size, size))
        ch = src.count
    sx, sy = transform * (x, y)
    empty = np.sum(t[3] == 0) if ch == 4 else np.sum(np.all(t[:3] == 0, axis=0))
    if empty / (size * size) >= empty_th:
        return None
    hwc = np.transpose(t, (1, 2, 0))
    bgr = (cv2.cvtColor(hwc, cv2.COLOR_RGBA2BGR) if ch == 4 else
           cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR) if ch >= 3 else hwc)
    name = f"tile_px{x}_py{y}.jpg"
    cv2.imwrite(str(out_dir / name), bgr)
    return [name, sx, sy, x, y]


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane tile")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    a = ap.parse_args(argv)
    p = cfg.get("imagery")
    mosaic = cfg.path("mosaic_tif")
    out_dir = cfg.path("tiles_dir", mkdir=not check)
    csv_path = cfg.path("tile_mappings_csv")
    print(f"tile  {cfg.region}\n  mosaic: {mosaic}\n  tiles → {out_dir}\n  mapping → {csv_path}")
    if check:
        print(f"  {'ok ' if mosaic.exists() else 'MISSING'} {mosaic}")
        return
    import rasterio
    size = cfg.tile_px
    stride = int(size * (1 - p["tile_overlap"]))
    with rasterio.open(mosaic) as src:
        w, h, tr = src.width, src.height, src.transform
        if abs(tr.a - cfg.resolution_m) > 1e-3:
            print(f"  WARNING: mosaic pixel size {tr.a} ≠ config resolution_m {cfg.resolution_m}")
    tasks = [(str(mosaic), out_dir, x, y, size, tr, p["empty_threshold"])
             for y in range(0, h - size + 1, stride) for x in range(0, w - size + 1, stride)]
    print(f"  {w}×{h} px, stride {stride} → {len(tasks)} candidate tiles, {a.workers} workers")
    saved = dropped = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        wr = csv.writer(f)
        wr.writerow(["image_name", "CRS_X", "CRS_Y", "Pixel_X", "Pixel_Y"])
        futs = [ex.submit(_tile, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r is None:
                dropped += 1
            else:
                wr.writerow(r)
                saved += 1
            if i % 1000 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} (saved {saved}, dropped {dropped})")
    print(f"  saved {saved} tiles, dropped {dropped} (≥{p['empty_threshold']:.0%} empty)")
    print("→ next: bikelane predict")
