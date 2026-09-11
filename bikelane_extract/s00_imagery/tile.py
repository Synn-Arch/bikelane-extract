"""tile — mosaic (GeoTIFF or VRT) → 1024 px JPG tiles (25 % overlap) + Tile_Mappings.csv

Ported from pipeline.ipynb. Tiles with ≥ empty_threshold nodata are dropped
(0.5; the earlier value 0 silently dropped 70 % of Boston). Filenames are
tile_px<X>_py<Y>.jpg with X/Y the global mosaic pixel offset; the CSV maps
each to the CRS coordinate of its top-left corner. Every later stage relies
on this file and naming.

Resumable: Tile_Mappings.csv (kept tiles) and tiles_dropped.txt (nodata
tiles) are appended to, and tiles listed in either are skipped on re-run.
Each worker opens the mosaic once (a VRT over ~2,000 JP2s is expensive to
open) with a small GDAL cache, so memory stays ~0.5 GB per worker.
"""
from __future__ import annotations

import csv
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..config import Config

_W = {}


def _init(path: str):
    os.environ.setdefault("GDAL_CACHEMAX", "256")          # MB, per process
    os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "64")
    import rasterio
    _W["src"] = rasterio.open(path)


def _tile(args):
    import cv2
    from rasterio.windows import Window
    out_dir, x, y, size, empty_th = args
    src = _W["src"]
    t = src.read(window=Window(x, y, size, size))
    ch = src.count
    sx, sy = src.transform * (x, y)
    empty = np.sum(t[3] == 0) if ch == 4 else np.sum(np.all(t[:3] == 0, axis=0))
    if empty / (size * size) >= empty_th:
        return None, x, y
    hwc = np.transpose(t, (1, 2, 0))
    bgr = (cv2.cvtColor(hwc, cv2.COLOR_RGBA2BGR) if ch == 4 else
           cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR) if ch >= 3 else hwc)
    name = f"tile_px{x}_py{y}.jpg"
    cv2.imwrite(str(Path(out_dir) / name), bgr)
    return [name, sx, sy, x, y], x, y


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane tile")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default min(cpu, 16); ~0.5 GB RAM each)")
    ap.add_argument("--cleanup", action="store_true",
                    help="after tiling completes, delete the downloaded zips, JP2s and VRT "
                         "(also imagery.cleanup_source: true). Tiles + Tile_Mappings.csv are "
                         "all later stages need; re-run `fetch` to get the sources back.")
    a = ap.parse_args(argv)
    p = cfg.get("imagery")
    mosaic = cfg.path("mosaic_tif")
    if not mosaic.exists() and mosaic.with_suffix(".vrt").exists():
        mosaic = mosaic.with_suffix(".vrt")             # fetch without build_tif
    out_dir = cfg.path("tiles_dir", mkdir=not check)
    csv_path = cfg.path("tile_mappings_csv")
    dropped_path = csv_path.with_name("tiles_dropped.txt")
    print(f"tile  {cfg.region}\n  mosaic: {mosaic}\n  tiles → {out_dir}\n  mapping → {csv_path}")
    if check:
        print(f"  {'ok ' if mosaic.exists() else 'MISSING'} {mosaic}")
        return
    if cfg.get("boundary", None):
        print("  note: tiles outside the boundary polygon but inside its bbox are also cut "
              "(only nodata tiles are dropped); clip outputs to the boundary afterwards if needed")
    import rasterio
    size = cfg.tile_px
    stride = int(size * (1 - p["tile_overlap"]))
    with rasterio.open(mosaic) as src:
        w, h, tr = src.width, src.height, src.transform
        if abs(tr.a - cfg.resolution_m) > 1e-3:
            print(f"  WARNING: mosaic pixel size {tr.a} ≠ config resolution_m {cfg.resolution_m}")

    # resume: skip anything already decided
    done = set()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((int(r["Pixel_X"]), int(r["Pixel_Y"])))
    if dropped_path.exists():
        for line in dropped_path.read_text().split():
            x, y = line.split(",")
            done.add((int(x), int(y)))
    tasks = [(str(out_dir), x, y, size, p["empty_threshold"])
             for y in range(0, h - size + 1, stride) for x in range(0, w - size + 1, stride)
             if (x, y) not in done]
    n_all = len(tasks) + len(done)
    workers = a.workers or p.get("workers") or min(multiprocessing.cpu_count(), 16)
    workers = max(1, min(int(workers), len(tasks) or 1))
    print(f"  {w}×{h} px, stride {stride} → {n_all} candidate tiles, {len(done)} already done, "
          f"{len(tasks)} to do, {workers} workers")
    cleanup = a.cleanup or bool(p.get("cleanup_source", False))
    if not tasks:
        if cleanup:
            _cleanup_sources(mosaic)
        print("→ next: bikelane predict")
        return

    saved = dropped = 0
    new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
    ctx = multiprocessing.get_context("fork")
    from concurrent.futures.process import BrokenProcessPool
    BATCH = 2000                       # futures in flight; keeps the parent small
    restarts = 0
    with open(csv_path, "a", newline="") as f, open(dropped_path, "a") as fd:
        wr = csv.writer(f)
        if new_csv:
            wr.writerow(["image_name", "CRS_X", "CRS_Y", "Pixel_X", "Pixel_Y"])
        pending = list(tasks)
        done_n = 0
        while pending:
            batch, pending = pending[:BATCH], pending[BATCH:]
            try:
                with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init,
                                         initargs=(str(mosaic),)) as ex:
                    for r, x, y in ex.map(_tile, batch, chunksize=8):
                        if r is None:
                            dropped += 1
                            fd.write(f"{x},{y}\n")
                        else:
                            wr.writerow(r)
                            saved += 1
                        done_n += 1
                        if done_n % 1000 == 0:
                            f.flush(); fd.flush()
                            print(f"  {done_n}/{len(tasks)} (saved {saved}, dropped {dropped})", flush=True)
            except BrokenProcessPool:
                # a worker was killed (usually OOM while decoding a JP2). Everything written so
                # far is on disk; redo this batch with a fresh pool and fewer workers.
                f.flush(); fd.flush()
                restarts += 1
                done_set = set()
                if csv_path.exists():
                    with open(csv_path) as fr:
                        done_set |= {(int(r["Pixel_X"]), int(r["Pixel_Y"])) for r in csv.DictReader(fr)}
                for line in dropped_path.read_text().split():
                    x_, y_ = line.split(",")
                    done_set.add((int(x_), int(y_)))
                batch = [t for t in batch if (t[1], t[2]) not in done_set]
                pending = batch + pending
                workers = max(2, workers * 2 // 3)
                print(f"  worker pool died (restart {restarts}); continuing with {workers} workers, "
                      f"{len(pending)} tiles left", flush=True)
                if restarts > 20:
                    raise SystemExit("tile: too many worker crashes — check dmesg for OOM, "
                                     "re-run with --workers 4")
        f.flush(); fd.flush()
    print(f"  saved {saved} tiles, dropped {dropped} (≥{p['empty_threshold']:.0%} empty); "
          f"total kept {len([1 for _ in open(csv_path)]) - 1}")
    if cleanup:
        _cleanup_sources(mosaic)
    print("→ next: bikelane predict")


def _cleanup_sources(mosaic: Path):
    """Delete ZipFiles/, Images/ (JP2), the VRT and its file list next to the mosaic."""
    import shutil
    region_dir = mosaic.parent
    freed = 0
    for sub in ("ZipFiles", "Images"):
        d = region_dir / sub
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    for f in (mosaic.with_suffix(".vrt"), mosaic.with_suffix(".txt"), mosaic.with_suffix(".tif")):
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    print(f"  cleanup: removed source imagery under {region_dir} ({freed/1e9:.0f} GB freed)")
