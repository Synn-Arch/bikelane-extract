"""fetch — MassDOT orthophoto tiles for a town → single mosaic GeoTIFF.

Ported from pipeline.ipynb. Uses the MassDOT COQ index shapefile to find the
JP2 tiles intersecting the town's bounding box (town polygon from pygris
county subdivisions), downloads the zips, extracts the .jp2, and builds a
VRT → LZW-tiled BigTIFF with GDAL.

Requires: geopandas, pygris, requests, and gdalbuildvrt/gdal_translate on PATH.
"""
from __future__ import annotations

import os
import subprocess
import zipfile

from ..config import Config


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane fetch")
    ap.parse_args(argv)
    p = cfg.get("imagery")
    index = cfg.path("massdot_index_shp")
    region_dir = cfg.path("imagery_dir") / cfg.region
    mosaic = cfg.path("mosaic_tif")
    print(f"fetch  {cfg.region} ({cfg.get('city')}, {cfg.get('state')})\n  index: {index}\n  mosaic → {mosaic}")
    if check:
        print(f"  {'ok ' if index.exists() else 'MISSING'} {index}")
        print(f"  gdal: {'ok' if subprocess.run('which gdal_translate', shell=True, capture_output=True).returncode == 0 else 'MISSING'}")
        return
    import geopandas as gpd
    import pygris
    import requests
    from shapely.geometry import box
    from tqdm import tqdm

    idx = gpd.read_file(index)
    towns = pygris.county_subdivisions(state=cfg.get("state"), year=p["towns_year"])
    town = towns[towns["NAME"] == cfg.get("city")].to_crs(idx.crs)
    if town.empty:
        raise SystemExit(f"town '{cfg.get('city')}' not found in county subdivisions")
    overlap = idx[idx.intersects(box(*town.total_bounds))]
    print(f"  {len(overlap)} index tiles intersect the town bbox")

    zip_dir, img_dir = region_dir / "ZipFiles", region_dir / "Images"
    zip_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    for url in tqdm(overlap["URL"], desc="  download"):
        dst = zip_dir / url.split("/")[-1]
        if dst.exists():
            continue
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            print(f"\n  [error] {url}: {e}")
    for zp in tqdm(sorted(zip_dir.glob("*.zip")), desc="  extract"):
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(".jp2"):
                        dst = img_dir / os.path.basename(info.filename)
                        if not dst.exists():
                            with z.open(info) as s, open(dst, "wb") as d:
                                d.write(s.read())
        except zipfile.BadZipFile:
            print(f"\n  [error] corrupt zip: {zp.name}")

    jp2 = sorted(str(f) for f in img_dir.glob("*.jp2"))
    vrt = mosaic.with_suffix(".vrt")
    subprocess.run(["gdalbuildvrt", str(vrt)] + jp2, check=True)
    subprocess.run(["gdal_translate", str(vrt), str(mosaic), "-co", "COMPRESS=LZW",
                    "-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS"],
                   check=True)
    print(f"  mosaic written: {mosaic}\n→ next: bikelane tile")
