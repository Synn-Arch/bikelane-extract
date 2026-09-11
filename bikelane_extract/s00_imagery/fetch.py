"""fetch — MassDOT orthophoto tiles for an area → mosaic (VRT, optionally BigTIFF).

Ported from pipeline.ipynb. The area is either a town (`city`/`state`, polygon
from pygris county subdivisions) or any polygon file (`boundary:` in
bikelane.yaml — e.g. a whole MPO region). The index shapefile picks the JP2
tiles intersecting the area, the zips are downloaded and extracted, and a
VRT is built. A BigTIFF is only written when imagery.build_tif is true; for
large areas the VRT is enough (tile reads it directly) and a BigTIFF would
be hundreds of GB.

The index shapefile itself is downloaded from MassGIS on first use
(imagery.index_url). Requires: geopandas, pygris, requests, and
gdalbuildvrt/gdal_translate on PATH.
"""
from __future__ import annotations

import os
import subprocess
import zipfile

from ..config import Config


def _download_index(url: str, shp):
    """Fetch the MassGIS index zip and extract it next to the expected .shp."""
    import io
    import requests
    shp.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading index {url}")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(shp.parent)
    if not shp.exists():                    # zip may nest a folder; find the .shp
        found = next(shp.parent.rglob(shp.name), None)
        if found is None:
            raise SystemExit(f"index zip did not contain {shp.name}")
        for f in found.parent.iterdir():
            f.rename(shp.parent / f.name)
    print(f"  index ready: {shp}")


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane fetch")
    ap.parse_args(argv)
    p = cfg.get("imagery")
    index = cfg.path("massdot_index_shp")
    region_dir = cfg.path("imagery_dir") / cfg.region
    mosaic = cfg.path("mosaic_tif")
    area = cfg.get("boundary", None) or f"{cfg.get('city')}, {cfg.get('state')}"
    print(f"fetch  {cfg.region} ({area})\n  index: {index}\n  mosaic → {mosaic.with_suffix('.vrt')}")
    if check:
        print(f"  {'ok ' if index.exists() else 'will download'} {index}")
        print(f"  gdal: {'ok' if subprocess.run('which gdal_translate', shell=True, capture_output=True).returncode == 0 else 'MISSING'}")
        return
    import geopandas as gpd
    import pygris
    import requests
    from shapely.geometry import box
    from tqdm import tqdm

    if not index.exists():
        _download_index(p["index_url"], index)
    idx = gpd.read_file(index)
    if cfg.get("boundary", None):
        aoi = gpd.read_file(cfg.get("boundary")).to_crs(idx.crs)
        geom = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.geometry.unary_union
        overlap = idx[idx.intersects(geom)]              # the polygon itself, not its bbox
        print(f"  {len(overlap)} index tiles intersect the boundary ({len(aoi)} polygons)")
    else:
        towns = pygris.county_subdivisions(state=cfg.get("state"), year=p["towns_year"])
        town = towns[towns["NAME"] == cfg.get("city")].to_crs(idx.crs)
        if town.empty:
            raise SystemExit(f"town '{cfg.get('city')}' not found in county subdivisions")
        overlap = idx[idx.intersects(box(*town.total_bounds))]
        print(f"  {len(overlap)} index tiles intersect the town bbox")
    gb = len(overlap) * 0.15
    print(f"  ≈ {gb:.0f} GB of JP2 to download" + (" — run in tmux" if gb > 20 else ""))

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
    lst = vrt.with_suffix(".txt")                       # file list: avoids "argument list too long"
    lst.write_text("\n".join(jp2) + "\n")
    subprocess.run(["gdalbuildvrt", "-input_file_list", str(lst), str(vrt)], check=True)
    if p.get("build_tif", False):
        subprocess.run(["gdal_translate", str(vrt), str(mosaic), "-co", "COMPRESS=LZW",
                        "-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS"],
                       check=True)
        print(f"  mosaic written: {mosaic}")
    else:
        print(f"  mosaic VRT written: {vrt}  (tile reads it directly; set imagery.build_tif: true "
              "for a single BigTIFF)")
    print("→ next: bikelane tile")
