"""bikelane <stage> --config configs/<region>.yaml [stage options]

Stages run one at a time; each reads the previous stage's files from the
paths in the config. Nothing runs end to end on purpose — every stage has
a check step (a sweep, a plot, a QGIS look) between it and the next.

  0  fetch         MassDOT index + town name → mosaic → 1024 px tiles + Tile_Mappings.csv
  1  prepare       OpenSatMap zips → baked masks → image/GT tile pairs
  2  train         seg model (U-Net R34) → weights/seg_unet_r34.pt
  3  predict       tiles + seg weights → predictions/<REGION>/*_pred.png
  4  centerlines   predictions → voronoi centerlines → cleaned centerlines
  5  signs         tiles + YOLO weights → bike-sign points (raw / filtered / dropped)
  6  join          centerlines × signs → bike lanes with type + evidence counts
  7  gaps          scan → join → intersections (via OSM junction nodes)
     osm           fetch OSM drive network and junction nodes for the tile extent
     weights       download released model weights into weights/
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .config import Config, TodoParameter

# subcommand → (module, function). Modules are imported lazily so that a
# missing optional dependency (torch, ultralytics, osmnx…) only breaks the
# stage that needs it.
STAGES = {
    "fetch":       ("bikelane_extract.s00_imagery.fetch", "run"),
    "tile":        ("bikelane_extract.s00_imagery.tile", "run"),
    "prepare":     ("bikelane_extract.s01_prepare.prepare", "run"),
    "train":       ("bikelane_extract.s02_train.train", "run"),
    "predict":     ("bikelane_extract.s03_predict.predict", "run"),
    "centerlines": ("bikelane_extract.s04_centerline.run", "run"),
    "signs":       ("bikelane_extract.s05_signs.run", "run"),
    "join":        ("bikelane_extract.s06_join.run", "run"),
    "gaps":        ("bikelane_extract.s07_gaps.run", "run"),
    "osm":         ("bikelane_extract.osm", "run"),
    "weights":     ("bikelane_extract.weights", "run"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bikelane", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", "-c", required=True, help="configs/<region>.yaml")
    ap.add_argument("--check", action="store_true",
                    help="load config and validate inputs for this stage, then exit")
    args, rest = ap.parse_known_args(argv)

    try:
        cfg = Config.load(args.config)
    except TodoParameter as e:
        sys.exit(f"config error: {e}")

    mod_name, fn_name = STAGES[args.stage]
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("bikelane_extract"):
            sys.exit(f"stage '{args.stage}' is not ported yet ({mod_name})")
        sys.exit(f"stage '{args.stage}' needs an optional dependency: {e.name}\n"
                 f"  pip install 'bikelane-extract[{_extra_for(args.stage)}]'")
    fn = getattr(mod, fn_name)
    try:
        return fn(cfg, rest, check=args.check)
    except TodoParameter as e:
        sys.exit(f"config error: {e}")


def _extra_for(stage: str) -> str:
    return {"fetch": "imagery", "tile": "imagery", "prepare": "seg", "train": "seg",
            "predict": "seg", "signs": "signs", "osm": "osm"}.get(stage, "all")


if __name__ == "__main__":
    main()
