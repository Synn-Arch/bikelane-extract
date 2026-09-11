"""bikelane <stage> [step] [options]

Reads ./bikelane.yaml (city, state, data_root, optional overrides) and runs one
stage. Stages run one at a time; each reads the previous stage's files from
<data_root>/<subdir>/<REGION>/. Nothing runs end to end on purpose — every
stage has a check step (a sweep, a plot, a QGIS look) between it and the next.

All parameters live in the package's default.yaml; override them in
bikelane.yaml, or for one run with --set key=value. `bikelane config` prints
what is in effect.
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
     config        print the parameters and paths in effect for a town
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .config import Config, TodoParameter, parse_value

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
    ap.add_argument("stage", choices=list(STAGES) + ["config"])
    ap.add_argument("--config", "-c", default=None, help="project file (default ./bikelane.yaml)")
    ap.add_argument("--city", default=None, help="override the town in the project file for this run")
    ap.add_argument("--state", default=None)
    ap.add_argument("--data-root", default=None, help="override data_root for this run")
    ap.add_argument("--crs", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a parameter for this run, e.g. --set join.radius_m=1.5")
    ap.add_argument("--check", action="store_true",
                    help="load config and validate inputs for this stage, then exit")
    args, rest = ap.parse_known_args(argv)

    overrides = {}
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"--set expects KEY=VALUE, got {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    try:
        cfg = Config.from_args(args.city, args.state, args.data_root, args.crs, overrides, args.config)
    except TodoParameter as e:
        sys.exit(f"config error: {e}")

    if args.stage == "config":
        _config_cmd(cfg, rest)
        return 0

    mod_name, fn_name = STAGES[args.stage]
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("bikelane_extract"):
            sys.exit(f"stage '{args.stage}' is not ported yet ({mod_name})")
        sys.exit(f"stage '{args.stage}' needs an optional dependency: {e.name}\n"
                 f"  pip install 'bikelane-extract[{_extra_for(args.stage)}]'")
    fn = getattr(mod, fn_name)
    if not args.check:
        cfg.snapshot(args.stage)
    try:
        fn(cfg, rest, check=args.check)     # stage return values are for tests, not the shell
    except TodoParameter as e:
        sys.exit(f"config error: {e}")
    return 0


def _config_cmd(cfg, rest):
    """bikelane config --city ... : print the effective configuration."""
    import yaml
    area = (f"boundary {cfg.get('boundary')}" if cfg.get("boundary", None)
            else f"{cfg.get('city', '?')}, {cfg.get('state', '')}")
    print(f"region: {cfg.region} ({area})  crs: {cfg.crs}  data_root: {cfg.data_root}")
    print("paths:")
    for k in ("tiles_dir", "tile_mappings_csv", "predictions_dir", "centerlines_final",
              "signs_filtered", "bikelanes_final", "osm_nodes"):
        print(f"  {k:20s} {cfg.path(k)}")
    print(f"project file: {cfg.source}\nparameters in effect:")
    body = {k: v for k, v in cfg._d.items() if k not in ("city", "state", "region", "crs", "data_root", "paths")}
    print("  " + yaml.safe_dump(body, sort_keys=False, allow_unicode=True).replace("\n", "\n  "))


def _extra_for(stage: str) -> str:
    return {"fetch": "imagery", "tile": "imagery", "prepare": "seg", "train": "seg",
            "predict": "seg", "signs": "signs", "osm": "osm"}.get(stage, "all")


if __name__ == "__main__":
    main()
