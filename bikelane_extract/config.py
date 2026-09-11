"""Configuration.

Two files:
  bikelane_extract/default.yaml   every parameter, shipped with the package; not edited
  ./bikelane.yaml                 the project file: city, state, data_root, and any
                                  overrides (paths to existing data, parameter changes)

`bikelane <stage>` reads ./bikelane.yaml from the current directory (or the
file given with -c). --city/--state/--data-root/--set on the command line
override it for one run. The effective configuration of every run is
snapshotted to <data_root>/logs/<REGION>/ for reproducibility.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

TODO = "TODO"


class TodoParameter(ValueError):
    """A value is the placeholder string TODO (only possible in user overrides)."""


def _merge(base: dict, over: dict) -> dict:
    """Recursive merge; child scalars win. A child value of `null` deletes the
    key (e.g. `paths: null` to drop all inherited path overrides)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


PACKAGE_DEFAULT = Path(__file__).with_name("default.yaml")
PROJECT_FILE = "bikelane.yaml"


def _resolve_parent(parent: str, child: Path) -> Path:
    """`extends: default` (or default.yaml) → the package default; else relative to the child."""
    if parent in ("default", "default.yaml") and not (child.parent / parent).exists():
        return PACKAGE_DEFAULT
    return (child.parent / parent).resolve()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _merge(_load_yaml(_resolve_parent(parent, path)), data)
    return data


def region_slug(city: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", city).strip("_").upper()


# subdir under data_root for each logical location
_DEFAULT_SUBDIR = {
    "imagery_dir":      "imagery",
    "tiles_dir":        "imagery/{REGION}/tiles",
    "tile_mappings_csv": "imagery/{REGION}/Tile_Mappings.csv",
    "predictions_dir":  "predictions/{REGION}",
    "centerlines_dir":  "centerlines/{REGION}",
    "chunks_dir":       "centerlines/{REGION}/chunks",
    "signs_dir":        "signs/{REGION}",
    "bikelanes_dir":    "bikelanes/{REGION}",
    "osm_dir":          "osm/{REGION}",
    "weights_dir":      "weights",
    "logs_dir":         "logs/{REGION}",
    # training data (region-independent)
    "opensatmap_dir":   "opensatmap",              # raw OpenSatMap download (zips, anno json)
    "opensatmap_tools_dir": "opensatmap/tools-release",   # OpenSatMap tools repo checkout
    "prep_work_dir":    "train/prep_work",         # baked masks + tiles
    "ckpt_dir":         "train/ckpts",
    "massdot_index_shp": "imagery/massdot_index/COQ2025INDEX_POLY.shp",
    "mosaic_tif":       "imagery/{REGION}/Merged_{REGION}.tif",
}

# canonical file names, relative to the directory key given
_FILES = {
    "centerlines_voronoi": ("centerlines_dir", "centerlines_{region}_voronoi.geojson"),
    "centerlines_final":   ("centerlines_dir", "centerlines_{region}_final.geojson"),
    "signs_raw_csv":       ("signs_dir", "{region}_bikesigns.csv"),
    "signs_filtered":      ("signs_dir", "{region}_bikesigns_filtered.geojson"),
    "signs_dropped":       ("signs_dir", "{region}_bikesigns_dropped.geojson"),
    "bikelanes":           ("bikelanes_dir", "{region}_bikelanes.geojson"),
    "bikelanes_unmatched": ("bikelanes_dir", "{region}_unmatched_signs.geojson"),
    "bikelanes_clean":     ("bikelanes_dir", "{region}_bikelanes_clean.geojson"),
    "gap_join":            ("bikelanes_dir", "{region}_gap_join.geojson"),
    "gap_crossing":        ("bikelanes_dir", "{region}_gap_crossing.geojson"),
    "bikelanes_joined":    ("bikelanes_dir", "{region}_bikelanes_joined.geojson"),
    "bikelanes_final":     ("bikelanes_dir", "{region}_bikelanes_final.geojson"),
    "intersection_links":  ("bikelanes_dir", "{region}_intersection_links.geojson"),
    "network_edges":       ("bikelanes_dir", "{region}_network_edges.geojson"),
    "network_nodes":       ("bikelanes_dir", "{region}_network_nodes.geojson"),
    "osm_nodes":           ("osm_dir", "{region}_osm_nodes.geojson"),
    "osm_edges":           ("osm_dir", "{region}_osm_edges.geojson"),
}


class Config:
    def __init__(self, data: dict, source: Path | None = None):
        self._d = data
        self.source = source
        self.region: str = str(data["region"]).upper()
        self.region_lower = self.region.lower()
        self.crs: str = data.get("crs", "EPSG:32619")
        self.resolution_m: float = float(data.get("resolution_m", 0.15))
        self.tile_px: int = int(data.get("tile_px", 1024))
        self.data_root = Path(data.get("data_root", "./data")).expanduser()

    # ── loading ──
    @classmethod
    def from_args(cls, city: str | None, state: str | None, data_root: str | Path = "./data",
                  crs: str | None = None, overrides: dict | None = None,
                  overrides_file: str | Path | None = None) -> "Config":
        data = _load_yaml(PACKAGE_DEFAULT)
        source = PACKAGE_DEFAULT
        if overrides_file is None and Path(PROJECT_FILE).exists():
            overrides_file = PROJECT_FILE
        if overrides_file:
            of = Path(overrides_file).expanduser().resolve()
            proj = _load_yaml(of)
            # relative paths in the project file are relative to the file, not the cwd
            if "data_root" in proj and not Path(proj["data_root"]).expanduser().is_absolute():
                proj["data_root"] = str(of.parent / proj["data_root"])
            data = _merge(data, proj)
            source = of
        if city:
            data["city"] = city
            data["region"] = region_slug(city)
        if state:
            data["state"] = state.upper()
        if "boundary" in data:                       # an area given by a polygon file
            b = Path(data["boundary"]).expanduser()
            if not b.is_absolute() and overrides_file:
                b = Path(overrides_file).expanduser().resolve().parent / b
            data["boundary"] = str(b)
            data.setdefault("region", region_slug(data.get("name") or b.stem))
        elif "city" in data and "region" not in data:
            data["region"] = region_slug(data["city"])
        if "region" not in data:
            raise SystemExit(f"no area set: `cp bikelane.example.yaml {PROJECT_FILE}` and write "
                             f"`city:`/`state:` (one town) or `boundary:` (any polygon file) in it")
        if data_root:
            data["data_root"] = str(Path(data_root).expanduser().resolve())
        data.setdefault("data_root", str(Path("./data").resolve()))
        if crs:
            data["crs"] = crs
        elif "crs" not in data:
            data["crs"] = _cached_crs(Path(data["data_root"]), data["region"],
                                      data.get("city"), data.get("state", ""),
                                      data.get("imagery", {}).get("towns_year"),
                                      data.get("boundary"))
        for k, v in (overrides or {}).items():
            node = data
            parts = k.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = v
        return cls(data, source=source)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """A standalone overrides yaml that also sets city/region (defaults merged in)."""
        path = Path(path).expanduser().resolve()
        return cls(_merge(_load_yaml(PACKAGE_DEFAULT), _load_yaml(path)), source=path)

    def snapshot(self, stage: str) -> Path:
        """Write the effective configuration next to the logs, once per stage run."""
        import time
        d = self.path("logs_dir", mkdir=True)
        p = d / f"config_{stage}_{time.strftime('%Y%m%d-%H%M%S')}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(self._d, f, sort_keys=False, allow_unicode=True)
        return p

    # ── parameters ──
    def get(self, dotted: str, default: Any = ...) -> Any:
        """cfg.get("gaps.scan.gap_max_m"). Raises on missing (unless default) or TODO."""
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not ...:
                    return default
                raise KeyError(f"config has no '{dotted}' ({self.source})")
            node = node[part]
        if node == TODO:
            raise TodoParameter(f"'{dotted}' is TODO in {self.source} — give it a value")
        if isinstance(node, dict):
            _check_no_todo(node, dotted)
        return node

    def section(self, name: str) -> dict:
        return dict(self.get(name, {}))

    # ── paths ──
    def path(self, key: str, *parts: str, mkdir: bool = False) -> Path:
        """Resolve a directory key (e.g. 'tiles_dir') or file key (e.g. 'bikelanes').

        Directory keys check `paths:` overrides first, then data_root defaults.
        File keys resolve to their canonical name inside the appropriate directory.
        """
        over = self._d.get("paths") or {}
        if key in _FILES:
            dkey, fname = _FILES[key]
            if key in over:                         # explicit file override
                p = Path(over[key]).expanduser()
            else:
                p = self.path(dkey) / fname.format(region=self.region_lower)
        elif key in over:
            p = Path(over[key]).expanduser()
        elif key in _DEFAULT_SUBDIR:
            p = self.data_root / _DEFAULT_SUBDIR[key].format(REGION=self.region)
        else:
            raise KeyError(f"unknown path key '{key}'")
        p = p.joinpath(*parts) if parts else p
        if mkdir:
            (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
        return p

    def weights(self, which: str) -> Path:
        """Relative weight paths live under data_root (weights/…)."""
        w = Path(self.get(f"weights.{which}")).expanduser()
        return w if w.is_absolute() else self.data_root / w

    def __repr__(self):
        return f"Config({self.region}, {self.source})"


def _check_no_todo(node: dict, prefix: str) -> None:
    for k, v in node.items():
        if v == TODO:
            raise TodoParameter(f"'{prefix}.{k}' is TODO — re-derive it for this region")
        if isinstance(v, dict):
            _check_no_todo(v, f"{prefix}.{k}")


def _cached_crs(data_root: Path, region: str, city, state: str, year=None, boundary=None) -> str:
    """UTM zone of the area, derived once and remembered in <data_root>/logs/<REGION>/crs.txt."""
    f = data_root / "logs" / region / "crs.txt"
    if f.exists():
        return f.read_text().strip()
    crs = (_utm_for_boundary(boundary) if boundary else _utm_for(city, state, year)) or "EPSG:32619"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(crs + "\n")
    return crs


def _utm_of_lonlat(x: float, y: float) -> str:
    zone = int((x + 180) // 6) + 1
    return f"EPSG:{(32600 if y >= 0 else 32700) + zone}"


def _utm_for_boundary(path) -> str | None:
    try:
        import geopandas as gpd
        g = gpd.read_file(path)
        c = g.geometry.to_crs(3857).union_all().centroid if hasattr(g.geometry, "union_all") \
            else g.geometry.to_crs(3857).unary_union.centroid
        c = gpd.GeoSeries([c], crs=3857).to_crs(4326).iloc[0]
        return _utm_of_lonlat(c.x, c.y)
    except Exception:
        return None


def _utm_for(city, state: str, year=None) -> str | None:
    if not city:
        return None
    try:
        import contextlib
        import io
        import warnings
        import pygris
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            towns = pygris.county_subdivisions(state=state, year=year, cache=True)
        t = towns[towns["NAME"].str.lower() == city.lower()]
        if t.empty:
            return None
        c = t.geometry.to_crs(3857).centroid.to_crs(4326).iloc[0]   # planar centroid, then lon/lat
        return _utm_of_lonlat(c.x, c.y)
    except Exception:
        return None


def parse_value(v: str):
    """'2.0' → 2.0, 'true' → True, 'TODO' stays a string."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
