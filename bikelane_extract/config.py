"""Region configuration.

A config is a YAML file (see configs/). It may `extends:` another YAML, in
which case mappings are merged recursively and scalars in the child win.

Every stage reads its parameters through `cfg.get("stage.key")` and its
files through `cfg.path(...)`. Paths default to `<data_root>/<subdir>/<REGION>/`
but any entry in `paths:` overrides that, so an existing on-disk layout does
not have to be moved.

Values set to the string "TODO" are placeholders for region-specific
parameters that have not been re-derived yet; `cfg.get()` raises on them so
a stage cannot silently run on another region's numbers.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

TODO = "TODO"


class TodoParameter(ValueError):
    pass


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


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        data = _merge(_load_yaml(parent_path), data)
    return data


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
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser().resolve()
        return cls(_load_yaml(path), source=path)

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
            raise TodoParameter(
                f"'{dotted}' is TODO in {self.source.name if self.source else 'config'}: "
                f"it is region-dependent and must be derived for {self.region} "
                f"(see README §7 for the --sweep that sets it)"
            )
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
        w = Path(self.get(f"weights.{which}")).expanduser()
        if not w.is_absolute():
            root = self.source.parent.parent if self.source else Path.cwd()
            w = root / w
        return w

    def __repr__(self):
        return f"Config({self.region}, {self.source})"


def _check_no_todo(node: dict, prefix: str) -> None:
    for k, v in node.items():
        if v == TODO:
            raise TodoParameter(f"'{prefix}.{k}' is TODO — re-derive it for this region")
        if isinstance(v, dict):
            _check_no_todo(v, f"{prefix}.{k}")
