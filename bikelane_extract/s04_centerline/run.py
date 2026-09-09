"""bikelane centerlines -c <cfg> {extract|graph|clean|all} [--fresh] [--limit N] [--chunk N]

  extract   predictions → per-chunk polylines (.npz cache, resumable)   08c run
  graph     chunks → dedup → graph → centerlines_<region>_voronoi.geojson 08c build_graph
  clean     hooks + overlaps → centerlines_<region>_final.geojson         08d
  all       the three in sequence

`extract` is the long one (tmux). Re-running it skips finished chunks;
`--fresh` clears the cache; `--limit 3` is a smoke test.
"""
from __future__ import annotations

import argparse

from ..config import Config
from . import clean, extract, graph


def run(cfg: Config, argv=None, check: bool = False):
    ap = argparse.ArgumentParser(prog="bikelane centerlines", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "graph", "clean", "all"])
    ap.add_argument("--fresh", action="store_true", help="(extract) clear the chunk cache")
    ap.add_argument("--limit", type=int, default=0, help="(extract) only the first N chunks")
    ap.add_argument("--chunk", type=int, default=None, help="(extract) tiles per chunk side")
    a = ap.parse_args(argv)
    if a.step in ("extract", "all"):
        extract.run(cfg, check=check, fresh=a.fresh, limit=a.limit, chunk_tiles=a.chunk)
    if a.step in ("graph", "all"):
        graph.run(cfg, check=check)
    if a.step in ("clean", "all"):
        clean.run(cfg, check=check)
