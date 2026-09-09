"""weights — fetch released model weights into weights/.

    bikelane weights -c configs/lexington.yaml [--tag v0.1.0]

Weights are GitHub Release assets, not committed to the repo (seg U-Net ~94 MB).
"""
from __future__ import annotations

import urllib.request

from .config import Config

RELEASE_REPO = "ssynn/bikelane-extract"      # TODO set once the repo exists


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane weights")
    ap.add_argument("--tag", default="latest")
    a = ap.parse_args(argv)
    for which in ("seg", "yolo"):
        dst = cfg.weights(which)
        if dst.exists():
            print(f"  ok  {dst}")
            continue
        if check:
            print(f"  MISSING {dst}")
            continue
        base = (f"https://github.com/{RELEASE_REPO}/releases/latest/download" if a.tag == "latest"
                else f"https://github.com/{RELEASE_REPO}/releases/download/{a.tag}")
        url = f"{base}/{dst.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"  → {dst} ({dst.stat().st_size/1e6:.0f} MB)")
