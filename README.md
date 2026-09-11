# bikelane-extract

Lane-level extraction of on-street bike facilities — dedicated bike lanes and
sharrows (shared-lane markings) — from high-resolution aerial orthophotos.

Given 15 cm/px orthophotos of a town, the pipeline produces a GeoJSON of
**bike-lane centerlines**, each tagged with its facility type (`BikeOnly` /
`Sharrow`), the number of pavement symbols that support it, and how much of
its length was directly observed versus filled in. It was developed on
MassDOT imagery of Lexington and Boston, MA, but nothing below is specific to
Massachusetts except the imagery download step.

---

## Contents

1. [How the pipeline is organised](#1-how-the-pipeline-is-organised)
2. [Requirements and installation](#2-requirements-and-installation)
3. [Configuration](#3-configuration)
4. [Model weights](#4-model-weights)
5. [Running the pipeline, stage by stage](#5-running-the-pipeline-stage-by-stage)
6. [Output files](#6-output-files)
7. [Running a new town, and when to re-check parameters](#7-running-a-new-town-and-when-to-re-check-parameters)
8. [Troubleshooting](#8-troubleshooting)
9. [Repository layout](#9-repository-layout)

---

## 1. How the pipeline is organised

The work is split into eight numbered stages. **Each stage is a separate
command** that reads the files written by the previous stage and writes its
own. Nothing runs end to end on purpose: between most stages there is a
number to look at, a sweep to run, or a file to open in QGIS before you
continue.

| # | Command | What it does | Time | Needs |
|---|---------|--------------|------|-------|
| 0 | `bikelane fetch`, `bikelane tile` | Download MassDOT orthophotos for a town, build one mosaic, cut it into 1024 px tiles | ~1 h | network, GDAL |
| 1 | `bikelane prepare` | Unpack the OpenSatMap training set and bake lane-marking masks | hours | OpenSatMap download |
| 2 | `bikelane train` | Train the lane-marking segmentation model (U-Net / ResNet34) | hours | GPU |
| 3 | `bikelane predict` | Run the segmentation model on every tile → class masks | 10–30 min | GPU |
| 4 | `bikelane centerlines` | Class masks → lane centerlines (Voronoi boundary between markings) → cleaned polylines | 1–3 h | CPU, RAM |
| 5 | `bikelane signs` | Detect bike pavement symbols with YOLO; filter by confidence | 30 min + | GPU |
| 6 | `bikelane join` | Attach symbols to centerlines → bike-lane lines with a type | seconds | — |
| 7 | `bikelane gaps` | Close short gaps between bike-lane pieces; connect across intersections via nodes | seconds | `osm` output |
| – | `bikelane osm` | Fetch the OSM road network and junction nodes for the region (used by stage 7) | minutes | network |
| – | `bikelane weights` | Download released model weights | minutes | network |

**If you already have the released weights, you skip stages 1–2 entirely.**
The normal path for a new town is:

```
fetch → tile → predict → centerlines → signs → join → osm → gaps
```

Every command has the same shape and reads the project file
`bikelane.yaml` in the current directory (section 3):

```
bikelane <stage> [step] [options]
bikelane <stage> --check [step]      # validate inputs for this stage, run nothing
```

`--check` prints every input file the stage needs and whether it exists.
Use it before every stage.

---

## 2. Requirements and installation

* Linux or macOS, Python ≥ 3.10. (Windows: everything except stage 1, which
  runs POSIX shell scripts — use WSL for that stage.)
* A GPU for stages 2, 3 and 5: CUDA on Linux, Apple MPS on macOS (detected
  automatically; `--device cpu` to force CPU). Stage 3 and 5 run on CPU in
  a few hours per town; stage 2 (training) is impractical without CUDA —
  use the released weights.
* Stage 4 needs ~3 GB RAM at the default chunk size; use `--chunk 4`
  (~1.5 GB) on a laptop, and lower `predict.batch` / `signs.batch` to 4 in a
  `bikelane.yaml`.
* GDAL command-line tools (`gdalbuildvrt`, `gdal_translate`) for stage 0.
* ~50 GB free disk per town (imagery + tiles + predictions).
* RAM: stage 4 peaks at ~3 GB with the default chunk size.

```bash
git clone https://github.com/Synn-Arch/bikelane-extract
cd bikelane-extract
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch first — see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Linux + CUDA
pip install torch torchvision                                                    # macOS (MPS included)

# 2. the package with everything
pip install -e ".[all]"
```

Optional-dependency groups, if you only need part of the pipeline:

| extra | installs | needed by |
|-------|----------|-----------|
| `imagery` | geopandas, pygris, rasterio, requests | stage 0 |
| `seg` | segmentation-models-pytorch | stages 1–3 |
| `signs` | ultralytics | stage 5 |
| `osm` | osmnx, geopandas | `osm` |
| `all` | all of the above | |

Stages 4, 6 and 7 need only the core dependencies (numpy, scipy, shapely,
networkx, opencv, scikit-image), which are installed with the package itself.

Check the install:

```bash
bikelane --help
bikelane config              # after setting city/state in bikelane.yaml
```

---

## 3. Configuration

Two files, and you edit only the first:

**`bikelane.yaml`** — the project file. Copy the template once, write the
town, and every command reads it from the current directory. It is
git-ignored: it describes *your* run on *your* machine, not the package.

```bash
cp bikelane.example.yaml bikelane.yaml
```
```yaml
city: Boston
state: MA
# data_root: ./data      # default; everything is created under it — set an absolute
#                        # path if the data should live on another disk
```

`city` is the town name as in the Census county subdivisions (that is what
the imagery download uses); it becomes the region tag (`Fall River` →
`FALL_RIVER`) in every output path. `state` is the two-letter code. The
projected CRS defaults to the town's UTM zone (`crs:` to override). A
relative `data_root` is relative to the project file, so the layout is the
same whichever directory you run from.

**`bikelane_extract/default.yaml`** — every parameter, shipped inside the
package with the values fixed during development (Lexington, MA, 2026).
Not edited. To see what is in effect for the current project:

```bash
bikelane config
```

### Changing something

Any key from `default.yaml` can be overridden in `bikelane.yaml`, and so
can any output directory — which is how you use data that already exists
somewhere else instead of re-creating it:

```yaml
city: Boston
state: MA
paths:
  tiles_dir:         /existing/Aerial_Images/BOSTON/BOSTON_Tiled
  tile_mappings_csv: /existing/Aerial_Images/BOSTON/Tile_Mappings.csv
join:
  radius_m: 1.5
centerlines:
  chunk_tiles: 4           # laptop
```

For a one-off, override on the command line without touching the file:

```bash
bikelane join match --set join.radius_m=1.5
bikelane predict --city "Fall River" --state MA      # another town, same project
bikelane predict -c /path/to/other/bikelane.yaml     # another project file
```

Every stage run writes the configuration it actually used to
`<data_root>/logs/<REGION>/config_<stage>_<timestamp>.yaml`, so a result
can always be traced back to its parameters.

### Where files go

Everything is under `data_root`:

```
<data_root>/
  weights/                                  bikelane weights
  imagery/<REGION>/Merged_<REGION>.tif      stage 0
  imagery/<REGION>/tiles/*.jpg              stage 0
  imagery/<REGION>/Tile_Mappings.csv        stage 0
  predictions/<REGION>/*_pred.png           stage 3
  centerlines/<REGION>/chunks/*.npz         stage 4 (cache)
  centerlines/<REGION>/centerlines_<region>_{voronoi,final}.geojson
  signs/<REGION>/<region>_bikesigns*.{csv,geojson}
  bikelanes/<REGION>/<region>_bikelanes*.geojson, gap_*.geojson, intersection_links.geojson
  osm/<REGION>/<region>_osm_{nodes,edges}.geojson
  logs/<REGION>/                            run logs + config snapshots
  opensatmap/, train/                       stages 1–2 (not per town)
```

Any of these directories can be overridden under `paths:` in
`bikelane.yaml`, so an existing layout does not have to be moved.

### Parameter sections

| YAML key | Used by | What it controls |
|----------|---------|------------------|
| `imagery` | stage 0 | tile overlap (0.25), empty-tile threshold (0.50), index/town years |
| `prepare` | stage 1 | OpenSatMap source, zip prefix, annotation file, image/clip sizes |
| `train` | stages 2, 3 | encoder, classes, mask dilation, batch, epochs, learning rate |
| `predict` | stage 3 | batch size |
| `centerlines` | stage 4 | chunk size, `voronoi` (mask → centerline), `graph`, `clean` |
| `signs` | stage 5 | inference confidence, per-class acceptance threshold, isolation radius |
| `join` | stage 6 | match radius, cleaning tolerances, opposite-direction handling |
| `osm` | `osm` | network type (`drive`), buffer, junction degree |
| `gaps` | stage 7 | gap ceiling / angle / lateral tolerances for `scan` and `intersections`, intersection test, OSM node snap distance |
| `weights` | stages 3, 5 | paths to the two weight files (relative → `<data_root>/weights/`) |

Four groups are marked *region-sensitive* in `default.yaml` — the sign
acceptance threshold, the join radius, and the gap-scan tolerances. They
were chosen on a suburban town and are reasonable defaults elsewhere, but
section 7 shows how to check them with the built-in sweeps if the setting
is very different (dense downtown, other imagery).

---

## 4. Model weights

Two trained models are required and are distributed as **GitHub Release
assets**, not in the repository:

| file | model | used by |
|------|-------|---------|
| `weights/seg_unet_r34.pt` | U-Net (ResNet34 encoder), 4 classes: background / lane line / curb / virtual line. Trained on OpenSatMap level-20 (0.15 m/px). ~94 MB | stage 3 |
| `weights/yolo26m_bikesign.pt` | YOLO, 3 classes: `BikeOnly`, `Sharrow`, `OnlyBikeBus`. ~45 MB | stage 5 |

```bash
bikelane weights            # → <data_root>/weights/, shared by all towns
bikelane weights --tag v0.1.0
```

If you trained your own segmentation model (stage 2), copy its `best.pt` to
`<data_root>/weights/seg_unet_r34.pt` (or set `weights.seg` in `bikelane.yaml`). The YOLO detector is distributed as
weights only; its training data is not part of this repository.

---

## 5. Running the pipeline, stage by stage

All examples assume `bikelane.yaml` names the town (section 3).
Long stages are marked **(long)** — run them inside `tmux` or `screen`.

### Stage 0 — imagery: `fetch` and `tile`

**Skip this stage if you already have 1024 px tiles named `tile_px<X>_py<Y>.jpg`
and a `Tile_Mappings.csv`** (columns `image_name, CRS_X, CRS_Y, Pixel_X, Pixel_Y`).
Point `paths.tiles_dir` and `paths.tile_mappings_csv` at them in `bikelane.yaml`.

`fetch` downloads the MassGIS 2025 orthophoto index shapefile on first use,
finds the tiles that intersect the town's bounding box, downloads their
zips, extracts the `.jp2` files and builds a single LZW-compressed BigTIFF
mosaic. Nothing has to be placed by hand.

```bash
bikelane fetch --check
bikelane fetch            # (long) ~44 zips for Lexington
```

`tile` cuts the mosaic into overlapping 1024 px JPGs and writes the mapping CSV.

```bash
bikelane tile --check
bikelane tile [--workers 16]
```

Expected: several thousand tiles; tiles that are ≥ 50 % nodata are dropped
(the counter is printed). **Do not lower `empty_threshold` to 0** — that
silently drops every tile touching the town boundary.

Outputs: `Merged_<REGION>.tif`, `tiles/*.jpg`, `Tile_Mappings.csv`.

### Stage 1 — `prepare` (only if training your own model)

Uses the OpenSatMap dataset (level 20, 0.15 m/px; CC BY-NC-SA 4.0 — check
the licence fits your use) and the OpenSatMap `tools-release` scripts.

```bash
bikelane prepare --check
bikelane prepare download  # (long) ~50 GB from Hugging Face + git clone of the tools
bikelane prepare unzip     # concatenate parts and extract
bikelane prepare bake      # (long) masks + split + 1024 px cut
bikelane prepare images    # image tiles matching the GT tiles
bikelane prepare verify    # image : GT tile counts must be 1:1
```

`download` fetches `20picstrainvaltest.zip.001…012` and `annotrainval20.json`
from the `z-hb/OpenSatMap` dataset on Hugging Face (resumable; needs
`huggingface_hub`) into `paths.opensatmap_dir`, and clones the OpenSatMap code
repository to obtain `tools-release`. If you already have both, skip it and
point `paths.opensatmap_dir` / `paths.opensatmap_tools_dir` at them.

`bake` copies `tools-release` into the work directory and patches it
(mmcv/mmengine → OpenCV; `shutil.copy` → `copyfile` for CIFS mounts) before
running it. Logs go to `<prep_work_dir>/bake.log` and `split_tile.log`.

Outputs under `<prep_work_dir>/picuse20save/final/`:
`allpic-cut/{train,val}/*.png` (images) and
`pic20gtsplit-cut/category/{train,val}/*-GT.png` (masks, values 0/1/2/3/255).

### Stage 2 — `train` (only if training your own model)

```bash
bikelane train --check
bikelane train                    # (long) 30 epochs, batch 8
bikelane train --resume           # continue from last.pt
bikelane train --epochs 5         # smoke test
```

Per epoch it prints loss and IoU for lane / curb / virtual. Checkpoints go
to `paths.ckpt_dir` as `last.pt` and `best.pt` (best mean IoU over the three
line classes). The reference model reached lane IoU ≈ 0.41 — low because
the lines are 1–3 px wide, not because the model misses them. When done:

```bash
cp <ckpt_dir>/best.pt weights/seg_unet_r34.pt
```

### Stage 3 — `predict`

```bash
bikelane predict --check
bikelane predict [--device cuda:0]
bikelane predict --summary    # stats on existing predictions only
```

Writes one `<tile stem>_pred.png` per tile (uint8, 0 = background, 1 = lane
line, 2 = curb, 3 = virtual line). Tiles that already have a prediction are
skipped, so an interrupted run can simply be restarted.

What to check: `--summary` prints how many tiles contain lane and curb pixels.
For a suburban town expect lane in ~40–60 % of tiles; a much lower number
means the tiles or the weights are wrong. Open a few `_pred.png` over their
JPGs in QGIS (same stem) — red should sit on painted lines.

### Stage 4 — `centerlines`

Three steps; `all` runs them in sequence.

```bash
bikelane centerlines --check extract
bikelane centerlines extract --limit 3   # smoke test: 3 chunks
bikelane centerlines extract             # (long)
bikelane centerlines graph
bikelane centerlines clean
```

`extract` merges 6×6 tiles onto one canvas at a time, extracts the
centerline once per canvas (so tile overlaps do not create duplicate lines)
and caches each chunk as `chunks/<REGION>_<gx>_<gy>.npz`. Restarting resumes
from the cache; `--fresh` clears it. If it dies with an out-of-memory
message, re-run with `--chunk 4`. Progress and errors are logged to
`logs/<REGION>/extract_<REGION>.log`.

`graph` loads all chunks, removes chunk-overlap duplicates and pieces
shorter than 3 m, builds a graph (nodes snapped to 1.5 m), drops components
under 20 m, and writes `centerlines_<region>_voronoi.geojson`.

`clean` cuts hook-shaped tails and removes lines that run on top of a
longer one, writing `centerlines_<region>_final.geojson`. It prints line
count / total km / component count after each step. **Read the numbers:**
total length should drop only modestly (Lexington: 786 → 650 km). If it
falls a lot, tighten `centerlines.clean.dup_lat_m` / `dup_frac`. If it
*rises*, stitching is on — it should not be (`stitch: false`); see
section 8.

What to check in QGIS: `centerlines_<region>_final.geojson` over the
orthophoto. Lines should sit in the middle of lanes, between painted
markings. There will be many breaks at intersections; that is expected and
handled in stage 7.

### Stage 5 — `signs`

```bash
bikelane signs --check detect
bikelane signs detect              # (long) YOLO on every tile
bikelane signs filter --sweep      # threshold table
bikelane signs filter
```

`detect` runs the YOLO model at a *low* confidence (0.25) and saves every
detection to `<region>_bikesigns.csv`. This is the expensive part and runs
only once; if the CSV exists it is skipped (`--fresh` to redo).

`filter --sweep` prints, for a range of thresholds, how many detections
survive and what share of them are **isolated** (no other detection within
30 m). Real bike symbols come in runs along a street, so isolated ones are
mostly false positives. Pick the threshold where the isolated share is
lowest — where it starts rising again you are cutting real symbols.
Lexington: minimum near 0.55, chosen 0.60. Put the value in
`signs.conf_keep` per class.

`filter` applies `conf_keep`, marks isolated accepted signs (`isolated=1`,
kept, not removed), and writes `_filtered.csv/.geojson` and `_dropped.geojson`.

What to check in QGIS: `_filtered.geojson` should trace streets; `_dropped`
should be scattered. `OnlyBikeBus` is dropped by default (`1.01`) — in
Lexington all five detections were false positives.

### Stage 6 — `join`

```bash
bikelane join --check match
bikelane join match --sweep     # radius table
bikelane join match [--radius 2.0]
bikelane join clean
```

`match` attaches every accepted sign to all centerlines within
`join.radius_m` of its centre (a lane half-width is 1.75 m). A centerline
with at least one sign becomes a bike lane; its type is the majority of its
signs' classes.

`match --sweep` prints, per radius, the match rate, the average number of
lines per sign and the total km. Use the largest radius at which lines per
sign stays near 1.0–1.1; when it jumps (Lexington: 1.10 → 1.34 between 2 m
and 3 m) the radius is reaching into the next lane. Unmatched signs are
saved with `nearest_m`, the distance to the nearest centerline:

| `nearest_m` | meaning | action |
|---|---|---|
| just above R | radius slightly small | consider a larger R |
| 5–20 m | a centerline exists but in the wrong lane | stage 4 accuracy, not a join problem |
| > 20 m | no centerline there | marking not detected in stage 3, or the sign is a false positive |

`clean` merges bike-lane lines that overlap (summing their sign counts) and
cuts hooks. Lines running side by side in *opposite* directions are kept —
that pattern is a two-way facility with symbols on both sides
(`join.clean.merge_opposite: true` to merge them anyway). Short lines are
reported only; `drop_short: true` removes them.

Outputs: `<region>_bikelanes.geojson`, `<region>_unmatched_signs.geojson`,
`<region>_bikelanes_clean.geojson`.

### `osm` — junction nodes for stage 7

```bash
bikelane osm --check
bikelane osm
```

Downloads the OSM `drive` network for the tile extent (+200 m), projects it
to the region CRS, and writes `<region>_osm_nodes.geojson` with
`is_junction = 1` for nodes of degree ≥ 3, plus `<region>_osm_edges.geojson`.
Only the junction nodes are used downstream. Deliberately does **not** fetch
OSM's bike tags, so they remain an independent source for validating the
result.

Stage 7 works without this file (it falls back to gap midpoints), but the
intersection nodes are then less accurate.

### Stage 7 — `gaps`

Three steps, in order.

```bash
bikelane gaps scan --sweep   # candidate count vs gap ceiling
bikelane gaps scan
#   → open <region>_gap_join.geojson and _gap_crossing.geojson in QGIS
bikelane gaps join
bikelane gaps intersections
```

`scan` finds pairs of bike-lane endpoints that face each other within
`gaps.scan.gap_max_m` and could be joined by a straight segment. **It joins
nothing.** Each candidate is classified: if a car-lane centerline crosses
the gap at ≥ 30°, it is an *intersection crossing* and goes to
`_gap_crossing.geojson`; otherwise to `_gap_join.geojson`. Both are
LineStrings you can lay over the orthophoto. There should be tens, not
thousands, of candidates — few enough to look at every one. A candidate that
leaves the pavement or cuts between two parallel lines means the tolerances
are too loose: reduce `lat_max_m` or `ang`.

`scan --sweep` shows how the candidate count grows with the gap ceiling.
The right ceiling is where growth flattens (Lexington: 25 m).

`join` applies `_gap_join.geojson`: each pair is connected with a straight
segment and the two lines become one. It is single-pass by design. Every
output line records how much of it is observation versus inference:

| property | meaning |
|---|---|
| `n_joins` | number of gaps filled inside this line |
| `gap_total_m` | filled (inferred) length |
| `obs_len_m` | original (observed) length |
| `obs_ratio` | observed / total — colour by this in QGIS |

`intersections` re-scans the joined lines with looser tolerances (turns
enter intersections at an angle) and handles the two kinds differently:
straight gaps are joined as before; **crossings get a node** — the nearest
OSM junction within 25 m, else the gap midpoint — and each side is linked to
that node. Lines are *not* merged across an intersection, because they are
different roads and a straight line through would leave no node for
turning movements.

Outputs: `<region>_bikelanes_joined.geojson` (after `join`),
**`<region>_bikelanes_final.geojson`** and `<region>_intersection_links.geojson`
(after `intersections`). The final line count and connected-component count
are printed.

---

## 6. Output files

All vector outputs are GeoJSON in the region CRS (`crs` in the config; UTM,
metres). Existing files are never overwritten: the previous version is
renamed to `<name>.<unix time>.geojson` and the new one written.

**Final products**

`<region>_bikelanes_final.geojson` — LineString features:

| property | type | meaning |
|---|---|---|
| `type` | str | `BikeOnly` or `Sharrow` (majority of supporting symbols) |
| `n_signs` | int | number of symbols supporting the line |
| `classes` | JSON str | per-class symbol counts, e.g. `{"Sharrow": 3, "BikeOnly": 1}` |
| `conf_max` | float | highest detector confidence among its symbols |
| `length_m` | float | |
| `n_joins`, `gap_total_m`, `obs_len_m`, `obs_ratio` | | observation vs inference, see stage 7 |
| `source` | str | `bikelane` |

`<region>_intersection_links.geojson` — 2-vertex LineStrings from a
bike-lane endpoint to an intersection node: `node_kind` (`osm_junction` /
`midpoint`), `node_snap_m`, `gap_m`, `line` (index of the bike-lane line).

**Intermediate files worth keeping**

| file | stage | use |
|---|---|---|
| `Tile_Mappings.csv` | 0 | the only link between tile pixels and coordinates — every later stage needs it |
| `centerlines_<region>_final.geojson` | 4 | all lane centerlines; background for `gaps`, and the thing to validate against a road inventory |
| `<region>_bikesigns.csv` | 5 | raw detections; lets you re-threshold without re-running YOLO |
| `<region>_unmatched_signs.geojson` | 6 | why signs did not become bike lanes (`nearest_m`) |
| `<region>_gap_crossing.geojson` | 7 | intersection gaps, for manual review |

---

## 7. Running a new town, and when to re-check parameters

Change two lines in `bikelane.yaml` and run the stages in order:

```yaml
city: Fall River
state: MA
```

```bash
bikelane weights
bikelane fetch
bikelane tile
bikelane predict
bikelane centerlines all
bikelane signs detect
bikelane signs filter
bikelane join match
bikelane join clean
bikelane osm
bikelane gaps scan          # look at the candidates in QGIS
bikelane gaps join
bikelane gaps intersections
```

For imagery other than MassDOT, skip `fetch`/`tile`, produce 1024 px tiles
named `tile_px<X>_py<Y>.jpg` plus a `Tile_Mappings.csv` (format under Stage
0) yourself, and point `paths.tiles_dir` / `paths.tile_mappings_csv` at
them in `bikelane.yaml`.

The defaults were chosen on a suburban town. Three of them depend on how
dense the symbols are and how the streets are shaped, and each stage has a
`--sweep` that shows whether the default still fits. Run them when the
setting is clearly different; otherwise the defaults are fine.

| parameter | default | check with | change it when |
|---|---|---|---|
| `signs.conf_keep` | 0.60 | `signs filter --sweep` | the isolated-share minimum is clearly not at 0.60 |
| `join.radius_m` | 2.0 m | `join match --sweep` | lines-per-sign is already above ~1.1 at 2.0 m (narrow lanes) |
| `gaps.scan.gap_max_m / ang / lat_max_m` | 25 m / 12° / 1 m | `gaps scan --sweep`, then the candidates in QGIS | candidates leave the pavement or cut between parallel lines |

Apply a change in `bikelane.yaml` or with `--set key=value`; the config
snapshot in `logs/` records what was used.

---

## 8. Troubleshooting

**`ImportError: … libstdc++.so.6: version GLIBCXX_… not found`, or a
traceback that imports from `/usr/lib/python3/dist-packages` instead of your
venv** — `PYTHONPATH` or `LD_LIBRARY_PATH` is pulling in system/conda
packages ahead of the venv (common on machines with QGIS or Anaconda). Run
`unset PYTHONPATH LD_LIBRARY_PATH` and check
`python -c "import shapely; print(shapely.__file__)"` points into the venv;
add the `unset` line to the end of `<venv>/bin/activate` to make it stick.

**`stage '<x>' needs an optional dependency: <module>`** — install the extra
listed in section 2, e.g. `pip install -e ".[signs]"`.

**`no tile … matches Tile_Mappings.csv` / `inconsistent mosaic origin`** —
tile filenames and CSV rows disagree. Every `image_name` in the CSV must
equal the JPG filename, and `Pixel_X/Y` must equal the `px…/py…` in the
name. This check is strict on purpose: a wrong origin produces plausible
lines in the wrong place.

**`predict` finds lane in < 20 % of tiles** — wrong weights, tiles in a
different colour space, or tiles not 1024 px. Look at one `_pred.png` over
its JPG.

**`centerlines extract` — out of memory** — re-run with `--chunk 4` (or 3).
Finished chunks are kept.

**`centerlines extract` — a chunk logs an error** — it is skipped and listed
at the end; re-running processes only the missing ones. If it fails
repeatedly, the log line names the exception.

**`centerlines clean` — total length rose** — stitching is on. Set
`centerlines.clean.stitch: false`. Endpoint-tangent stitching draws chords
across curves and was abandoned; gaps are closed in stage 7 instead.

**`join match` — many unmatched signs at 5–20 m** — centerlines are in the
wrong lane. Check stage 4 output over the imagery; the radius is not the
problem.

**`gaps join` — `N candidates did not match`** — the bike-lane file changed
after `scan` (e.g. you re-ran `join clean`). Re-run `gaps scan`.

**`gaps intersections` — `node source {'midpoint': …}` only** — no OSM nodes
found: run `bikelane osm` first, or the nodes file is at a different path
than `paths.osm_dir` expects.

**A stage says `backed up existing file`** — normal; the previous output was
renamed with a timestamp rather than overwritten.

---

## 9. Repository layout

```
bikelane_extract/
  cli.py            entry point: `bikelane <stage>`; reads ./bikelane.yaml
  config.py         default.yaml + bikelane.yaml + --set, path resolution, run snapshots
  io.py             GeoJSON read/write with backup, tile↔UTM mapping, sign CSV
  geom.py           polyline geometry shared by stages 4, 6, 7
  facility.py       facility-type vocabulary (BikeOnly / Sharrow / OnlyBikeBus)
  s00_imagery/      fetch.py, tile.py
  s01_prepare/      prepare.py            OpenSatMap → masks → tile pairs
  s02_train/        train.py, common.py   model/loader code also used by predict
  s03_predict/      predict.py
  s04_centerline/   voronoi.py  (mask → centerline)   skeleton.py  (pixels → polylines)
                    extract.py  (chunked run)         graph.py     clean.py     run.py
  s05_signs/        detect.py, filter.py, run.py
  s06_join/         spatial_join.py, clean.py, run.py
  s07_gaps/         scan.py, join.py, intersections.py, common.py, run.py
  osm.py            OSM drive network + junction nodes
  weights.py        download release weights
  default.yaml      ALL parameters (not edited)
bikelane.example.yaml   template for the project file
bikelane.yaml           your copy (git-ignored): city, state, and any overrides — the only file you edit
pyproject.toml
```

Dependencies run in one direction only: a stage imports from `config`,
`io`, `geom`, `facility` and from lower-numbered stages, never from a
higher-numbered one.
