# bikelane-extract

Lane-level extraction of on-street bike facilities — dedicated bike lanes and
sharrows (shared-lane markings) — from high-resolution aerial orthophotos.

Given 15 cm/px orthophotos of a town, the pipeline produces a GeoJSON of
**bike-lane centerlines**, each tagged with its facility type (`BikeOnly` /
`Sharrow`), the number of pavement symbols that support it, and how much of
its length was directly observed versus filled in. It was developed on
MassDOT imagery of Lexington and Boston, MA, but nothing below is specific to
Massachusetts except the imagery download step.

The output is deliberately **not a routable network**. Bike facilities are
scattered; the road network already provides connectivity. What this adds is
*where exactly* each facility is (which lane, to ~2 m) and *what type* it is.

---

## Contents

1. [How the pipeline is organized](#1-how-the-pipeline-is-organised)
2. [Requirements and installation](#2-requirements-and-installation)
3. [Setting up a region config](#3-setting-up-a-region-config)
4. [Model weights](#4-model-weights)
5. [Running the pipeline, stage by stage](#5-running-the-pipeline-stage-by-stage)
6. [Output files](#6-output-files)
7. [Adapting to a new region](#7-adapting-to-a-new-region)
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
| 2 | `bikelane train` | Train the lane-marking segmentation model (U-Net / ResNet34) | hours | GPU strongly recommended |
| 3 | `bikelane predict` | Run the segmentation model on every tile → class masks | 10–30 min GPU, hours CPU | GPU recommended; CPU supported |
| 4 | `bikelane centerlines` | Class masks → lane centerlines (Voronoi boundary between markings) → cleaned polylines | 1–3 h | CPU, RAM |
| 5 | `bikelane signs` | Detect bike pavement symbols with YOLO; filter by confidence | 30 min + GPU, hours CPU | GPU recommended; CPU supported |
| 6 | `bikelane join` | Attach symbols to centerlines → bike-lane lines with a type | seconds | — |
| 7 | `bikelane gaps` | Close short gaps between bike-lane pieces; connect across intersections via nodes | seconds | `osm` output |
| – | `bikelane osm` | Fetch the OSM road network and junction nodes for the region (used by stage 7) | minutes | network |
| – | `bikelane weights` | Download released model weights | minutes | network |

**If you already have the released weights, you skip stages 1–2 entirely.**
The normal path for a new town is:

```
fetch → tile → predict → centerlines → signs → join → osm → gaps
```

Every command takes the same two things:

```
bikelane <stage> -c configs/<region>.yaml [step] [options]
bikelane <stage> -c configs/<region>.yaml --check [step]   # validate inputs, run nothing
```

`--check` prints every input file the stage needs and whether it exists.
Use it before every stage.

---

## 2. Requirements and installation

* Linux or macOS, Python ≥ 3.10. On Windows, everything except stage 1 should work natively. Stage 1 runs POSIX shell scripts, so use WSL for that stage.
* A GPU is recommended for stages 2, 3, and 5. CUDA is supported on Linux and Apple MPS on macOS. The device is detected automatically, and `--device cpu` forces CPU execution.

  * Stages 3 (`predict`) and 5 (`signs`) can run on CPU, although a full town may take several hours.
  * Stage 2 (`train`) is generally impractical on CPU. Use the released weights unless you intend to train a new model.
* Stage 4 needs about 3 GB RAM with the default chunk size. Use `--chunk 4` (about 1.5 GB) on a laptop or memory-constrained machine. `configs/_local_example.yaml` collects the laptop settings.
* GDAL command-line tools (`gdalbuildvrt`, `gdal_translate`) are required for stage 0.
* Allow about 50 GB free disk space per town for imagery, tiles, and predictions.

### Standard installation

On a normal local filesystem:

```bash
git clone https://github.com/<you>/bikelane-extract
cd bikelane-extract

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install PyTorch first. Choose the build that matches your platform and GPU environment using the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

Examples:

```bash
# Linux + NVIDIA GPU, example for CUDA 12.4
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Linux, CPU only
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# macOS, MPS support is included when available
pip install torch torchvision
```

Then install the package and all optional dependencies:

```bash
pip install -e ".[all]"
```

### Repositories on mounted or network filesystems

Some mounted or network filesystems do not support all filesystem operations used by Python `venv` or setuptools. Symptoms include errors such as:

```text
Error: [Errno 5] Input/output error: 'lib' -> '.venv/lib64'
Operation not permitted: '.venv/bin/python3'
Operation not permitted: 'bikelane_extract.egg-info/...'
```

In this case, the repository and data can remain on the mounted filesystem. Put only the Python virtual environment on a local filesystem:

```bash
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/bikelane
source ~/.venvs/bikelane/bin/activate

python -m pip install --upgrade pip
```

Install the appropriate PyTorch build as described above.

If editable installation also fails because setuptools cannot write `*.egg-info` into the repository, install from a temporary local copy:

```bash
# Run from the bikelane-extract repository root.

rm -rf /tmp/bikelane-install-src
cp -a . /tmp/bikelane-install-src

python -m pip install "/tmp/bikelane-install-src[all]"
```

Then point Python back to the working copy so edits to the repository take effect immediately:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

`PYTHONPATH` must be set again when opening a new shell unless it is added to your shell or development-environment configuration.

Verify that the installed CLI uses the working copy:

```bash
which bikelane

python - <<'PY'
import bikelane_extract
print("Using source:", bikelane_extract.__file__)
PY

bikelane --help
```

The reported source path should point to your working repository rather than `/tmp/bikelane-install-src` or `site-packages`.

### Optional dependency groups

| extra     | installs                              | needed by  |
| --------- | ------------------------------------- | ---------- |
| `imagery` | geopandas, pygris, rasterio, requests | stage 0    |
| `seg`     | segmentation-models-pytorch           | stages 1–3 |
| `signs`   | ultralytics                           | stage 5    |
| `osm`     | osmnx, geopandas                      | `osm`      |
| `all`     | all of the above                      |            |

Stages 4, 6, and 7 need only the core dependencies (numpy, scipy, shapely, networkx, opencv, scikit-image), which are installed with the package itself.

Check the installation:

```bash
bikelane --help
bikelane gaps -c configs/lexington.yaml --check scan
```

---

## 3. Setting up a region config

Configuration lives in `configs/`:

| file | role |
|---|---|
| `default.yaml` | every parameter that does **not** depend on the region — model, centerline extraction, cleaning tolerances, OSM settings. Region-dependent keys are set to `TODO` here. |
| `_template.yaml` | skeleton for a new region: copy it, fill in the header, derive the `TODO`s |
| `lexington.yaml`, `boston.yaml` | worked regions: `extends: default.yaml` plus their header, data paths, and (for Lexington) the derived values |

For a new town:

```bash
cp configs/_template.yaml configs/mytown.yaml
```

and fill in the header:

```yaml
extends: default.yaml

region: MYTOWN             # upper-case tag; appears in every output filename (lower-cased)
city: My Town              # town name as it appears in Census county subdivisions (stage 0)
state: MA
crs: "EPSG:32619"          # projected CRS in metres — the UTM zone of your region
data_root: /data/bikelane  # where everything is written by default
```

Only four parameter groups are region-dependent and left as `TODO` — the
sign acceptance threshold, the join radius, and the gap-scan tolerances.
Section 7 walks through deriving them with the `--sweep` commands.
Everything else comes from `default.yaml` and does not normally change.

### Where files go

By default every stage reads and writes under `data_root`:

```
<data_root>/
  imagery/<REGION>/Merged_<REGION>.tif     stage 0
  imagery/<REGION>/tiles/*.jpg             stage 0
  imagery/<REGION>/Tile_Mappings.csv       stage 0
  predictions/<REGION>/*_pred.png          stage 3
  centerlines/<REGION>/chunks/*.npz        stage 4 (cache)
  centerlines/<REGION>/centerlines_<region>_{voronoi,final}.geojson
  signs/<REGION>/<region>_bikesigns*.{csv,geojson}
  bikelanes/<REGION>/<region>_bikelanes*.geojson, gap_*.geojson, intersection_links.geojson
  osm/<REGION>/<region>_osm_{nodes,edges}.geojson
  logs/<REGION>/
  opensatmap/, train/                      stages 1–2 (not per region)
```

If your data already lives somewhere else, override any directory in
`paths:` and nothing has to move:

```yaml
paths:
  tiles_dir:         /existing/Aerial_Images/LEXINGTON/LEXINGTON_Tiled
  tile_mappings_csv: /existing/Aerial_Images/LEXINGTON/Tile_Mappings.csv
  predictions_dir:   /existing/predictions/LEXINGTON
```

To clear all inherited overrides in a child config, write `paths: null`.

### Inheritance and `TODO`

A config can `extends:` another one; mappings merge recursively and the
child's scalars win, `null` deletes an inherited key. Region-dependent
parameters are the literal string `TODO` in `default.yaml`. **A stage
refuses to run while a parameter it uses is `TODO`**, and names it. A
region config becomes complete once it overrides all of them.

### Parameter sections

| YAML key | Used by | What it controls |
|----------|---------|------------------|
| `imagery` | stage 0 | tile overlap (0.25), empty-tile threshold (0.50), index/town years |
| `prepare` | stage 1 | OpenSatMap zip prefix, annotation file, image/clip sizes |
| `train` | stages 2, 3 | encoder, classes, mask dilation, batch, epochs, learning rate |
| `predict` | stage 3 | batch size |
| `centerlines` | stage 4 | chunk size, `voronoi` (mask → centerline), `graph`, `clean` |
| `signs` | stage 5 | inference confidence, **per-class acceptance threshold**, isolation radius |
| `join` | stage 6 | **match radius**, cleaning tolerances, opposite-direction handling |
| `osm` | `osm` | network type (`drive`), buffer, junction degree |
| `gaps` | stage 7 | **gap ceiling / angle / lateral tolerances** for `scan` and `intersections`, intersection test, OSM node snap distance |
| `weights` | stages 3, 5 | paths to the two weight files |

Bold items are the region-dependent ones.

---

## 4. Model weights

Two trained models are required and are distributed as **GitHub Release
assets**, not in the repository:

| file | model | used by |
|------|-------|---------|
| `weights/seg_unet_r34.pt` | U-Net (ResNet34 encoder), 4 classes: background / lane line / curb / virtual line. Trained on OpenSatMap level-20 (0.15 m/px). ~94 MB | stage 3 |
| `weights/yolo26m_bikesign.pt` | YOLO, 3 classes: `BikeOnly`, `Sharrow`, `OnlyBikeBus`. ~45 MB | stage 5 |

```bash
bikelane weights -c configs/lexington.yaml            # downloads both into weights/
bikelane weights -c configs/lexington.yaml --tag v0.1.0
```

If you trained your own segmentation model (stage 2), copy its `best.pt` to
the `weights.seg` path in the config. The YOLO detector is distributed as
weights only; its training data is not part of this repository.

---

## 5. Running the pipeline, stage by stage

All examples use `-c configs/lexington.yaml`; substitute your config.
Long stages are marked **(long)** — run them inside `tmux` or `screen`.

### Stage 0 — imagery: `fetch` and `tile`

**Skip this stage if you already have 1024 px tiles named `tile_px<X>_py<Y>.jpg`
and a `Tile_Mappings.csv`** (columns `image_name, CRS_X, CRS_Y, Pixel_X, Pixel_Y`).
Point `paths.tiles_dir` and `paths.tile_mappings_csv` at them.

`fetch` needs the MassDOT orthophoto index shapefile (`COQ2025INDEX_POLY.shp`,
from MassGIS) at `paths.massdot_index_shp`. It finds the tiles that intersect
the town's bounding box, downloads the zips, extracts the `.jp2` files and
builds a single LZW-compressed BigTIFF mosaic.

```bash
bikelane fetch -c configs/lexington.yaml --check
bikelane fetch -c configs/lexington.yaml            # (long) ~44 zips for Lexington
```

`tile` cuts the mosaic into overlapping 1024 px JPGs and writes the mapping CSV.

```bash
bikelane tile -c configs/lexington.yaml --check
bikelane tile -c configs/lexington.yaml [--workers 16]
```

Expected: several thousand tiles; tiles that are ≥ 50 % nodata are dropped
(the counter is printed). **Do not lower `empty_threshold` to 0** — that
silently drops every tile touching the town boundary.

Outputs: `Merged_<REGION>.tif`, `tiles/*.jpg`, `Tile_Mappings.csv`.

### Stage 1 — `prepare` (only if training your own model)

Uses the OpenSatMap dataset (level 20, 0.15 m/px; CC BY-NC-SA 4.0 — check
the licence fits your use) and the OpenSatMap `tools-release` scripts.

```bash
bikelane prepare -c configs/lexington.yaml --check
bikelane prepare -c configs/lexington.yaml download  # (long) ~50 GB from Hugging Face + git clone of the tools
bikelane prepare -c configs/lexington.yaml unzip     # concatenate parts and extract
bikelane prepare -c configs/lexington.yaml bake      # (long) masks + split + 1024 px cut
bikelane prepare -c configs/lexington.yaml images    # image tiles matching the GT tiles
bikelane prepare -c configs/lexington.yaml verify    # image : GT tile counts must be 1:1
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
bikelane train -c configs/lexington.yaml --check
bikelane train -c configs/lexington.yaml                    # (long) 30 epochs, batch 8
bikelane train -c configs/lexington.yaml --resume           # continue from last.pt
bikelane train -c configs/lexington.yaml --epochs 5         # smoke test
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
bikelane predict -c configs/lexington.yaml --check
bikelane predict -c configs/lexington.yaml [--device cuda:0]
bikelane predict -c configs/lexington.yaml --summary    # stats on existing predictions only
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
bikelane centerlines -c configs/lexington.yaml --check extract
bikelane centerlines -c configs/lexington.yaml extract --limit 3   # smoke test: 3 chunks
bikelane centerlines -c configs/lexington.yaml extract             # (long)
bikelane centerlines -c configs/lexington.yaml graph
bikelane centerlines -c configs/lexington.yaml clean
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
bikelane signs -c configs/lexington.yaml --check detect
bikelane signs -c configs/lexington.yaml detect              # (long) YOLO on every tile
bikelane signs -c configs/lexington.yaml filter --sweep      # threshold table
bikelane signs -c configs/lexington.yaml filter
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
bikelane join -c configs/lexington.yaml --check match
bikelane join -c configs/lexington.yaml match --sweep     # radius table
bikelane join -c configs/lexington.yaml match [--radius 2.0]
bikelane join -c configs/lexington.yaml clean
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
bikelane osm -c configs/lexington.yaml --check
bikelane osm -c configs/lexington.yaml
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
bikelane gaps -c configs/lexington.yaml scan --sweep   # candidate count vs gap ceiling
bikelane gaps -c configs/lexington.yaml scan
#   → open <region>_gap_join.geojson and _gap_crossing.geojson in QGIS
bikelane gaps -c configs/lexington.yaml join
bikelane gaps -c configs/lexington.yaml intersections
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

## 7. Adapting to a new region

1. **Config.** Copy `configs/_template.yaml` → `configs/<region>.yaml`. Set
   `region`, `city`, `state`, `crs` (the UTM zone in metres), `data_root`,
   and any `paths:` overrides. The region-dependent parameters are already
   `TODO`.
2. **Imagery** (stage 0) — MassDOT-specific. For other imagery, produce the
   tiles and `Tile_Mappings.csv` yourself (see the format under Stage 0) and
   point the config at them. Tiles must be 1024 px, filename
   `tile_px<X>_py<Y>.jpg`, with `<X>,<Y>` the pixel offset of the tile's
   top-left corner in a common mosaic.
3. **Predict, centerlines** — no region parameters, run as is. Check the
   `clean` numbers as described in Stage 4.
4. **`signs.conf_keep`** — run `signs filter --sweep`, pick the threshold at
   the isolated-share minimum. Dense urban areas may also need a smaller
   `signs.neighbor_r_m`.
5. **`join.radius_m`** — run `join match --sweep`, pick the largest radius
   before lines-per-sign jumps.
6. **`gaps.scan.gap_max_m` / `ang` / `lat_max_m`** — run `gaps scan --sweep`,
   set the ceiling where the count flattens, then inspect every candidate in
   QGIS before `gaps join`.

Fill the derived values into the config (replacing `TODO`) so the run is
reproducible.

---

## 8. Troubleshooting

**`venv` or `pip install -e` fails with `Input/output error` or `Operation not permitted`**: the repository is likely on a mounted or network filesystem that does not support one of the filesystem operations required by Python `venv` or setuptools.

The repository itself does not need to move. Create the virtual environment on a local filesystem such as `$HOME`:

```bash
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/bikelane
source ~/.venvs/bikelane/bin/activate
```

If `pip install -e ".[all]"` fails while creating `bikelane_extract.egg-info`, install from a temporary local copy and use `PYTHONPATH` to load the working tree:

```bash
rm -rf /tmp/bikelane-install-src
cp -a . /tmp/bikelane-install-src
python -m pip install "/tmp/bikelane-install-src[all]"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Verify the source being imported:

```bash
python - <<'PY'
import bikelane_extract
print(bikelane_extract.__file__)
PY
```

The virtual environment does not need to live inside the repository. You can still run `pip install -e ".[all]"` from the repository root.

**`config error: '<key>' is TODO …`** — the parameter must be derived for
this region (section 7) and written into the config.

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
  cli.py            entry point: `bikelane <stage> -c <config>`
  config.py         YAML loading, `extends:`, path resolution, TODO guard
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
configs/
  default.yaml      all region-independent parameters; region-dependent ones TODO
  _template.yaml    copy for a new region
  _local_example.yaml  laptop settings (smaller chunks/batches, MPS)
  lexington.yaml    worked example with derived values
  boston.yaml       header + paths only; TODOs still to derive
pyproject.toml
```

Dependencies run in one direction only: a stage imports from `config`,
`io`, `geom`, `facility` and from lower-numbered stages, never from a
higher-numbered one.
