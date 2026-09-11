#!/usr/bin/env bash
# run_all.sh — run the whole pipeline end to end for the area in ./bikelane.yaml.
# No manual checks in between; gap candidates are joined as found.
# `tile --cleanup` deletes the downloaded zips/JP2/VRT once tiling is complete
# (hundreds of GB for a region); remove the flag below to keep them.
# Safe to re-run: every stage skips work that is already done.
#
#   tmux new -s bikelane
#   bash run_all.sh                       # everything
#   bash run_all.sh predict               # start from this stage (skip fetch/tile)
#   WORKERS=16 bash run_all.sh            # centerline extract parallelism
#
# Log: <data_root>/logs/run_all_<REGION>.log  (also printed to the terminal)

set -u -o pipefail
cd "$(dirname "$0")"

# the venv must be active (or on PATH) — otherwise every stage silently fails
if ! command -v bikelane >/dev/null; then
  for v in "$HOME/.venvs/bikelane/bin/activate" ".venv/bin/activate"; do
    [[ -f "$v" ]] && { source "$v"; unset PYTHONPATH LD_LIBRARY_PATH; break; }
  done
fi
command -v bikelane >/dev/null || { echo "bikelane not found — activate the venv first"; exit 127; }

WORKERS="${WORKERS:-}"
START="${1:-fetch}"

STAGES=(
  "weights"
  "fetch"
  "tile --cleanup"
  "predict"
  "centerlines extract${WORKERS:+ --workers $WORKERS}"
  "centerlines graph"
  "centerlines clean"
  "signs detect"
  "signs filter"
  "join match"
  "join clean"
  "osm"
  "gaps scan"
  "gaps join"
  "gaps intersections"
  "gaps network"
)

# resolve data_root / region for the log file
eval "$(python - <<'EOF'
from bikelane_extract.config import Config
c = Config.from_args(None, None, None)
print(f'DATA_ROOT="{c.data_root}"; REGION="{c.region}"')
EOF
)"
mkdir -p "$DATA_ROOT/logs"
LOG="$DATA_ROOT/logs/run_all_${REGION}.log"
echo "=== run_all $REGION  $(date)  start=$START ===" | tee -a "$LOG"

started=0
t0=$(date +%s)
for stage in "${STAGES[@]}"; do
  name="${stage%% *}"
  [[ "$started" == 0 && "$stage" != "$START"* ]] && { echo "skip  $stage" | tee -a "$LOG"; continue; }
  started=1
  echo "----- bikelane $stage  ($(date +%H:%M))" | tee -a "$LOG"
  ts=$(date +%s)
  bikelane $stage 2>&1 | tee -a "$LOG"
  rc=$?
  echo "----- done $stage  rc=$rc  $(( ($(date +%s) - ts) / 60 )) min" | tee -a "$LOG"
  if [[ "$rc" != 0 ]]; then
    echo "FAILED at: bikelane $stage  — fix and re-run:  bash run_all.sh $name" | tee -a "$LOG"
    exit "$rc"
  fi
done
echo "=== all stages done  $(( ($(date +%s) - t0) / 3600 )) h  ===" | tee -a "$LOG"
echo "final: $DATA_ROOT/bikelanes/$REGION/$(echo "$REGION" | tr 'A-Z' 'a-z')_network_edges.geojson"
