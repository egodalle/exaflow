#!/usr/bin/env bash
# Full demo: generate -> scaling benchmark -> chaos run -> queries -> exabyte model
# Usage: ./run_demo.sh [small|medium|large]
set -euo pipefail
SCALE=${1:-medium}
case $SCALE in
  small)  SYMS=200;  DAYS=5;  TICKS=20000 ;;   # ~20M ticks, seconds
  medium) SYMS=500;  DAYS=15; TICKS=50000 ;;   # ~375M ticks, minutes
  large)  SYMS=2000; DAYS=60; TICKS=50000 ;;   # ~6B ticks, needs ~60 GB disk
esac
BUCKETS=16
CORES=$(python3 -c "import os; print(os.cpu_count())")
rm -rf bench && mkdir -p bench

echo "== 1. generate synthetic tick data ($SCALE)"
python3 -m exaflow.generate --root data/ticks --symbols $SYMS --days $DAYS \
        --buckets $BUCKETS --ticks $TICKS

echo "== 2. scaling benchmark: 1 -> $CORES workers"
W=1
while [ $W -le $CORES ]; do
  python3 -m exaflow.pipeline --src data/ticks --dst data/bars --workers $W --fresh \
          --bench-out bench/pipeline_w$W.json
  W=$((W*2))
done
[ -f bench/pipeline_w$CORES.json ] || python3 -m exaflow.pipeline --src data/ticks \
          --dst data/bars --workers $CORES --fresh --bench-out bench/pipeline_w$CORES.json

echo "== 3. chaos run: 20% of tasks crash on first attempt"
python3 -m exaflow.pipeline --src data/ticks --dst data/bars_chaos --fresh --fail-rate 0.2 \
        --bench-out bench/chaos.json
python3 compare_outputs.py data/bars data/bars_chaos

echo "== 4. resume: re-running is a no-op (everything already committed)"
python3 -m exaflow.pipeline --src data/ticks --dst data/bars

echo "== 5. research queries + partition pruning"
python3 -m exaflow.query --ticks data/ticks --bars data/bars --buckets $BUCKETS \
        --bench-out bench/query.json

echo "== 6. exabyte scale model"
python3 -m exaflow.scale_model --bench-dir bench
