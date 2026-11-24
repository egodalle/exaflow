#!/usr/bin/env bash
# Recreate three real trading problems, measure the damage, then show each fix.
# Results land in demo_results/*.json. Takes about 30 seconds.
set -euo pipefail
python3 -m demos.demo_quality     # problem 4: bad data gets into signals
python3 -m demos.demo_pit         # problem 2: research data isn't point-in-time
python3 -m demos.demo_costs       # problem 6: research ignores trading costs
