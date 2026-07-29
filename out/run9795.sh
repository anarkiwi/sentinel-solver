#!/bin/bash
# Solve the hardest landscape (atlas rank 1 of 10000) on whichever host runs this.
cd /scratch/anarkiwi/cbm/sentinel-solver || exit 1
export PYTHONUNBUFFERED=1 PYTHONPATH=/scratch/anarkiwi/cbm/sentinel-solver
exec timeout 3300 python3 -m sentinel.phase_player "${1:-9795}"
