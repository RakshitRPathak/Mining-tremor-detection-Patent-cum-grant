#!/usr/bin/env bash
# Runs the full simulation pipeline in the required order.
set -e
cd "$(dirname "$0")/src"
echo "[1/3] Binary model, detector comparison, threshold sweep, multi-node consensus"
python tremor_simulation.py > /dev/null
echo "[2/3] Streaming stimulus matrix, latency, duplicates, firmware fault injection"
python streaming_and_firmware_simulation.py > /dev/null
echo "[3/3] 4-class model, annotated multi-node run, figures"
python extended_analysis_simulation.py > /dev/null
echo "Done. Results in results/, figures in figures/."
