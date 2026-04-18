#!/bin/bash
# Run the qframe agentic pipeline from the repo root.
# Usage:
#   ./run_pipeline.sh --domain momentum              # 1 iteration, momentum domain
#   ./run_pipeline.sh --domain volatility --n 3      # 3 iterations, one domain
#   ./run_pipeline.sh --domain all --n 5             # 5 iterations × 5 domains = 25 total
#   ./run_pipeline.sh --help
cd "$(dirname "$0")"
PYTHONPATH=src /Users/franciscosimao/opt/anaconda3/envs/qframe/bin/python3 -m qframe.pipeline.run "$@"
