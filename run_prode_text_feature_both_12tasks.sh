#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
bash run_prode_text_feature_12tasks.sh
bash run_prode_text_feature_iic_12tasks.sh
