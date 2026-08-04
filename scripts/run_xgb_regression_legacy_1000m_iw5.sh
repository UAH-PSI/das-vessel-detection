#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
mkdir -p results

"${PYTHON_BIN}" src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/baseline_xgb_regression_model.py \
  --is_NN false \
  --is_regression true \
  --regression_threshold 1000 \
  --regression_target_method legacy \
  --reduction_timestamp_method legacy \
  --regression_evaluation_method legacy \
  --evaluation_timestamp_method legacy \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-25 \
  --n_seconds 50 \
  --n_overlapping_seconds -10 \
  --average_signals channel \
  --apply_log true \
  --reduce_to_size 250 \
  --instance_window 5 \
  --random_state 42 \
  --run_name xgb-regression-legacy-1000m-iw5 \
  --mlflow_experiment_name DAS-XGBoost-regression-jstars-legacy \
  --mlflow_tracking_uri sqlite:///mlflow.db
