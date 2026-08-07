#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
mkdir -p results

"${PYTHON_BIN}" src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/baseline_xgb_classification_model.py \
  --is_NN false \
  --is_regression false \
  --classification_thresholds 1000 \
  --classification_target_method central_t \
  --reduction_timestamp_method central_t \
  --classification_evaluation_method majority \
  --evaluation_timestamp_method central_i \
  --invert_threshold_logic false \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-25 \
  --n_seconds 50 \
  --n_overlapping_seconds -10 \
  --average_signals channel \
  --apply_log true \
  --reduce_to_size 250 \
  --instance_window 1 \
  --join_higher_classes true \
  --balance_classes unbalanced \
  --random_state 42 \
  --run_name xgb-classification-baseline-1000m-iw1 \
  --mlflow_experiment_name DAS-XGBoost-classification-baseline \
  --mlflow_tracking_uri sqlite:///mlflow.db
