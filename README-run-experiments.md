# Running and analyzing model experiments

This guide explains how to launch, monitor, inspect, compare, and preserve experiments produced by `src/model_experiment_hdf5.py`. It is organized as an operational reference with a short first-run walkthrough.

The primary examples use the best baseline XGBoost configurations identified in the available experiment work:

- regression: 1,000 m target range, 50-second feature windows, five-instance evaluation windows, and channel averaging;
- classification: 1,000 m class threshold, 50-second feature windows, five-instance evaluation windows, and channel averaging.

These are examples of the execution workflow, not universal guarantees that the same configuration is optimal for another dataset, target definition, or date range.

For model implementation details, see `README-develop-models.md`. For a guided model-development exercise, see `README-tutorial-develop-models.md`.

## 1. Quick start

Run commands from the repository root.

### 1.1 Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/model_experiment_hdf5.py --help
```

The maintained scripts expect the dataset at:

```text
data/dataset_sensor_range_1440_1690_0.h5
```

If your dataset is elsewhere, either change `--h5_path` in the command or make a copy of the relevant launch script and edit that copy.

### 1.2 Run the best regression example

The direct command is:

```bash
python src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/baseline_xgb_regression_model.py \
  --is_NN false \
  --is_regression true \
  --regression_threshold 1000 \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-25 \
  --n_seconds 50 \
  --n_overlapping_seconds -10 \
  --instance_window 5 \
  --average_signals channel \
  --apply_log true \
  --reduce_to_size 250 \
  --use_mid_target true \
  --random_state 42 \
  --run_name xgboost-regression-all-folds-1000-best \
  --mlflow_experiment_name DAS-XGBoost-regression-jstars \
  --mlflow_tracking_uri sqlite:///mlflow.db
```

The equivalent maintained launcher is:

```bash
bash scripts/run_xgb_regress_baseline_all_folds-best-1000.sh
```

This experiment removes targets above 1,000 m before training and evaluation. Its error values must therefore be compared only with experiments using the same target population. A lower MAE than a 5,000 m experiment does not, by itself, prove that the model is better.

### 1.3 Run the best classification example

```bash
python src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/baseline_xgb_classification_model.py \
  --is_NN false \
  --is_regression false \
  --classification_thresholds 1000 \
  --invert_threshold_logic false \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-25 \
  --n_seconds 50 \
  --n_overlapping_seconds -10 \
  --average_signals channel \
  --apply_log true \
  --reduce_to_size 250 \
  --instance_window 5 \
  --use_mid_target true \
  --join_higher_classes true \
  --balance_classes unbalanced \
  --random_state 42 \
  --run_name xgboost-classification-all-folds-best \
  --mlflow_experiment_name DAS-XGBoost-classification-jstars \
  --mlflow_tracking_uri sqlite:///mlflow.db
```

Or use:

```bash
bash scripts/run_xgb_classif_baseline_all_folds-best.sh
```

With `--invert_threshold_logic false`, class 1 means distance greater than 1,000 m and class 0 means distance at most 1,000 m. Preserve this interpretation when reading class-specific precision, recall, and F1.

### 1.4 Confirm completion

A successful run prints and logs:

- the local Joblib path;
- the CSV path for a date-range run;
- the execution log path;
- the MLflow experiment, run name, run ID, and execution ID;
- instructions for opening MLflow.

Do not judge completion only by the presence of a CSV or `mlruns/`. Single-day runs intentionally produce no CSV, and the default SQLite tracking URI does not use MLflow's legacy `mlruns/` file-store layout.

## 2. Understand the experiment lifecycle

### 2.1 One invocation can contain many fitted models

With only `--test_date_start`, the runner performs one fold: that day is the test set and all other available days form the training pool.

With both start and end dates, every date in the inclusive interval becomes a separate held-out test fold. The runner creates and trains a fresh model for each fold. It then aggregates all daily results.

The serialized model artifact from a date-range run is the model fitted during the final fold. It is not a model refitted on the complete dataset.

### 2.2 Use a smoke test before a full run

While changing code or moving to a new machine, first remove `--test_date_end` and use a single day:

```bash
python src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/baseline_xgb_classification_model.py \
  --is_regression false \
  --is_NN false \
  --classification_thresholds 1000 \
  --average_signals channel \
  --n_seconds 50 \
  --n_overlapping_seconds -10 \
  --instance_window 5 \
  --test_date_start 2023-06-16 \
  --run_name xgb-classification-smoke
```

This checks imports, data access, feature shapes, model fitting, prediction, and Joblib persistence. It normally does not generate CSV or the complete global MLflow metric set. After it succeeds, rerun with the full date range.

### 2.3 Keep development, selection, and final evaluation distinct

A robust workflow has three phases:

1. **Development:** single-day smoke tests catch technical failures.
2. **Model selection:** compare a predefined parameter grid using the same folds and selection metric.
3. **Final evaluation:** freeze the chosen configuration and evaluate it once under the stated final protocol.

Repeatedly choosing configurations from the same reported final folds can produce optimistic estimates. Record which results were used for selection.

## 3. Choose the task and model family

The task switch and model-family switch are independent:

| Experiment                    | `--is_regression` | `--is_NN` | Model example                                            |
|-------------------------------|------------------:|----------:|----------------------------------------------------------|
| XGBoost regression            |            `true` |   `false` | `models/baseline_xgb_regression_model.py`                |
| XGBoost classification        |           `false` |   `false` | `models/baseline_xgb_classification_model.py`            |
| Neural-network regression     |            `true` |    `true` | Create a model using `README-tutorial-develop-models.md` |
| Neural-network classification |           `false` |    `true` | Create a model using `README-tutorial-develop-models.md` |

`--is_NN false` accepts any compatible model file whose `load_model()` returns an sklearn-style estimator. `--is_NN true` expects a PyTorch loader that returns the model, optimizer, criterion, and optionally a scheduler.

The non-NN path reserves 20% of each fold's available training population for validation. If an estimator does not accept `eval_set`, that reserved portion is currently not used for fitting.

The public repository includes the two best baseline launchers:

```text
scripts/run_xgb_regress_baseline_all_folds-best-1000.sh
scripts/run_xgb_classif_baseline_all_folds-best.sh
```

They reproduce the quick-start commands with the best XGBoost settings identified in the experiment sequence.

## 4. Command-line controls by purpose

Use `python src/model_experiment_hdf5.py --help` for the authoritative defaults of the checked-out revision.

### 4.1 Required inputs

| Option            | Purpose                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `--h5_path`       | HDF5 dataset containing `X`, `y`, and `datetimes`                        |
| `--model_file`    | Python file containing the model's `load_model` function                 |
| `--is_regression` | Select continuous regression (`true`) or classification (`false`)        |
| `--is_NN`         | Select the PyTorch (`true`) or general estimator (`false`) training path |

### 4.2 Dates and fold scope

| Option                         | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| `--test_date_start YYYY-MM-DD` | Single test day or first date of a range                       |
| `--test_date_end YYYY-MM-DD`   | Optional inclusive last date; activates multi-fold aggregation |

Every selected day must be represented in the dataset. Classification folds must have both true classes for ROC AUC to be defined.

### 4.3 Feature construction

| Option                    | Meaning                                                               |
|---------------------------|-----------------------------------------------------------------------|
| `--n_seconds`             | Duration of one feature-aggregation group; use a multiple of 10       |
| `--n_overlapping_seconds` | Positive literal overlap, or negative overlap relative to `n_seconds` |
| `--average_signals`       | `none`, `time`, `channel`, or `time_channel`                          |
| `--apply_log`             | Apply `log(max(x, 1e-23))` before aggregation                         |
| `--reduce_to_size`        | Retain this many central sensor channels before aggregation           |
| `--time_offset_seconds`   | Pair features at time `t` with targets near `t + offset`              |
| `--use_mid_target`        | Choose central/modal rather than minimum target during reduction      |

XGBoost and the simple fully connected NNs should normally use `channel` or `time_channel`, which produce a two-dimensional batch. The best examples use `channel`.

For `n_seconds=50` and `n_overlapping_seconds=-10`, the effective overlap is 40 seconds and the stride is 10 seconds.

### 4.4 Classification controls

| Option                          | Meaning                                                                       |
|---------------------------------|-------------------------------------------------------------------------------|
| `--classification_thresholds T` | Construct binary labels at distance `T`                                       |
| `--invert_threshold_logic`      | Make class 1 mean `y <= T` rather than `y > T`                                |
| `--join_higher_classes`         | Collapse labels above zero into binary class 1                                |
| `--balance_classes`             | `unbalanced`, `smote`, `adasyn`, `naive`, or `undersample` for training folds |

The current evaluator and global aggregation are binary. Keep `--join_higher_classes true` for supported comparisons.

`--balance_classes` changes training data only. The accepted `--balance_test` option is not connected in the active pipeline and should not be used as a claim that test folds were balanced.

### 4.5 Regression controls

| Option                                | Meaning                                                                          |
|---------------------------------------|----------------------------------------------------------------------------------|
| `--regression_threshold T`            | Remove targets above `T` before grouping, training, and testing                  |
| `--regression_evaluation_threshold T` | Add metrics for test targets `<= T` without changing training or overall metrics |
| `--y_min`, `--y_max`                  | Filter using the original HDF5 `y` before target construction                    |

Use `--regression_evaluation_threshold 1000` when the scientific question is “how does a model trained on the full range perform within 1,000 m?” Use `--regression_threshold 1000` when the intended model itself is trained and evaluated only within 1,000 m.

The accepted `--saturation_threshold` is not connected to the active reducer and does not currently clip targets.

### 4.6 Evaluation smoothing and uncertainty

| Option                        | Meaning                                                     |
|-------------------------------|-------------------------------------------------------------|
| `--instance_window W`         | Aggregate stride-one windows of `W` consecutive predictions |
| `--center_truth`              | Use the central true value instead of mean/majority truth   |
| `--compute_daywise_bootstrap` | Add sample-level daily classification uncertainty           |

Regression averages predictions within an instance window. Classification uses majority vote. This changes the evaluated unit and support; do not compare smoothed and unsmoothed metrics as if they represented the same observations.

Classification probabilities are not smoothed in the same way as predicted labels, so interpret AUC from an instance-window run cautiously.

`--random_state` controls most stochastic operations, but the classification confusion-matrix bootstrap currently uses a fixed seed of 42.

### 4.7 Neural-network controls

| Option            | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `--nn_epochs`     | Maximum epochs                                           |
| `--nn_hidden_dim` | Hidden dimension passed to `load_model`                  |
| `--nn_batch_size` | Training batch size                                      |
| `--nn_patience`   | Epochs without training-loss improvement before stopping |

The NN path monitors training loss, not validation loss. Its optimizer and actual learning rate come from the model file. `--nn_lr` is recorded but does not currently override the optimizer.

### 4.8 Identity and output controls

| Option                     | Meaning                                                         |
|----------------------------|-----------------------------------------------------------------|
| `--run_name`               | Human-readable stem used in the unique execution identity       |
| `--mlflow_experiment_name` | MLflow experiment and automatic results/log directory component |
| `--mlflow_tracking_uri`    | Local SQLite database or remote MLflow server                   |
| `--results_dir`            | Root for automatically named result directories                 |
| `--log_dir`                | Root for automatically named logs                               |
| `--joblib_save_file`       | Optional exact Joblib path                                      |
| `--log_file`               | Optional exact log path; existing files are not overwritten     |
| `--output_suffix`          | Insert a suffix before the Joblib extension                     |

Automatic paths include a UTC timestamp and random execution identifier, so repeating a command does not overwrite a previous automatic run.

## 5. Run experiments locally

### 5.1 Prefer maintained launchers for known experiments

The launch scripts resolve their own location, change to the repository root, and run the complete command. They are convenient reproducibility records:

```bash
bash scripts/run_xgb_regress_baseline_all_folds-best-1000.sh
bash scripts/run_xgb_classif_baseline_all_folds-best.sh
```

Inspect a script before running it on another machine, particularly its dataset path, date range, virtual-environment assumptions, and MLflow URI.

### 5.2 Run in a terminal multiplexer on remote machines

For a long local process on a remote host, use the site's preferred persistent session tool, such as `tmux` or `screen`, then launch the script normally. The experiment itself does not provide process resumption: if Python is terminated, the active fold does not resume from a checkpoint.

The execution log remains useful after a failure because it contains the full command and the last completed stage.

### 5.3 Select CPU or GPU appropriately

XGBoost baseline scripts use CPU-oriented settings. The NN handler automatically uses CUDA when PyTorch reports it available, otherwise CPU. Confirm the device and dependency installation on the target machine before launching a large NN grid.

## 6. Run experiments through a scheduler

The public baseline distribution does not include the project-specific HTCondor templates or institutional submission commands. To run a parameter sweep on a cluster, translate the commands in Section 1 into jobs for the local scheduler. Keep one unique `--run_name` per configuration, make the HDF5 file visible on worker nodes, and test one job before submitting the full sweep.

## 7. Monitor a running experiment

### 7.1 Follow the execution log

Automatic logs are stored under:

```text
logs/<experiment>/<run-name>-<execution-id>/execution.log
```

The console and log show data loading, reduced shapes and class counts, each test day, NN epoch loss when applicable, final metrics, saved paths, and MLflow identifiers.

### 7.2 Monitor MLflow

For the default local database, run:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`. On a remote host, bind the UI to loopback and use an SSH tunnel from your computer:

```bash
ssh -L 5000:127.0.0.1:5000 USER@REMOTE_HOST
```

Then open the same local browser address. If the experiment uses an HTTP MLflow tracking URI, open that configured service instead.

An active run may not yet contain final global metrics or artifacts. Those are written after all requested folds finish.

### 7.3 Recognize failure versus expected partial output

Expected behavior:

- single-day run: Joblib and log, normally no CSV;
- date-range run still in progress: incomplete MLflow artifacts;
- SQLite tracking: `mlflow.db` and artifact storage, not necessarily `mlruns/`.

Failure indicators:

- a traceback or error in the execution/Condor log;
- no successful completion message;
- a date-range run ending without the expected final Joblib;
- an MLflow run marked failed or left running after the process ended.

## 8. Understand generated files

With automatic naming, results use:

```text
results/<experiment>/<run-name>-<execution-id>/metrics.joblib
results/<experiment>/<run-name>-<execution-id>/metrics.csv
results/<experiment>/<run-name>-<execution-id>/metrics_model.joblib
```

The corresponding log lives beneath `logs/`. Depending on scope and failure point, not every listed file must exist.

### 8.1 Joblib is the complete machine-readable record

Its top-level structure is:

```python
{
    "metrics": {...},
    "metadata": {...},
}
```

Metadata records the resolved command configuration without the large `X`, `y`, and datetime arrays. Date-range metrics include `metrics_by_day`, aggregate summaries, and `final_results`. For both tasks, `final_results` contains the principal pooled metrics and complete-fold bootstrap intervals. Regression additionally stores its individual-frame bootstrap alternative in `frame_resampled_results`. Prediction and residual arrays are retained where configured, so Joblib may be large.

### 8.2 CSV is a compact date-range report

CSV is generated only when `metrics_by_day` exists. It is intended for quick inspection and tables, not as a complete replacement for Joblib.

Classification CSV omits AUC and confusion matrices. Regression CSV omits MSE, RMSE_STD, residual diagnostics, prediction arrays, and several other fields.

### 8.3 The model file is the last fold model

`metrics_model.joblib` contains the trained estimator from the final date fold. Do not describe it as a final model trained on all observations. For NN runs, the saved wrapper also contains its preprocessing state.

### 8.4 MLflow stores a comparison-oriented view

MLflow receives resolved parameters, principal scalar metrics, result files, plots, the last model, and the execution log for normal date-range runs. Use the execution ID to correlate MLflow with local directories.

## 9. Print and compare Joblib results

The repository provides `src/print_experiment_results.py`.

### 9.1 Print one experiment

```bash
python src/print_experiment_results.py \
  results/DAS-XGBoost-classification/RUN_DIRECTORY/metrics.joblib \
  --show-metadata \
  --csv classification-report.csv
```

The report detects task and scope and also writes a flattened comparison CSV. For classification, it prints pooled global metrics with complete-fold bootstrap intervals, per-class precision/recall/F1, the fold-weighted classification report, the summed confusion matrix, and individual results by day/fold. For regression, it prints pooled global metrics with complete-fold bootstrap intervals, the legacy individual-frame bootstrap results, equal-fold and support-weighted aggregate metrics, individual results by day/fold, and optional threshold-specific results.

### 9.2 Compare compatible experiments

```bash
python src/print_experiment_results.py \
  results/DAS-XGBoost-classification/BASELINE/metrics.joblib \
  results/DAS-XGBoost-classification/CANDIDATE/metrics.joblib \
  --show-metadata \
  --csv xgb-classification-comparison.csv
```

All input files must share task and scope. Compare regression with regression, classification with classification, and multi-fold with multi-fold.

The first file is treated as the baseline. For regression the main comparison uses MAE/RMSE, where lower is better. For classification it reports accuracy, macro/weighted F1, and class-specific precision/recall, where higher is better.

### 9.3 Inspect raw fields when needed

```bash
python - <<'PY'
from joblib import load

path = "results/EXPERIMENT/RUN/metrics.joblib"
result = load(path)

print(result.keys())
print(result["metrics"].keys())
print(result["metadata"])
PY
```

Use the exact result path printed by the experiment rather than guessing the latest directory.

## 10. Select and interpret metrics

### 10.1 Classification

#### Class polarity and confusion-matrix convention

For a single classification threshold `T`, `invert_threshold_logic` determines the semantic meaning of the numeric classes. Binary metric notation always treats class 1 as the conventional positive class and class 0 as the conventional negative class. Confusion matrices use actual classes as rows and predicted classes as columns:

```text
                         Predicted class 0   Predicted class 1
Actual class 0                   TN                  FP
Actual class 1                   FN                  TP
```

Equivalently, the stored layout is `[[TN, FP], [FN, TP]]`.

With `--invert_threshold_logic false`, the default mapping is:

```text
Class 0: distance <= T (nearby-vessel condition; conventional negative class)
Class 1: distance > T  (far/no-nearby-vessel condition; conventional positive class)
```

- TP means correctly predicting the far/no-nearby-vessel condition.
- TN means correctly predicting the nearby-vessel condition.
- FP means predicting the far/no-nearby-vessel condition when the vessel is actually nearby.
- FN means predicting the nearby-vessel condition when the vessel is actually farther than `T`.

With `--invert_threshold_logic true`, the mapping is:

```text
Class 0: distance > T  (far/no-nearby-vessel condition; conventional negative class)
Class 1: distance <= T (nearby-vessel condition; conventional positive class)
```

- TP means correctly predicting the nearby-vessel condition.
- TN means correctly predicting the far/no-nearby-vessel condition.
- FP means predicting the nearby-vessel condition when the vessel is actually farther than `T`.
- FN means predicting the far/no-nearby-vessel condition when the vessel is actually nearby.

Per-class precision, recall, and F1 treat each class as positive in turn. Thus, class 0 metrics remain meaningful even though class 0 is the conventional negative class in binary TP/TN/FP/FN notation. Always determine the mapping from the saved `classification_thresholds` and `invert_threshold_logic` metadata; do not infer it from a display label or assume that class 1 always means vessel detection.

Use the metric that matches the scientific cost:

- **Accuracy** summarizes the fraction correct but can hide class imbalance.
- **Weighted F1** weights each class F1 by observed support and is the primary global F1 stored by this pipeline.
- **Macro F1** treats both classes equally and is useful when minority-class performance matters.
- **Per-class precision** answers how often predictions of a class are right.
- **Per-class recall** answers how much of a true class is detected.
- **ROC AUC** measures score ranking across thresholds, but its class-1 meaning follows `--invert_threshold_logic`.
- **Confusion matrix** exposes the actual error types and should accompany summary metrics.

For the uploaded best XGBoost classification result, the final values were approximately weighted F1 `0.9011`, accuracy `0.9024`, and mean daily AUC `0.9442`. These values belong to its exact threshold, dates, preprocessing, and smoothing configuration.

### 10.2 Regression

- **MAE** is the mean absolute distance error and is usually the easiest to interpret.
- **RMSE** penalizes large errors more strongly.
- **MSE** is the squared-error quantity underlying RMSE.
- **R2** measures variance explained relative to predicting the target mean.
- **Residual summaries and plots** reveal bias, heavy tails, and changing variance that a single error metric can hide.

The uploaded 1,000 m, 50-second time-channel result has global MAE about `151.28` m and RMSE about `195.92` m. The channel-averaged command in the quick start is the preferred execution example identified in the subsequent configuration work. Results from the channel run should be read from its own new artifact rather than assigning the time-channel values to it.

### 10.3 Use the pooled global results for scientific reporting

For classification date ranges, `final_results` contains the principal pooled metrics and 95% intervals. Accuracy, precision, recall, F1, and per-class intervals resample complete daily confusion matrices; AUC is the mean daily AUC with complete days resampled for its interval.

For regression date ranges, `final_results` likewise contains the principal results: its point estimates pool all evaluated frames, and its intervals resample complete folds with replacement before recalculating each metric. The `frame_resampled_results` section retains the alternative that resamples individual pooled frames; `src/print_experiment_results.py` presents it separately as "Frame-resampled results and 95% confidence intervals." Threshold-specific frame-bootstrap values use the same explicit `frame_resampled_results` name inside `regression_threshold_evaluation`.

The result file does not store a separate generic confidence-interval section: each result entry carries the interval produced by its stated resampling procedure alongside its point estimate.

### 10.4 Always report support and interval

A point estimate without support and uncertainty can be misleading. Report:

- the metric and 95% interval;
- total and per-class support where relevant;
- the held-out date range;
- target/class definitions;
- preprocessing and smoothing controls.

## 11. Compare experiments fairly

Two results are directly comparable only when the scientific population and evaluation protocol match.

Keep fixed unless deliberately studied:

- dataset and target source;
- regression target inclusion range;
- classification threshold and inversion logic;
- held-out dates;
- feature duration, overlap, averaging, log transform, and sensor count;
- target grouping and time offset;
- evaluation instance window and center-truth choice;
- training balancing;
- random state;
- code and dependency revision.

Changing `--regression_threshold 5000` to `1000` changes the training and test population. Compare algorithms at 1,000 m with one another, or algorithms at 5,000 m with one another. To compare close-range performance while retaining a common 5,000 m training population, use the same `--regression_evaluation_threshold 1000` for every candidate.

For classification, state explicitly whether class 1 is near or far. The MLflow confusion-matrix display labels are fixed and may not reflect inverted threshold logic; metadata is authoritative.

## 12. Organize parameter sweeps

### 12.1 Define the grid before running it

The supplied factorial suites vary threshold, feature duration, evaluation window, and averaging. Record:

- every factor and allowed value;
- the primary selection metric;
- any tie-break metric;
- the common dates and random state;
- excluded/failed runs and reasons.

Do not add configurations only after seeing favorable results without marking that work as a new exploratory phase.

### 12.2 Use informative run names

The generated suites use names such as:

```text
xgb-classification-baseline-ct1000-ns50-iw5-avg-channel
nn-regression-rt5000-ns30-iw3-avg-time-channel-hd64-bs32
```

Recommended abbreviations are:

- `ct`: classification threshold;
- `rt`: regression threshold;
- `ns`: feature duration in seconds;
- `iw`: evaluation instance window;
- `avg`: averaging mode;
- `hd`: NN hidden dimension;
- `bs`: NN batch size.

The automatic execution suffix already makes each run unique; the human stem should explain its scientific configuration.

### 12.3 Treat parsed but inactive options carefully

The active custom-model path does not currently execute `--perform_grid_search` or `--param_grid`. Use explicit commands or generated jobs for sweeps. The concise warnings beside `--nn_lr`, `--balance_test`, and `--saturation_threshold` likewise describe their current behavior.

## 13. Troubleshooting

### No CSV was created

Check whether `--test_date_end` was supplied. A single-day result has no `metrics_by_day` aggregation and currently generates no CSV. This is a known output limitation, not evidence that the Joblib run itself failed.

### No `mlruns/` directory was created

The default is `sqlite:///mlflow.db`. Start the UI against that database. A remote tracking URI stores the run through the remote service.

### The HDF5 file cannot be found

`--h5_path` is independent of the current directory only when it is absolute. The maintained scripts change to the repository root and use a repository- relative `data/...` path. Verify the file exists on the machine or worker node.

### A classification fold fails during AUC

The test day likely contains only one class, or the classifier does not provide two probability columns. Inspect the reduced per-day class counts in the log.

### A model receives an incompatible feature shape

Use `channel` or `time_channel` for XGBoost and fully connected NNs. Reserve three-dimensional representations for models designed for them.

### An NN run fails with an empty batch loader

The NN loader uses `drop_last=True`. Ensure a training fold contains at least one complete `--nn_batch_size` batch, or reduce the batch size.

### An NN learning-rate change had no effect

Change the optimizer construction in the NN model file. `--nn_lr` is not currently applied to that optimizer.

### A run stopped and should be resumed

The training pipeline has no fold/epoch checkpoint-resume mechanism. Use the execution log to recover the exact command, fix the cause, and launch a new uniquely identified run. Do not reuse a partial result as a completed run.

### MLflow is unavailable but local files exist

The tracking call is part of execution and some failures can stop the run. Preserve the local log and any valid Joblib. Re-run with a reachable local SQLite URI if a complete tracked experiment is required.

## 14. Reproducibility and archiving checklist

For every result retained for comparison or publication, preserve:

- the exact model Python file;
- the exact command or launch script;
- the HDF5 dataset identity and version/checksum;
- the Git commit and dirty-worktree status;
- Python and dependency versions;
- Joblib and CSV results;
- the execution log;
- the MLflow experiment/run ID and exported artifacts where applicable;
- the selected metric, confidence interval, and support;
- the meaning of classification classes or regression target limits;
- the dates and all feature/evaluation controls;
- notes explaining failed or excluded folds/runs.

A compact pre-run record can be captured with:

```bash
git rev-parse HEAD
git status --short
python -V
python -m pip freeze
```

Store that information beside the experiment manifest or execution log. The automatic metadata is extensive, but it does not replace dataset versioning or a record of the source revision.

## 15. Recommended reporting pattern

For classification, report a statement such as:

> Binary XGBoost classification used a 1,000 m threshold, 50-second features, channel averaging, five-instance majority-vote evaluation, and leave-one-day- out folds from 16–25 June 2023. We report global weighted F1, accuracy, AUC, per-class precision/recall/F1, confusion matrix, support, and 95% bootstrap intervals.

For regression:

> XGBoost regression was trained and evaluated within the stated distance range using 50-second features, channel averaging, five-instance mean evaluation, and leave-one-day-out folds from 16–25 June 2023. We report global MAE, RMSE, MSE, R2, residual diagnostics, support, and 95% bootstrap intervals.

Replace these descriptions with the exact metadata from the retained Joblib. This closes the chain from command, through stored configuration and metrics, to a reproducible scientific claim.

## 16. Unused command-line arguments

The active pipeline accepts the following compatibility arguments but does not use them. Supplying them has no effect on execution or persisted artifact naming:

- `--model_output_suffix` does not change the trained-model artifact path; use a unique `--run_name` instead.
- `--vessel_joblib_path` does not affect targets or model execution.
- `--skip_if_output_exists` does not check for existing outputs or skip a run; rely on automatically unique paths and do not reuse an explicit `--joblib_save_file`.

<!-- Local Variables: -->
<!-- mode: markdown -->
<!-- ispell-local-dictionary: "en_US" -->
<!-- End: -->
