# Tutorial: developing regression and classification models

This tutorial walks through adding a model to the DAS vessel experiment framework, testing it on one day, running the full day-based experiment, and reading the results. The repository supplies the two baseline models:

- `models/baseline_xgb_regression_model.py`;
- `models/baseline_xgb_classification_model.py`.

The NN paths create new model files from the code contracts shown in Sections
6 and 7; prebuilt NN model files are intentionally not distributed.

The goal is to teach the development workflow. For an exhaustive description of every metric, output field, and current limitation, see the [model reference](model-reference.md).

All commands in this tutorial are run from the repository root.

> **Contributions and bug reports are welcome.** This research software is under active development. If you find a bug, an incorrect calculation, misleading documentation, or a reproducibility problem, please report it through the repository [issue tracker](https://github.com/UAH-PSI/das-vessel-detection/issues). Source-code contributions, tests, documentation corrections, and independently reproduced results are especially welcome. Before contributing substantial code changes, please describe them in an issue so their scope and compatibility with the experimental methodology can be discussed.

## 1. Choose a development path

There are two independent decisions:

1. Is the scientific target continuous regression or categorical classification?
2. Is the implementation a general sklearn-compatible model or a PyTorch neural network?

That produces four paths:

| Path                          | Task switch             | Model switch    | Demonstration file                            |
|-------------------------------|-------------------------|-----------------|-----------------------------------------------|
| XGBoost regression            | `--is_regression true`  | `--is_NN false` | `models/baseline_xgb_regression_model.py`     |
| XGBoost classification        | `--is_regression false` | `--is_NN false` | `models/baseline_xgb_classification_model.py` |
| Neural-network regression     | `--is_regression true`  | `--is_NN true`  | Create `models/tutorial_nn_regressor.py` in Section 6 |
| Neural-network classification | `--is_regression false` | `--is_NN true`  | Create `models/tutorial_nn_classifier.py` in Section 7 |

The task and implementation switches are separate. An XGBoost model is not implicitly a classifier, and setting `--is_NN true` does not determine whether the network performs regression or classification.

The rest of the tutorial first covers the steps shared by all four paths. You will then select the appropriate model-development branch before returning to a common experiment and result-inspection workflow.

## 2. Prepare the shared environment

### 2.1 Create and activate a virtual environment

For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Confirm that the experiment command can be imported and that its help is available:

```bash
python src/model_experiment_hdf5.py --help
```

The command must be launched from the repository root because the entry point imports other modules from `src/` and dynamically loads the model file passed on the command line.

### 2.2 Locate the HDF5 dataset

The runner expects the HDF5 datasets `X`, `y`, and `datetimes`. In the examples, replace the shell variable below with the actual dataset path:

```bash
H5_PATH=/absolute/path/to/dataset_sensor_range_1440_1690_0.h5
```

Using a shell variable keeps the later commands readable. Do not add quotation marks around the assignment unless the path contains spaces.

### 2.3 Choose a quick test day and a full date range

Model development should proceed in two stages:

1. Use a single held-out day to catch import, shape, loss, and estimator API errors quickly.
2. Use an inclusive date range for the scientific run and global results.

The examples use 16 June 2023 as the quick test day and 16–26 June 2023 as the full range. Adjust these dates to dates represented in your dataset.

### 2.4 Understand what the runner does for every model

You do not write data loading, splitting, evaluation, or persistence code in a model file. The runner:

1. loads `X`, `y`, and timestamps;
2. constructs the regression target or classification labels;
3. transforms consecutive 10-second source records into model examples;
4. holds out a complete day for testing;
5. trains a fresh model on all other days;
6. predicts and evaluates the held-out day;
7. repeats this for every requested test day;
8. aggregates results and writes Joblib, CSV, logs, and MLflow data.

This separation is important: a new model file should define model behavior, not quietly duplicate or change the experimental protocol.

## 3. Select shared feature and temporal settings

Before choosing the model path, decide how the DAS features become one input example.

### 3.1 Start with a two-dimensional model input

Both baseline XGBoost and the simple fully connected neural networks expect a batch shaped `(number_of_examples, number_of_features)`. The safest starting point is therefore:

```text
--average_signals channel
```

`channel` averages each source record across sensors and concatenates the result across the selected time span. `time_channel` is another tested option, but it averages over both sensors and time and consequently retains less temporal detail.

The other modes can create three-dimensional batches and are intended for models, such as CNNs, that explicitly support those shapes.

### 3.2 Select the feature window

Use `--n_seconds` to select how many consecutive seconds contribute to one model example. Because source records represent 10 seconds, choose a multiple of 10. For this tutorial:

```text
--n_seconds 30 --n_overlapping_seconds -10
```

This combines three source records. A negative overlap is interpreted relative to the window length: `30 + (-10) = 20` seconds of overlap, producing a 10-second stride.

### 3.3 Decide whether to log-transform features

The default is `--apply_log true`. Each value becomes `log(max(value, 1e-23))` before feature aggregation. Keep this fixed while comparing the four model paths unless the experiment is explicitly about the feature transform.

### 3.4 Select the reduction target and timestamp

Reduction combines source records into one model instance, selects its
training/test target, and independently assigns a representative timestamp.
This tutorial uses the canonical central convention:

```text
classification: --classification_target_method central_t
regression:     --regression_target_method central_t
both tasks:     --reduction_timestamp_method central_t
```

With the three-source-frame groups selected in Section 3.2, `central_t`
means the second source frame. The framework also supports positional,
aggregate, and binary class-presence target methods; see the
[model reference](model-reference.md#14-target-selection-and-temporal-controls).

Independent `legacy` values reproduce the maintained best IEEE JSTARS
configurations after substantial framework improvements. They are not a
generally validated compatibility mode for arbitrary model-development
experiments. Start new models with explicit central methods and use the
maintained legacy launchers when reproducing the paper.

## 4. Path A: develop a general regression model

Use this path for estimators that follow the scikit-learn interface, including XGBoost, LightGBM, random forests, linear models, and compatible custom estimators.

### 4.1 Study the baseline XGBoost regressor

Open `models/baseline_xgb_regression_model.py`. Its complete integration point is a zero-argument function:

```python
def load_model(seed=42):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=seed,
    )
```

Although the function declares an optional seed, the runner calls `load_model()` without arguments, so its default is used.

The returned estimator must provide:

- `fit(X_train, y_train)`;
- `predict(X_test)`.

The runner reserves 20% of every non-test-day training pool as validation. If the estimator's `fit` signature accepts `eval_set`, the validation arrays are passed through. Hyperparameters, objectives, and any early-stopping settings belong in the model file.

### 4.2 Create your own regression model file

Copy the baseline rather than changing it in place:

```bash
cp models/baseline_xgb_regression_model.py models/tutorial_xgb_regressor.py
```

Edit `tutorial_xgb_regressor.py`. For a first controlled experiment, change one factor, for example `max_depth`, and give the run a name that records that factor. Avoid putting data splitting or metric calculations into this file.

### 4.3 Run a one-day integration test

```bash
python src/model_experiment_hdf5.py \
  --h5_path "$H5_PATH" \
  --model_file models/tutorial_xgb_regressor.py \
  --is_regression true \
  --is_NN false \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --regression_threshold 5000 \
  --regression_target_method central_t \
  --reduction_timestamp_method central_t \
  --test_date_start 2023-06-16 \
  --run_name tutorial-xgb-regression-smoke
```

`--regression_threshold 5000` removes targets above 5,000 before grouping, so it changes both training and test populations. If you only want an additional evaluation view for close vessels, add, for example, `--regression_evaluation_threshold 1000`; that option does not change fitting or the overall metrics.

When the command succeeds, continue at Section 8.

## 5. Path B: develop a general classification model

Use this path for sklearn-compatible classifiers. A classifier needs one more method than a regressor because the framework calculates ROC AUC.

### 5.1 Study the baseline XGBoost classifier

Open `models/baseline_xgb_classification_model.py`:

```python
def load_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=42,
    )
```

The returned estimator must provide:

- `fit(X_train, y_train)`;
- `predict(X_test)` returning class indices;
- `predict_proba(X_test)` returning two probability columns.

The evaluator uses probability column 1 for ROC AUC. The current experiment aggregation is binary, so a new classifier should initially preserve this two-class contract.

The highest-impact inactive controls and implementation restrictions are
summarized in the [model reference](model-reference.md#6-known-implementation-issues).
Report additional defects through the repository
[issue tracker](https://github.com/UAH-PSI/das-vessel-detection/issues).

### 5.2 Define what class 1 means

With one threshold, the default mapping is:

```text
class 0: distance <= threshold
class 1: distance > threshold
```

If the scientific positive class should mean that a vessel is near the sensor, use `--invert_threshold_logic true`:

```text
class 0: distance > threshold
class 1: distance <= threshold
```

This choice changes the interpretation of precision, recall, F1, AUC, and the confusion matrix. Record it with every result and do not change it between models being compared.

Binary evaluation treats class 1 as the conventional positive class. Confusion
matrices use actual classes as rows and predicted classes as columns, producing
`[[TN, FP], [FN, TP]]`. The semantic meaning of those entries depends on
`invert_threshold_logic`; see the complete convention in the
[experiment guide](run-experiments.md#class-polarity-and-confusion-matrix-convention).

### 5.3 Create and test your classifier

```bash
cp models/baseline_xgb_classification_model.py models/tutorial_xgb_classifier.py
```

After making one model change, run:

```bash
python src/model_experiment_hdf5.py \
  --h5_path "$H5_PATH" \
  --model_file models/tutorial_xgb_classifier.py \
  --is_regression false \
  --is_NN false \
  --classification_thresholds 1000 \
  --classification_target_method central_t \
  --reduction_timestamp_method central_t \
  --invert_threshold_logic true \
  --join_higher_classes true \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --balance_classes unbalanced \
  --test_date_start 2023-06-16 \
  --run_name tutorial-xgb-classification-smoke
```

The quick-test day must contain both classes because ROC AUC is undefined for a test set containing only one true class.

After establishing an unbalanced baseline, `--balance_classes` can be changed to `smote`, `adasyn`, `naive`, or `undersample`. It affects the training fold only. Do not compare a balanced model with an unbalanced one without treating balancing as an experimental factor.

When the command succeeds, continue at Section 8.

## 6. Path C: develop a neural-network regression model

The NN path uses PyTorch but retains the same data preparation, held-out days, regression metrics, output structure, and MLflow experiment organization.

### 6.1 Study the simple regression network

Create `models/tutorial_nn_regressor.py` using the loader contract below.
Unlike a general estimator, its loader receives the already reduced training
arrays:

```python
def load_model(X_train, y_train, hidden_dim=64, seed=42):
    ...
    input_dim = X_train.shape[1]
    model = SimpleRegressionNN(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.HuberLoss()
    return model, optimizer, criterion
```

This contract lets the model infer its input dimension. The returned objects are:

1. a `torch.nn.Module`;
2. an optimizer bound to its parameters;
3. a loss function;
4. optionally, a scheduler as a fourth element.

A regression network must output `(batch,)` or `(batch, 1)`. The handler converts regression targets to `float32` and squeezes a one-column output before calculating loss.

The example uses Huber loss rather than MSE loss. This only changes training; the framework still reports MAE, MSE, RMSE, R2, and the shared regression diagnostics.

### 6.2 Complete your NN regression file

Add the imports, `SimpleRegressionNN` class, and `load_model` implementation
shown above to `models/tutorial_nn_regressor.py`. You can change the internal
layers, activation, regularization, optimizer, or criterion without changing
the runner. Keep the final layer at one output for scalar distance regression.

### 6.3 Run the NN integration test

```bash
python src/model_experiment_hdf5.py \
  --h5_path "$H5_PATH" \
  --model_file models/tutorial_nn_regressor.py \
  --is_regression true \
  --is_NN true \
  --nn_hidden_dim 64 \
  --nn_epochs 100 \
  --nn_batch_size 32 \
  --nn_patience 20 \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --regression_threshold 5000 \
  --regression_target_method central_t \
  --reduction_timestamp_method central_t \
  --test_date_start 2023-06-16 \
  --run_name tutorial-nn-regression-smoke
```

For two-dimensional input, the NN handler fits a `StandardScaler` on each training fold and applies it to test data. It uses all training-fold examples for optimization and stops according to training loss. There is no NN validation split in the current implementation.

The optimizer's learning rate is the one written in `load_model`. Although the CLI accepts and records `--nn_lr`, it does not currently override that value.

When the command succeeds, continue at Section 8.

## 7. Path D: develop a neural-network classification model

This path combines binary label construction and classification metrics with the PyTorch training contract.

### 7.1 Study the simple classification network

Create `models/tutorial_nn_classifier.py`. Its essential construction is:

```python
input_dim = X_train.shape[1]
output_dim = len(set(y_train))
model = SimpleClassificationNN(input_dim, hidden_dim, output_dim)

class_weights = torch.FloatTensor([0.3, 0.7])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.01,
)

return model, optimizer, criterion
```

The network must return raw logits shaped `(batch, number_of_classes)`. Do not apply softmax in `forward`: `CrossEntropyLoss` expects logits, and the runner applies softmax when probabilities are needed for AUC.

The demonstration uses loss weights `[0.3, 0.7]`. Those weights assume two classes in a specific order. If you change label semantics with `--invert_threshold_logic`, decide whether the weights still represent the intended scientific priority.

### 7.2 Complete and test your NN classifier

Add the imports, `SimpleClassificationNN` class, and loader implementation
shown above to `models/tutorial_nn_classifier.py`. Then run:

```bash
python src/model_experiment_hdf5.py \
  --h5_path "$H5_PATH" \
  --model_file models/tutorial_nn_classifier.py \
  --is_regression false \
  --is_NN true \
  --classification_thresholds 1000 \
  --classification_target_method central_t \
  --reduction_timestamp_method central_t \
  --invert_threshold_logic true \
  --join_higher_classes true \
  --balance_classes unbalanced \
  --nn_hidden_dim 128 \
  --nn_epochs 100 \
  --nn_batch_size 32 \
  --nn_patience 20 \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --test_date_start 2023-06-16 \
  --run_name tutorial-nn-classification-smoke
```

Always produce two logits for the supported binary task. Inferring `output_dim` solely from one fold can be unsafe if a training fold lacks a class; for a known binary experiment, explicitly setting `output_dim = 2` is a reasonable hardening change.

## 8. Inspect the shared smoke-test output

All four paths return here. A successful invocation logs the resolved Joblib and log paths. If `--joblib_save_file` was not supplied, the results follow this pattern:

```text
results/<MLflow experiment>/<run name>-<execution ID>/metrics.joblib
```

The execution log follows the same run identity under `logs/` and contains the full reconstructed command.

Load the Joblib result interactively:

```bash
python - <<'PY'
from joblib import load
from pathlib import Path

paths = sorted(Path("results").rglob("metrics.joblib"))
path = paths[-1]
result = load(path)

print("File:", path)
print("Top level:", result.keys())
print("Metric fields:", result["metrics"].keys())
print("Model file:", result["metadata"]["model_file"])
print("Task is regression:", result["metadata"]["is_regression"])
print("Task is NN:", result["metadata"]["is_NN"])
print("Prediction triplets:", result["metrics"].get("prediction_triplets"))
PY
```

This selects the lexically last metrics path only as a convenient tutorial shortcut. For rigorous analysis, use the exact path printed by the run.

A single-day smoke test writes Joblib but normally no CSV because global aggregation has not been performed. This is a known output limitation, not evidence that the Joblib run failed.

For regression, inspect `MAE`, `RMSE`, `MSE`, `R2`, `SUPPORT`, and residual diagnostics. For classification, inspect `accuracy`, `auc`, `classification_report`, and `confusion_matrix`.

## 9. Run the full shared date-range experiment

After the selected path succeeds on one day, reuse the same command and add an end date:

```text
--test_date_start 2023-06-16 --test_date_end 2023-06-25
```

Also change the run name from `...-smoke` to a descriptive scientific name. For example, the full XGBoost classification command becomes:

```bash
python src/model_experiment_hdf5.py \
  --h5_path "$H5_PATH" \
  --model_file models/tutorial_xgb_classifier.py \
  --is_regression false \
  --is_NN false \
  --classification_thresholds 1000 \
  --classification_target_method central_t \
  --reduction_timestamp_method central_t \
  --invert_threshold_logic true \
  --join_higher_classes true \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --balance_classes unbalanced \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-25 \
  --run_name tutorial-xgb-classification-full
```

The runner now fits a fresh model for every held-out date in the inclusive range. The final saved model artifact is the model fitted for the last fold; the Joblib metrics contain all daily evaluations and their aggregate results.

## 10. Read and compare the full results

### 10.1 Use the Joblib file as the complete record

The file contains:

```python
{
    "metrics": {...},
    "metadata": {...},
}
```

For date ranges, `metrics_by_day` contains every fold. For both tasks, `final_results` contains the principal global metrics and their 95% complete-fold bootstrap intervals. There is no separate generic confidence-interval dictionary: each result stores its point estimate together with the interval produced by its stated resampling procedure.

Classification `final_results` contains accuracy; support-weighted F1, precision, and recall; per-class F1, precision, and recall; and optional AUC. The confusion-matrix metrics use the matrix obtained by summing all fold matrices. Each bootstrap replicate samples complete folds with replacement, sums their matrices, and recalculates the metric. AUC instead uses the arithmetic mean of fold AUC values and resamples those fold values.

Regression `final_results` contains MAE, RMSE, MSE, and R2. Its point estimates pool all evaluated frames, while each interval resamples complete folds with replacement and recalculates the metric from the selected folds. Regression also stores `frame_resampled_results`, an explicitly separate alternative that bootstraps individual pooled frames. Threshold-specific frame-bootstrap values are stored under `regression_threshold_evaluation[threshold]["frame_resampled_results"]`. The detailed per-day results remain in `metrics_by_day`.

Where predictions are retained, each fold contains synchronized `datetimes`,
`y_pred`, and `y_true` arrays plus `prediction_triplets` of the form
`(timestamp, prediction, true_value)`. These are test-set values after any
evaluation aggregation and timestamp selection; train-set triplets are not
currently generated.

### 10.2 Use CSV for a compact report

The date-range run writes `metrics.csv` beside `metrics.joblib`.

- Classification CSV contains flattened daily classification reports, simple and fold-weighted summaries, and a `bootstrap final results` row populated from canonical `final_results`. AUC and confusion matrices remain in Joblib/MLflow.
- Regression CSV contains daily and aggregate MAE, support, RMSE, R2, and MAE_STD, followed by separate `final results` and `frame-resampled results` rows. Threshold-specific columns are added when requested. MSE and other regression diagnostics remain in Joblib.

### 10.3 Use MLflow to compare runs

The default tracking database is `sqlite:///mlflow.db`. Start the UI with:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`, select the experiment, and compare runs by their parameters and canonical global metrics from `final_results`. Regression `frame_resampled_results` is retained in Joblib and the textual report as an alternative, not promoted as the principal MLflow result. The artifacts include result files, plots, the last-fold model, and the execution log for a date-range run.

### 10.4 Make comparisons scientifically fair

When comparing XGBoost with a simple NN—or two variants of either—keep these fixed unless they are the intended experimental variable:

- task and class-threshold direction;
- training/test dates;
- feature averaging, window length, overlap, and log transform;
- regression inclusion/evaluation thresholds;
- classification balancing;
- reduction target and timestamp methods;
- evaluation window, task-specific evaluation method, and evaluation timestamp method;
- random state.

Do not compare a single-day value with a date-range global value. Do not treat the simple average of folds as interchangeable with the pooled bootstrap result. Record both the model file and complete CLI configuration.

## 11. Optional evaluation controls after the baseline works

### 11.1 Smooth consecutive predictions

Add `--instance_window 3` to group three consecutive reduced instances at
evaluation time. The canonical choices are:

```text
classification: --classification_evaluation_method majority
regression:     --regression_evaluation_method mean
both tasks:     --evaluation_timestamp_method central_i
```

The task-specific method is applied symmetrically to predictions and true
values. Majority is the natural categorical smoother for classification, while
mean is the corresponding numeric smoother for regression. The central
timestamp represents the temporal center of the evaluated window. Classification
also supports first, central, last, and binary class-presence methods.
Regression also supports first, central, last, minimum, maximum, and median.
Timestamp selection is independent of value selection. `central_i` requires an
odd instance window.

With no `--instance_window`, or with `instance_window=1`, evaluation value
and timestamp methods have no effect. With a larger window, aggregation changes
the evaluated unit and support; adjacent stride-one outputs overlap and are
correlated. Classification probability scores are not aggregated consistently
with the label method, so AUC from an `instance_window > 1` run is not a
comparable smoothed metric.

### 11.2 Add close-range regression metrics

Add:

```text
--regression_evaluation_threshold 1000
```

The runner retains overall metrics and adds a second set using test examples whose true reduced target is at most 1,000. This is preferable to changing `--regression_threshold` when the aim is to evaluate close-range performance without retraining on a restricted target population.

### 11.3 Compute daily classification bootstrap uncertainty

Add:

```text
--compute_daywise_bootstrap true
```

Each classification fold then stores sample-level bootstrap uncertainty for accuracy, macro F1, and weighted F1. This does not replace or alter the global date-range bootstrap intervals. The option is currently passed but not used by the regression evaluator.

## 12. Troubleshooting development failures

### The model file cannot be imported

Confirm that `--model_file` points to a Python file and that it defines `load_model` at module level. Run from the repository root and test syntax with:

```bash
python -m py_compile models/tutorial_xgb_regressor.py
```

### A general model receives a three-dimensional array

Use `--average_signals channel` or `time_channel`, or implement an estimator that deliberately accepts the higher-dimensional representation.

### `predict_proba` is missing

Classification evaluation always calls it for ROC AUC. Use a probabilistic classifier or provide a compatible wrapper that implements `predict_proba`.

### A PyTorch matrix multiplication has incompatible shapes

Infer `input_dim` from `X_train.shape[1]`, as the simple NN examples do, and use a two-dimensional input mode for a fully connected network. Print or log the reduced training shape while developing if necessary.

### Cross-entropy reports an output or target error

Return raw logits with shape `(batch, 2)` for binary classification. Do not apply softmax in the network, and use `CrossEntropyLoss`, which accepts the integer targets supplied by the handler.

### NN loss never improves or stops unexpectedly

Remember that early stopping monitors training loss. Check the optimizer learning rate in the model file, because `--nn_lr` does not currently change it. Also ensure that the training set has at least one full batch: the loader uses `drop_last=True`.

### Classification ROC AUC fails

Inspect class counts for the held-out day. Both true classes must occur. Also verify that the estimator or NN returns two probability columns and that class 1 has the meaning intended by the threshold switches.

### No CSV was generated

A single-day run has no `metrics_by_day` aggregation and currently produces no
CSV. Add `--test_date_end` for a date-range experiment. The output limitation
is summarized in the [model reference](model-reference.md#6-known-implementation-issues).

### No `mlruns/` directory appeared

The default tracker is the repository-local SQLite URI `sqlite:///mlflow.db`, not MLflow's legacy file-store default. Use the UI command from Section 10.3 or provide a different `--mlflow_tracking_uri`.

## 13. From tutorial model to publishable experiment

Before publishing a new model:

1. Keep its `load_model` contract minimal and self-contained.
2. State whether it is regression/classification and general/PyTorch.
3. Record model hyperparameters, optimizer/loss details, and random seeds.
4. Report the complete preprocessing, target, balancing, and smoothing CLI.
5. Use the same inclusive test-date range for every comparison.
6. Retain Joblib, CSV, MLflow artifacts, and the execution log.
7. Report global results with confidence intervals and relevant daily results.
8. Explain class-1 semantics for classification and target inclusion limits for regression.
9. Distinguish model selection experiments from final evaluation.
10. Update documentation if the model requires extending the evaluator or its binary/output contracts.

At that point, the small model file defines the algorithm, while the shared runner and stored metadata provide a reproducible account of how it was tested.

## 14. Disclaimer

This tutorial provides educational examples for research software that may contain bugs, incomplete features, or documentation errors. The examples are starting points for development, not validated production models. Results can depend on the dataset version, selected dates, class and threshold conventions, preprocessing, model implementation, random seeds, software dependencies, and evaluation options.

Users are responsible for validating example-derived models, metrics, confidence intervals, and generated artifacts against the saved metadata and the active runner code. Results should be independently reproduced where relevant before being used in scientific publications or operational decisions.

The software is provided without warranty under the terms of the repository license. It is not intended as a certified vessel-detection, navigation, safety, surveillance, or emergency-response system, and it must not be relied upon as the sole basis for operational or safety-critical decisions.

<!-- Local Variables: -->
<!-- mode: markdown -->
<!-- ispell-local-dictionary: "en_US" -->
<!-- End: -->
