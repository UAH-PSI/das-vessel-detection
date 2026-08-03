# Developing models in the experiment framework

This guide explains how to add a regression or classification model to the experiment runner in `src/model_experiment_hdf5.py`. It is intended for model authors: the runner owns data loading, temporal reduction, day-based train/test splits, optional class balancing, evaluation, and persistence, while a model file supplies only the estimator and its model-specific training objects.

The examples below assume commands are run from the repository root. Use `python src/model_experiment_hdf5.py --help` as the authoritative list of CLI defaults for the checked-out revision.

> **Contributions and bug reports are welcome.** This research software is under active development. If you find a bug, an incorrect calculation, misleading documentation, or a reproducibility problem, please report it through the repository [issue tracker](https://github.com/UAH-PSI/das-vessel-detection/issues). Source-code contributions, tests, documentation corrections, and independently reproduced results are especially welcome. Before contributing substantial code changes, please describe them in an issue so their scope and compatibility with the experimental methodology can be discussed.

## 1. General description and rationale

### 1.1 Why the framework separates models from experiments

An experiment should change one scientific factor at a time. Keeping the model in a small, dynamically loaded Python file lets researchers compare algorithms while reusing the same:

- HDF5 input and target preparation;
- feature aggregation and temporal alignment;
- leave-one-day-out evaluation protocol;
- classification balancing policy;
- metric implementation and uncertainty calculation;
- Joblib, CSV, log, and MLflow outputs.

The entry point imports `load_model` from the path supplied with `--model_file`. There are two integration contracts:

1. With `--is_NN false` (the default), `load_model()` returns an sklearn-compatible estimator.
2. With `--is_NN true`, `load_model(X_train, y_train, hidden_dim=...)` returns PyTorch training components.

Task type is independent of implementation type. Select regression with `--is_regression true`; classification is the default.

### 1.2 End-to-end data flow

For every invocation, the runner performs the following operations:

1. Load `X`, `y`, and `datetimes` from the HDF5 file.
2. Optionally replace the learning target from `--target_file` and `--target_key`. The HDF5 `y` still controls `--y_min`/`--y_max` filtering.
3. Optionally use `--reduce_to_size N` to keep only `N` sensor channels centered on the original channel axis; omit it to retain all channels.
4. Construct regression targets or classification labels.
5. Sort chronologically, optionally shift targets, group consecutive 10-second source records, transform features, and choose one target per group.
6. Hold out one complete day for testing. A date range repeats this process, fitting a fresh model for every day in the inclusive range.
7. For classification only, optionally rebalance each fold's training set.
8. Fit the model, predict the held-out day, optionally smooth predictions, and calculate fold metrics.
9. Aggregate date-range metrics and write local and MLflow results.

This is leave-one-day-out evaluation over the requested days, not ordinary random cross-validation. Samples from all other available dates form that fold's training pool. Consequently, date-range runs retrain the model once per test day; the model artifact saved at the end is the model from the last fold.

### 1.3 Input shapes and feature preparation

Each original `X[i]` is expected to be a two-dimensional sensor-by-feature map, and source observations are assumed to be 10 seconds apart. The number of source observations in an aggregation group is `n_seconds // 10`; an incomplete last group is discarded.

`--average_signals` determines the feature shape supplied to the model:

| Value          | Reduction of one group                      | Typical result             |
|----------------|---------------------------------------------|----------------------------|
| `none`         | concatenate records on axis 1               | 2-D sample; a batch is 3-D |
| `time`         | mean of records on axis 0                   | 2-D sample; a batch is 3-D |
| `channel`      | mean each record over sensors, then flatten | 1-D sample; a batch is 2-D |
| `time_channel` | mean over record and sensor axes            | 1-D sample; a batch is 2-D |

Only `channel` and `time_channel` are described by the CLI as tested modes. Most sklearn estimators require a 2-D batch, so use one of those modes unless the estimator itself accepts higher-dimensional input. The PyTorch handler deals with the input data depending on the input shapes:

- 3-D input, normally (batch, channels, features): treated as CNN input and left unscaled.
- 2-D input, normally (batch, features): treated as a dense-network input and standardized column by column using a StandardScaler fitted only on the training fold.

With `--apply_log true`, each source feature value becomes `log(max(x, 1e-23))` before aggregation. The transform therefore requires the scientific meaning of non-positive values to be considered carefully.

The grouping step advances by `group_size - overlap_size`. A positive `--n_overlapping_seconds` is literal overlap. A negative value produces overlap equal to `n_seconds + n_overlapping_seconds`; for example, `n_seconds=30` and `n_overlapping_seconds=-10` yields 20 seconds of overlap and a 10-second stride. The defaults (`10` and `-10`) also yield a 10-second stride.

### 1.4 Target selection and temporal controls

`--time_offset_seconds N` pairs the feature at time `t` with a target near `t + N` (within five seconds). Unmatched samples are removed.

Within each feature-aggregation group:

- Regression with `--use_mid_target true` uses the central target; `false` uses the minimum target.
- Classification with `--use_mid_target true --center_truth true` uses the central label.
- Classification with `--use_mid_target true --center_truth false` uses the modal label (ties resolve to the smallest class through `numpy.bincount`).
- Classification with `--use_mid_target false` uses the minimum label.

`--center_truth` has a second role during evaluation smoothing, described in Section 4.2. These are separate stages: first it can affect the target of each feature group, and later it can affect the ground truth of an instance window.

### 1.5 Implementing a conventional estimator

A conventional model file must define a zero-argument `load_model` and return an object with `fit(X, y)` and `predict(X)`. Classification additionally needs `predict_proba(X)` returning at least two columns because AUC is calculated from column 1.

```python
from sklearn.ensemble import RandomForestRegressor


def load_model():
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
```

For each fold, the runner randomly divides the training pool into 80% fitting and 20% validation data using `--random_state`. It fits only the 80% portion. If inspection of `fit` finds an `eval_set` parameter, the runner passes `[(X_validation, y_validation)]`; if it finds `verbose`, it attempts a quiet fit. Any early stopping policy and all hyperparameters belong in the estimator returned by `load_model`.

The supplied XGBoost examples are:

- `models/baseline_xgb_regression_model.py`
- `models/baseline_xgb_classification_model.py`

### 1.6 Implementing a PyTorch model

A PyTorch model file must define:

```python
def load_model(X_train, y_train, hidden_dim=256):
    # Build objects after inspecting the reduced training shape/classes.
    return model, optimizer, criterion
```

It may instead return `(model, optimizer, criterion, scheduler)`. If present, the scheduler is called once per epoch as `scheduler.step(training_loss)`.

For regression, the network must produce either shape `(batch,)` or `(batch, 1)`, and the loss must accept floating-point targets. For classification, it must produce unnormalized logits of shape `(batch, number_of_classes)` and the loss must accept integer class indices; prediction uses `argmax`, and probability prediction uses `softmax`.

The framework sets Python, NumPy, and PyTorch seeds, uses deterministic cuDNN settings, trains shuffled batches, and sets `drop_last=True`. Early stopping monitors mean **training loss**, not validation loss. Two-dimensional inputs are standardized; three-dimensional inputs are not, so a CNN should provide its own normalization (the existing examples use batch normalization).

The NN execution controls are `--nn_hidden_dim N` (the `hidden_dim` passed to `load_model`, default 256), `--nn_batch_size N` (training batch size, default 32), `--nn_epochs N` (maximum epochs, default 100), and `--nn_patience N` (epochs without training-loss improvement before early stopping, default 20). Because batches use `drop_last=True`, a training fold smaller than `--nn_batch_size` produces no complete training batch. Although `--nn_lr RATE` is recorded, the current handler does not apply it to the optimizer; set the learning rate when constructing the optimizer in `load_model`.

The public repository intentionally distributes only the two baseline XGBoost
model files. Sections 2.3 and 3.3 provide complete inline PyTorch contracts
that can be saved as new model files when developing an NN.

### 1.7 Command-line control map

The following table maps every accepted command-line option to the part of execution it controls. Boolean options take the literal value `true` or `false`. Defaults are shown by `python src/model_experiment_hdf5.py --help` and may change in later revisions.

| Area                       | Command-line option                                                          | Execution effect                                                                                                                                                                                                   |
|----------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input and model            | `--h5_path FILE`                                                             | Required HDF5 dataset containing `X`, `y`, and `datetimes`.                                                                                                                                                        |
| Input and model            | `--model_file FILE`                                                          | Required Python model module that supplies `load_model`.                                                                                                                                                           |
| Input and model            | `--target_file FILE` with `--target_key KEY`                                 | Replaces the learning target from an HDF5 or Joblib object. Both options must be supplied together.                                                                                                                |
| Task and implementation    | `--is_regression true\|false`                                                | Selects regression (`true`) or classification (`false`, the default).                                                                                                                                              |
| Task and implementation    | `--is_NN true\|false`                                                        | Selects the PyTorch handler (`true`) or the conventional sklearn-compatible path (`false`, the default).                                                                                                           |
| Classification labels      | `--classification_thresholds T [T ...]`                                      | Converts distance to one or more classification boundaries; the default is 1000 m. The old name `--thresholds` remains an accepted compatibility alias, but new commands should use `--classification_thresholds`. |
| Classification labels      | `--invert_threshold_logic true\|false`                                       | For one threshold, makes class 1 mean `y <= T` instead of the default `y > T`.                                                                                                                                     |
| Classification labels      | `--join_higher_classes true\|false`                                          | Collapses every class above zero into class 1 during grouping; keep the default `true` for the currently supported binary evaluator.                                                                               |
| Target inclusion           | `--y_min VALUE`, `--y_max VALUE`                                             | Inclusively filters observations using the original HDF5 `y`, before reduction. Either bound may be used alone.                                                                                                    |
| Regression target          | `--regression_threshold T`                                                   | Removes regression targets above `T` before grouping; the default is 5000.                                                                                                                                         |
| Regression evaluation      | `--regression_evaluation_threshold T`                                        | Adds metrics for reduced test targets `<= T` without changing training or overall metrics.                                                                                                                         |
| Target selection           | `--use_mid_target true\|false`                                               | Selects the central target of a group (`true`, default) or its minimum (`false`), subject to the classification behavior in Section 1.4.                                                                           |
| Temporal alignment         | `--time_offset_seconds N`                                                    | Pairs features at time `t` with targets near `t + N`; unmatched observations are removed.                                                                                                                          |
| Feature geometry           | `--reduce_to_size N`                                                         | Retains a centered slice of `N` sensor channels; omission retains every channel.                                                                                                                                   |
| Feature grouping           | `--n_seconds N`                                                              | Sets the duration of each aggregation window; source observations are assumed to be 10 seconds apart.                                                                                                              |
| Feature grouping           | `--n_overlapping_seconds N`                                                  | Sets literal overlap when positive; a negative value produces overlap `n_seconds + N`.                                                                                                                             |
| Feature grouping           | `--average_signals MODE`                                                     | Selects `none`, `time`, `channel`, or `time_channel` feature reduction as detailed in Section 1.3.                                                                                                                 |
| Feature transform          | `--apply_log true\|false`                                                    | Enables or disables the pre-aggregation logarithmic feature transform.                                                                                                                                             |
| Training folds             | `--test_date_start DATE`, `--test_date_end DATE`                             | Selects one held-out day or an inclusive range of held-out daily folds. Omitting the end date gives single-day mode.                                                                                               |
| Classification training    | `--balance_classes METHOD`                                                   | Selects `unbalanced`, `smote`, `adasyn`, `naive`, or `undersample` balancing of each training fold.                                                                                                                |
| Evaluation smoothing       | `--instance_window W`                                                        | Evaluates stride-one temporal windows of `W` reduced observations; omission evaluates individual reduced observations.                                                                                             |
| Evaluation smoothing       | `--center_truth true\|false`                                                 | Uses the center ground truth rather than an average or vote during instance-window evaluation; it also participates in grouped classification target selection.                                                    |
| Reproducibility            | `--random_state N`                                                           | Controls the conventional train/validation split, resampling, PyTorch seeds, and most bootstrap operations.                                                                                                        |
| NN training                | `--nn_hidden_dim N`, `--nn_batch_size N`, `--nn_epochs N`, `--nn_patience N` | Controls the NN handler as detailed in Section 1.6.                                                                                                                                                                |
| NN training                | `--nn_lr RATE`                                                               | Recorded but currently does not alter the optimizer; define the optimizer learning rate in `load_model`.                                                                                                           |
| Uncertainty                | `--compute_daywise_bootstrap true\|false`                                    | Adds sample-level uncertainty to classification fold metrics; it is currently unused by the regression evaluator.                                                                                                  |
| Interpretation             | `--freq_limit_joblib FILE`                                                   | Enables optional frequency-band SHAP analysis and storage; it does not change ordinary fitting or prediction.                                                                                                      |
| Local results              | `--results_dir DIRECTORY`                                                    | Sets the root for automatically named result directories; the default is `results`.                                                                                                                                |
| Local results              | `--joblib_save_file FILE`                                                    | Replaces automatic result placement with an exact Joblib path.                                                                                                                                                     |
| Local results              | `--output_suffix TEXT`                                                       | Inserts a suffix into the Joblib output name; the derived CSV follows that name.                                                                                                                                   |
| Logging                    | `--log_dir DIRECTORY`                                                        | Sets the root for automatically named execution logs; the default is `logs`.                                                                                                                                       |
| Logging                    | `--log_file FILE`                                                            | Requests an exact, non-overwriting execution-log path instead of automatic placement.                                                                                                                              |
| MLflow                     | `--mlflow_experiment_name NAME`                                              | Selects the MLflow experiment and is also used in automatic local result paths.                                                                                                                                    |
| MLflow                     | `--run_name NAME`                                                            | Supplies the human-readable run component; a unique execution ID is still appended.                                                                                                                                |
| MLflow                     | `--mlflow_tracking_uri URI`                                                  | Selects the local or remote MLflow backend; the default is `sqlite:///mlflow.db`.                                                                                                                                  |
| Diagnostic export          | `--save_fold_txt true\|false`                                                | Requests fold feature/label/datetime text exports for `channel` or `time_channel`; this facility is marked unavailable in the public version.                                                                      |
| Accepted inactive controls | `--saturation_threshold T`                                                   | Accepted but currently does not clip or otherwise change targets.                                                                                                                                                  |
| Accepted inactive controls | `--balance_test true\|false`                                                 | Accepted but currently does not rebalance the test fold.                                                                                                                                                           |
| Accepted inactive controls | `--perform_grid_search true\|false`, `--param_grid JSON`                     | Accepted, but the active dynamically loaded model path does not execute grid search.                                                                                                                               |
| Accepted inactive controls | `--model_output_suffix TEXT`                                                 | Accepted but does not affect the active model artifact path.                                                                                                                                                       |
| Accepted inactive controls | `--vessel_joblib_path FILE`                                                  | Accepted but does not affect the active pipeline.                                                                                                                                                                  |
| Accepted inactive controls | `--skip_if_output_exists true\|false`                                        | Accepted and defaults to `true`, but the active pipeline does not currently consult it before executing.                                                                                                           |

### 1.8 Reproducibility and model-development checklist

Before a full date-range run:

1. Copy the closest model example and change only `load_model` and its local classes/helpers.
2. Confirm the reduced feature dimensionality for the selected `--n_seconds`/`--average_signals` combination.
3. Fix estimator, optimizer, and framework seeds where supported.
4. Run one held-out day as a quick integration test.
5. Check that classification test days contain both classes; ROC AUC is not defined otherwise and the evaluator does not currently catch that error.
6. Run an inclusive date range to obtain global metrics, CSV, and plots.
7. Compare runs using identical preprocessing, thresholds, dates, balancing, and smoothing controls.
8. Retain the execution log: it records the complete reconstructed command.

## 2. Developing regression models

### 2.1 Regression target and sample inclusion

Use `--is_regression true`. Without an external target, the HDF5 `y` distances are the continuous targets. With `--target_file FILE --target_key KEY`, the external array becomes the learning target, but its length must align with the HDF5 arrays.

Three controls that sound similar have different purposes:

- `--y_min` and `--y_max` filter observations before target reduction. Their mask is always based on the original HDF5 `y`, even with an external target.
- `--regression_threshold T` (default `5000`) removes targets above `T` before grouping and therefore affects training and testing. Values equal to `T` are retained.
- `--regression_evaluation_threshold T` leaves training and overall evaluation unchanged, then reports a second view for test examples whose reduced true target is `<= T`.

The accepted `--saturation_threshold` currently does not reach the reducer and does not clip targets. Do not use it as evidence that clipping occurred.

### 2.2 Minimal conventional regression model

```python
from xgboost import XGBRegressor


def load_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
    )
```

Example date-range run:

```bash
python src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/my_regressor.py \
  --is_regression true \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --regression_threshold 5000 \
  --regression_evaluation_threshold 1000 \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-26 \
  --run_name my-regressor
```

### 2.3 Minimal PyTorch regression model

```python
import torch
from torch import nn


class Regressor(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_model(X_train, y_train, hidden_dim=256):
    model = Regressor(X_train.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    return model, optimizer, criterion
```

Add `--is_NN true` to the preceding command. Put optimizer parameters in the model file, including the learning rate.

## 3. Developing classification models

### 3.1 Label construction

Classification is selected with `--is_regression false` (the default).
Set a binary distance threshold `T` with
`--classification_thresholds T`. For one threshold:

- default: class 1 means raw distance `> T`, class 0 means `<= T`;
- with `--invert_threshold_logic true`: class 1 means `<= T`.

For several sorted thresholds `T0 < T1 < ...`, the initial label is zero and is incremented for every crossed threshold, producing ordinal classes. However, `--join_higher_classes true` (the default) clips all classes above zero to one during grouping.

The current evaluator and date-range aggregation are explicitly binary: they select probability column 1, reshape daily confusion matrices to 2 by 2, and calculate final metrics for classes 0 and 1. Therefore, keep `--join_higher_classes true` for supported experiments. A true multiclass extension requires corresponding changes to `ModelEvaluator` and the global aggregation in `PipelineExecutorHDF5`, not only a multiclass estimator.

If an external target is supplied, threshold construction is skipped. The current binary restrictions still apply to evaluation.

### 3.2 Class polarity and confusion-matrix convention

Binary evaluation uses class 1 as the conventional positive class and class 0
as the conventional negative class. Confusion matrices use actual classes as
rows and predicted classes as columns, giving `[[TN, FP], [FN, TP]]`.

With the default `--invert_threshold_logic false`, class 0 is the nearby-vessel
condition (`distance <= T`) and class 1 is the far/no-nearby-vessel condition
(`distance > T`). With `--invert_threshold_logic true`, these meanings are
reversed, so class 1 becomes the nearby-vessel condition. Per-class metrics
treat each class as positive in turn.

The complete TP, TN, FP, and FN interpretation for both polarities is in the
[experiment guide](README-run-experiments.md#class-polarity-and-confusion-matrix-convention).

### 3.3 Minimal conventional classifier

```python
from xgboost import XGBClassifier


def load_model():
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
    )
```

Example date-range run:

```bash
python src/model_experiment_hdf5.py \
  --h5_path data/dataset_sensor_range_1440_1690_0.h5 \
  --model_file models/my_classifier.py \
  --is_regression false \
  --classification_thresholds 1000 \
  --invert_threshold_logic true \
  --average_signals channel \
  --n_seconds 30 \
  --n_overlapping_seconds -10 \
  --balance_classes unbalanced \
  --test_date_start 2023-06-16 \
  --test_date_end 2023-06-26 \
  --run_name my-classifier
```

### 3.4 Minimal PyTorch classifier

```python
import numpy as np
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_model(X_train, y_train, hidden_dim=256):
    n_classes = len(np.unique(y_train))
    model = Classifier(X_train.shape[1], hidden_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion
```

Add `--is_NN true`. Ensure both class indices are represented in every training fold, and always return two logits for the supported binary case.

### 3.5 Training-set balancing

`--balance_classes` operates only on classification training folds, after the test day is held out:

- `unbalanced`: retain the observed distribution;
- `smote`: synthetic minority oversampling with two neighbors;
- `adasyn`: adaptive synthetic oversampling with two neighbors;
- `naive`: random oversampling;
- `undersample`: random majority undersampling.

The test fold is not modified by this path. Although `--balance_test` is accepted, the main pipeline does not pass it to the splitter, so it currently has no effect. Synthetic methods also require a conventional 2-D feature matrix and enough minority examples for their neighbor setting.

## 4. Evaluation metrics and command-line effects

### 4.1 Evaluation levels

Metrics appear at three distinct levels:

1. **Fold/day metrics** are calculated directly from predictions on one held-out day.
2. **Average metrics** are unweighted means of fold values; **fold-weighted metrics** weight fold values by their support.
3. **Final/global metrics** and 95% bootstrap intervals are produced only when both `--test_date_start` and `--test_date_end` select date-range mode.

A single-day run writes the evaluator's fold dictionary. It has no `metrics_by_day`, so `CsvSaver` deliberately writes no CSV. It also does not execute the main global MLflow metric-logging branches.

### 4.2 Evaluation-time temporal smoothing

With no `--instance_window`, metrics use one prediction per reduced feature group. With `--instance_window W` greater than one, test observations are sorted by datetime and all stride-one windows of `W` observations are used. There are `N - W + 1` evaluated windows.

For regression, predictions are averaged within each instance window. Ground truth is also averaged unless `--center_truth true`, in which case the central true value is used.

For classification, predictions use majority vote. Ground truth uses majority vote unless `--center_truth true`, in which case the central true class is used. Ties resolve to the smallest class. Classification probabilities are currently only truncated to the new output length; they are not grouped or centered in parallel with the voted labels. Therefore, interpret AUC from a smoothed run with caution.

### 4.3 Classification metrics

Interpret every class-specific metric and confusion-matrix entry using the
[class polarity convention](#32-class-polarity-and-confusion-matrix-convention)
defined above.

For a binary confusion matrix `[[TN, FP], [FN, TP]]`, each fold stores:

- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`.
- **ROC AUC**: area under the ROC curve using `predict_proba(X_test)[:, 1]` as the score for class 1.
- A scikit-learn **classification report**, containing precision, recall, F1-score, and support for each class plus accuracy, macro average, and support-weighted average.
- The **confusion matrix** itself.

For a class `c`, `precision = TPc / (TPc + FPc)`, `recall = TPc / (TPc + FNc)`, and `F1 = 2 * precision * recall / (precision + recall)` when denominators are nonzero. Weighted F1 weights each class F1 by its true support.

For date ranges, daily confusion matrices are summed to form a global matrix. Global accuracy; support-weighted F1, precision, and recall; and per-class F1, precision, and recall are computed from that matrix and stored in `final_results`. Their 95% intervals come from 1,000 bootstrap resamples of whole daily confusion matrices, with replacement, using seed 42. Each resample sums the selected matrices before recalculating the metric.

Global AUC is different: it is the arithmetic mean of daily AUC values. Its 95% interval comes from 1,000 bootstrap resamples of those daily values using `--random_state`; `auc_std` is the population standard deviation across days.

`--compute_daywise_bootstrap true` adds an `uncertainty` dictionary to every day for accuracy, macro F1, and weighted F1. Each uses 1,000 sample-level bootstrap resamples and stores mean, population standard deviation, and the 2.5/97.5 percentiles. Resamples containing only one true class are skipped. This switch does not change the final/global intervals.

### 4.4 Regression metrics

Let residual `ei = yi - yhat_i` and let `n` be the evaluated support. Every fold stores:

- `MEAN_TARGET = mean(yi)`;
- `SUPPORT = n`;
- `MAE = mean(abs(ei))`;
- `MSE = mean(ei^2)`;
- `RMSE = sqrt(MSE)`;
- `R2 = 1 - sum(ei^2) / sum((yi - mean(y))^2)`;
- `MAE_STD = std(abs(ei))`, using NumPy's population standard deviation;
- `RMSE_STD = std(ei^2)`. Despite its name, this is the standard deviation of squared errors, not uncertainty of RMSE and not in the same units as RMSE.

Diagnostic fold fields include one- and two-sided one-sample t tests of absolute and squared errors against zero, Shapiro-Wilk statistic/p-value, residual skewness, and excess kurtosis. Shapiro-Wilk is skipped and stored as NaN above 5,000 residuals. The `fold_summary` dictionary retains means and sample variances (`ddof=1`) of residual, squared error, and absolute error.

In a date-range run, `average_metrics` is the simple mean of every scalar fold field (apart from `day` and `fold_summary`). `fold_weighted_metrics` weights these values by fold `SUPPORT`; its `SUPPORT` is the sum. Prediction arrays, residual arrays, SHAP dictionaries, and other nonscalars remain only under `metrics_by_day`.

Final MAE, MSE, RMSE, and R2 use exact pooled point estimates with complete-fold bootstrap intervals. The runner pools all original fold statistics for each point estimate, then draws 1,000 sets of complete folds with replacement using `--random_state`, pools each selected set, and recalculates the metric. The 2.5/97.5 percentiles form the interval. These principal values and intervals are stored in `final_results`.

The alternative `frame_resampled_results` concatenates fold residuals, resamples individual frames 1,000 times, and stores the mean bootstrap value plus 2.5/97.5 percentiles. It is retained as a separately labeled result and is not used as the principal global metric.

With `--regression_evaluation_threshold T`, the same fold metrics are calculated on `yi <= T`, and a second 1,000-resample individual-frame bootstrap produces threshold-specific `frame_resampled_results`. A fold with no eligible observations has zero support and NaN scalar metrics. This view never changes fitting or the overall results.

`--compute_daywise_bootstrap` is passed to the regression evaluator but is not currently used there; regression global bootstrap results are always produced for date-range runs.

### 4.5 Confidence-interval storage

There is no separate generic confidence-interval dictionary. Each entry in `final_results` stores its canonical point estimate together with the corresponding complete-fold bootstrap interval. Regression entries in `frame_resampled_results` similarly carry their individual-frame bootstrap intervals. This keeps each interval attached to the calculation and sampling unit that produced it.

### 4.6 Controls that change metric interpretation

For valid comparisons, keep these controls fixed:

- `--classification_thresholds`, `--invert_threshold_logic`, and `--join_higher_classes` define classification labels.
- `--regression_threshold`, `--y_min`, and `--y_max` change included samples; `--regression_evaluation_threshold` only adds a subset view.
- `--n_seconds`, `--n_overlapping_seconds`, `--average_signals`, `--apply_log`, `--use_mid_target`, and `--time_offset_seconds` change the training examples or targets.
- `--instance_window` and `--center_truth` change the units evaluated and can reduce support.
- `--balance_classes` changes classification training data, not test data.
- `--test_date_start`/`--test_date_end` determine the held-out folds and whether global aggregation exists.
- `--random_state` affects the internal train/validation split, balancing, PyTorch training, and regression/AUC bootstraps. Classification confusion- matrix bootstraps use their helper's fixed seed 42.

## 5. Joblib, CSV, and MLflow outputs

These are alternative views of the same run, but they are not identical. The Joblib file is the complete machine-readable record; CSV is a compact report; MLflow is a searchable scalar-and-artifact view.

### 5.1 Paths and run identity

Unless `--joblib_save_file FILE` supplies an exact output path, `--results_dir DIRECTORY` controls the automatic result root and output is created as:

```text
<results-dir>/<experiment>/<run-name>-<UTC-timestamp>-<random-id>/metrics.joblib
```

Here, `<results-dir>` comes from `--results_dir DIRECTORY` (default `results`), `<experiment>` comes from `--mlflow_experiment_name NAME`, and the human-readable part of the directory comes from `--run_name NAME`; a unique execution ID is appended. The corresponding log uses the same identity beneath `--log_dir DIRECTORY` (default `logs`), unless `--log_file FILE` supplies an exact path. Existing explicit log files are never overwritten. `--output_suffix TEXT` is inserted before the Joblib extension, and the CSV derives its name from the resulting Joblib path. Although `--skip_if_output_exists` is accepted and defaults to `true`, the active pipeline does not currently consult it, so it does not protect a manually selected output path from an attempted rerun.

The `--mlflow_tracking_uri URI` option selects the MLflow backend. Its default, `sqlite:///mlflow.db`, creates a repository-local SQLite tracking database and artifact directories rather than the legacy default `mlruns/` layout. Supply an HTTP(S) URI through this option to use a remote server.

### 5.2 Joblib schema

The Joblib file contains:

```python
{
    "metrics": <single-fold or aggregated metrics dictionary>,
    "metadata": <resolved command configuration without X, y, datetimes>,
}
```

For date ranges, classification `metrics` contains aggregated and average confusion matrices, average and fold-weighted classification reports, `metrics_by_day`, and canonical `final_results`. Regression contains `average_metrics`, `fold_weighted_metrics`, `metrics_by_day`, canonical `final_results`, and the alternative `frame_resampled_results`, plus `regression_threshold_evaluation` when requested.

Unless disabled by programmatic configuration, each day also stores `y_true`, `y_pred`, and datetimes. Regression stores residuals. With `--freq_limit_joblib FILE`, each day additionally stores sampled train/test SHAP values, data, base values, and frequency-band feature names. This option enables an interpretation/export path; it does not alter ordinary model fitting or prediction. These arrays can make the Joblib file large.

### 5.3 CSV schema

CSV is written only when `metrics_by_day` exists, which currently means a date-range run.

Classification CSV columns flatten the average classification-report schema. Rows contain each day, `macro avg` (the simple average of fold reports), `fold-weighted avg`, and `bootstrap final results`. The final row inserts global accuracy, weighted F1, and per-class F1 with their intervals into matching columns. AUC and confusion matrices are not CSV columns.

Regression CSV columns are limited to `MAE`, `SUPPORT`, `RMSE`, `R2`, and `MAE_STD`. Rows contain each day, macro average, fold-weighted average, canonical final results, and frame-resampled results. When evaluation thresholding is enabled, parallel `threshold_...` columns are appended. Although MSE, RMSE_STD, diagnostics, and other metrics remain in Joblib, they are not written to this CSV.

### 5.4 MLflow parameters, metrics, and artifacts

The runner logs nearly every resolved CLI/config value as an MLflow parameter, excluding the feature and target arrays. It adds a feature suffix inferred from the HDF5 filename and an `execution_id` tag.

For a classification date range, scalar MLflow metrics include:

- `global_f1`, `global_accuracy`, optional `global_auc`, and their `_low_CI`/`_high_CI` bounds;
- class-0/class-1 F1 and bounds;
- per-class and weighted precision/recall;
- `avg_accuracy`, optional `avg_auc` and `auc_std`;
- total test support and class supports.

For a regression date range, scalar MLflow metrics include:

- `global_MAE`, `global_RMSE`, `global_MSE`, `global_R2` and bounds;
- all scalar simple averages as `avg_<key>` and support-weighted values as `fold_weighted_<key>` (excluding support);
- pooled target count, mean, standard deviation, and median;
- pooled residual mean/std and mean daily skewness/kurtosis;
- when requested, `threshold_support`; threshold frame-resampled metrics remain in Joblib and the human-readable report rather than being promoted to MLflow globals.

Date-range artifacts include Joblib and CSV beneath the automatic `--results_dir` location (or beside the exact `--joblib_save_file`), the last fold's serialized model under the MLflow `model/` artifact path, classification confusion/per-day plots or regression residual/per-day plots, and the execution log beneath `--log_dir` or at the exact `--log_file` path.

Single-day runs do not log the main classification/regression scalar set or standard result artifacts. A single-day regression run with `--regression_evaluation_threshold` logs its threshold support. In all cases, the local Joblib and execution log are the reliable records.

## 6. Known implementation issues

The framework still has concrete implementation gaps that can make an accepted option ineffective, reject an otherwise useful fold, or produce a misleading metric or artifact. The complete backlog, workarounds, and resolution criteria are maintained in [EXPERIMENT-KNOWN-ISSUES.md](EXPERIMENT-KNOWN-ISSUES.md).

The highest-impact restrictions for model authors are:

| Area | Current restriction | Required user action |
|------|---------------------|----------------------|
| Inactive controls | `--nn_lr`, `--balance_test`, `--saturation_threshold`, `--perform_grid_search`, `--param_grid`, `--model_output_suffix`, `--vessel_joblib_path`, and `--skip_if_output_exists` do not affect the active execution path. | Do not cite these values as operations or hyperparameters that occurred. Use the workarounds in the detailed issue document. |
| Partial controls | `--compute_daywise_bootstrap` is classification-only in practice, and `--save_fold_txt` is unavailable as a supported public workflow. | Use regression `final_results` for global uncertainty and Joblib for supported fold data. |
| Classification | Evaluation is binary even though the CLI can construct multiple classes; single-class folds can fail AUC. | Keep `--join_higher_classes true`, profile per-day class support, and do not claim multiclass support. |
| Smoothed AUC | `--instance_window` votes labels but does not aggregate probability scores equivalently. | Do not use smoothed-run AUC as a comparable result. |
| NN training | Early stopping monitors training loss, `--nn_lr` is inactive, and `drop_last=True` can empty a small fold. | Set learning rate in `load_model` and keep `--nn_batch_size` no larger than the smallest training fold. |
| Conventional training | Estimators without `eval_set` still lose the 20% validation split. | Report that such models currently fit on 80% of the non-test pool. |
| Metrics | `RMSE_STD` is the standard deviation of squared errors, not uncertainty of RMSE. | Do not interpret `RMSE_STD` as RMSE uncertainty; use `final_results` for principal intervals. |
| Outputs | Single-day CSV/MLflow output is incomplete, and a date-range run persists only its last-fold model. | Treat single-day Joblib/log files as authoritative and label the saved date-range model as the last-fold model. |
| Presentation and seeds | MLflow confusion labels may contradict `--invert_threshold_logic`, and classification confusion bootstrap uses fixed seed 42. | Determine class meaning from metadata and record the fixed-seed exception. |

Treat these as explicit boundaries when publishing results. A fix should include focused tests and updates to the CLI help, this guide, the detailed issue document, and any affected run/tutorial documentation.

## 7. Disclaimer

This guide describes extension points in research software that may contain bugs, incomplete features, unsupported interfaces, or documentation errors. Results can depend on the dataset version, selected dates, class and threshold conventions, preprocessing, model implementation, random seeds, software dependencies, and evaluation options.

Model authors and users are responsible for validating new implementations, tests, metrics, confidence intervals, and generated artifacts against the saved metadata and the active runner code. Do not assume that an undocumented interface or accepted command-line argument is functional. Results should be independently reproduced where relevant before being used in scientific publications or operational decisions.

The software is provided without warranty under the terms of the repository license. It is not intended as a certified vessel-detection, navigation, safety, surveillance, or emergency-response system, and it must not be relied upon as the sole basis for operational or safety-critical decisions.

<!-- Local Variables: -->
<!-- mode: markdown -->
<!-- ispell-local-dictionary: "en_US" -->
<!-- End: -->
