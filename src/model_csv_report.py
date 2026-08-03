#!/usr/bin/env python3
"""
collect_metrics.py

Extended to handle two types of uncertainty:
1) --uncertainty=uncertainty
   - Classification => (upper_bound - lower_bound) / 2, but here the final CSV will show value (lower_limit, upper_limit)
   - Regression     => margin_of_error from "uncertainty"["RMSE"], with final CSV showing value (lower_limit, upper_limit)

2) --uncertainty in (confidence_interval, CI, ci)
   - Classification => 1.96 * sigma / sqrt(k)
       where k is the number of independent folds (days) in metrics_by_day (classification)
   - Regression => same formula, but k is the number of day_metrics in metrics_by_day,
                   sigma is stdev of the daily _STD metric (e.g. "MAE_STD" or "RMSE_STD").
"""

import argparse
import os
import re
import glob
import statistics
import math
from joblib import load
import csv
import sys
import numpy as np
from scipy.stats import t



def parse_args():
    parser = argparse.ArgumentParser(description="Collect metrics from joblib files into a CSV.")
    parser.add_argument("--result-dir", required=True, help="Directory containing joblib files.")
    parser.add_argument("--output-csv", required=True, help="Path for the output CSV file.")
    parser.add_argument("--threshold", required=True, type=str, help="Threshold value to filter files.")
    parser.add_argument("--metric", default="f1", help="Metric to collect (default: f1).")
    parser.add_argument(
        "--uncertainty",
        default=None,
        help="Uncertainty key. "
             "For classification default='uncertainty', for regression default='confidence_intervals'. "
             "Also accepts 'confidence_interval', 'CI', or 'ci' for custom daily approach."
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="If true and classification, output per-class metrics (plus uncertainty if CI)."
    )
    # New flag: Swap classes 0 and 1 (i.e. treat negatives as positives)
    parser.add_argument(
        "--swap-classes",
        action="store_true",
        help="If true, swap classes 0 and 1 before computing classification metrics."
    )
    # New flag: Compute final per-class F1 using aggregate counts instead of daily averages.
    parser.add_argument(
        "--aggregate-f1",
        action="store_true",
        help="If true, compute per-class F1 scores from aggregated counts instead of averaging daily values."
    )

    return parser.parse_args()

def debug_compute_f1(agg_cm, class_index):
    print("In lambda: aggregated matrix shape (after fix_shape):", agg_cm.shape)
    result = compute_f1_for_class(agg_cm, class_index)
    print("Computed f1 for class", class_index, ":", result)
    return result

def debug_lambda(cms, class_index):
    # Stack the matrices and print the shape.
    stacked = np.stack(cms, axis=0)
    print("debug_lambda: stacked shape:", stacked.shape)
    # Sum over the first axis.
    aggregated = np.sum(stacked, axis=0)
    print("debug_lambda: aggregated shape (before fix_shape):", aggregated.shape)
    # Apply fix_shape
    fixed = fix_shape(aggregated)
    print("debug_lambda: fixed shape:", fixed.shape)
    result = compute_f1_for_class(fixed, class_index)
    print("debug_lambda: computed f1 for class", class_index, "=", result)
    return result


def combine_metric(fold_stats, key_mean, key_var):
    """
    Combine per-fold estimates of a metric using a weighted average and a pooled variance.

    Parameters:
        fold_stats (list of dict): Each dict must contain:
            - 'n': number of observations in the fold
            - key_mean: the fold's mean for the metric
            - key_var: the fold's sample variance (ddof=1) for the metric.
        key_mean (str): The key for the mean value.
        key_var (str): The key for the variance value.

    Returns:
        overall_mean (float): The weighted overall mean.
        ci (tuple of float): The 95% confidence interval (lower, upper).
    """
    total_n = sum(item['n'] for item in fold_stats)
    if total_n < 2:
        raise ValueError("Not enough observations to compute CI.")

    # Compute the weighted overall mean.
    weighted_mean = sum(item[key_mean] * item['n'] for item in fold_stats) / total_n

    # Compute the pooled variance.
    numerator = sum(
        (item['n'] - 1) * item[key_var] + item['n'] * (item[key_mean] - weighted_mean) ** 2
        for item in fold_stats
    )
    pooled_variance = numerator / (total_n - 1)

    # Standard error and t-critical for a 95% CI.
    standard_error = (pooled_variance / total_n) ** 0.5
    df = total_n - 1
    t_crit = t.ppf(0.975, df)
    ci_lower = weighted_mean - t_crit * standard_error
    ci_upper = weighted_mean + t_crit * standard_error

    return weighted_mean, (ci_lower, ci_upper)


def compute_global_accuracy(cm):
    # For binary classification, assume cm = [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

def compute_f1_for_class(cm, class_index):
    """
    For class_index 0 or 1, compute F1 score.
    For a given class i:
      TP = cm[i, i]
      FP = sum(cm[:, i]) - TP
      FN = sum(cm[i, :]) - TP
    """
    TP = cm[class_index, class_index]
    FP = cm[:, class_index].sum() - TP
    FN = cm[class_index, :].sum() - TP
    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

def compute_weighted_f1(cm):
    """
    Compute weighted F1 by summing the per-class F1 weighted by support.
    """
    num_classes = cm.shape[0]
    weighted_f1_sum = 0.0
    total_support = 0.0
    for i in range(num_classes):
        support = cm[i, :].sum()
        f1 = compute_f1_for_class(cm, i) if support > 0 else 0.0
        weighted_f1_sum += f1 * support
        total_support += support
    return weighted_f1_sum / total_support if total_support > 0 else 0.0

def bootstrap_metric(daily_cms, metric_func, n_bootstrap=1000, random_seed=42):
    """
    Performs bootstrap resampling on daily confusion matrices.
    - daily_cms: list of confusion matrices for each day.
    - metric_func: a function that accepts an aggregated confusion matrix and returns the metric.
    Returns (lower_bound, upper_bound) as the 2.5th and 97.5th percentiles.
    """
    np.random.seed(random_seed)
    bootstrap_values = []
    n_days = len(daily_cms)
    for i in range(n_bootstrap):
        # Sample days with replacement.
        sample_indices = np.random.randint(0, n_days, size=n_days)
        sample_cms = [daily_cms[idx] for idx in sample_indices]
        aggregated_sample = np.sum(sample_cms, axis=0)
        # Debug: check the shapes
        try:
            metric_sample = metric_func(aggregated_sample)
        except Exception as e:
            print("DEBUG: In bootstrap iteration", i)
            print("DEBUG: sample_cms shapes:", [cms.shape for cms in sample_cms])
            print("DEBUG: aggregated_sample:", aggregated_sample, "with shape", aggregated_sample.shape)
            raise e
        bootstrap_values.append(metric_sample)
    lower_bound = np.percentile(bootstrap_values, 2.5)
    upper_bound = np.percentile(bootstrap_values, 97.5)
    return lower_bound, upper_bound

def bootstrap_metric_numbers(values, metric_func, n_bootstrap=1000, random_seed=42):
    """
    Bootstraps an iterable of numbers.

    :param values: list of numbers.
    :param metric_func: a function to compute the metric on the sampled values.
                        Typically, this could be statistics.mean.
    :param n_bootstrap: number of bootstrap samples.
    :param random_seed: seed for reproducibility.
    :return: (lower_bound, upper_bound) percentiles.
    """
    np.random.seed(random_seed)
    bootstrap_values = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample_indices = np.random.randint(0, n, size=n)
        sample = [values[i] for i in sample_indices]
        metric_sample = metric_func(sample)
        bootstrap_values.append(metric_sample)
    lower_bound = np.percentile(bootstrap_values, 2.5)
    upper_bound = np.percentile(bootstrap_values, 97.5)
    return lower_bound, upper_bound


def identify_task_type(user_metric):
    """
    Determine whether this is classification or regression based on the metric.
    """
    classification_metrics = {"f1", "accuracy", "precision", "recall", "f1-score"}
    regression_metrics = {"MSE", "RMSE", "MAE", "R2"}
    if user_metric.lower() in classification_metrics:
        return "classification"
    elif user_metric.upper() in {m.upper() for m in regression_metrics}:
        return "regression"
    else:
        # Fallback guess: classification if unknown
        return "classification"

def extract_experiment_name(fname):
    """
    Remove threshold, n_channels, and classification/regression to produce a shorter experiment name.
    """
    base = os.path.basename(fname)
    if base.endswith(".joblib"):
        base = base[:-6]  # remove ".joblib"

    base = base.replace("_classification_", "_")
    base = base.replace("_regression_", "_")

    base = re.sub(r"_threshold_\d+", "", base)
    base = re.sub(r"_n_channels_\d+", "", base)

    # reduce multiple underscores
    base = re.sub(r"__+", "_", base)
    base = base.strip("_")
    return base

def extract_n_channels(fname):
    """
    Return the integer n_channels if present, else None.
    """
    match = re.search(r"_n_channels_(\d+)", fname)
    if match:
        return int(match.group(1))
    return None

def compute_n_classification(classif_fpath):
    """
    Loads the classification joblib, sums over 'metrics_by_day' -> 'weighted avg' -> 'support'.
    Returns that sum as an integer (or float).
    """
    data = load(classif_fpath)
    metrics_dict = data["metrics"]
    metrics_by_day = metrics_dict.get("metrics_by_day", [])
    total_support = 0.0
    for day_metrics in metrics_by_day:
        cr = day_metrics.get("classification_report", {})
        wavg = cr.get("weighted avg", {})
        support_val = wavg.get("support", 0.0)
        total_support += support_val
    return total_support

def compute_std_over_days_classification(metrics_dict, user_metric):
    """
    For classification (daily), we want the standard deviation of the requested metric
    across days. By default, for "f1"/"f1-score", we look at 'weighted avg' -> 'f1-score'.
    If user_metric = "accuracy", we look at day_metrics["accuracy"].
    Returns the stdev (float) or 0.0 if there's not enough data.
    """
    mbd = metrics_dict.get("metrics_by_day", [])
    if not mbd:
        return 0.0

    label_metric = user_metric.lower()
    if label_metric in ("f1", "f1-score"):
        label_metric_for_report = "f1-score"
        fetch_from_weighted_avg = True
    elif label_metric == "accuracy":
        label_metric_for_report = "accuracy"
        fetch_from_weighted_avg = False
    else:
        # fallback
        label_metric_for_report = "f1-score"
        fetch_from_weighted_avg = True

    values = []
    for day_metrics in mbd:
        cr = day_metrics.get("classification_report", {})
        if fetch_from_weighted_avg:
            wavg = cr.get("weighted avg", {})
            val = wavg.get(label_metric_for_report)
        else:
            val = cr.get("accuracy")
        if val is not None:
            values.append(val)

    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)

def compute_std_over_days_classification_per_class(metrics_dict, user_metric, class_index):
    """
    Compute the standard deviation across days for the per-class f1 value.
    """
    mbd = metrics_dict.get("metrics_by_day", [])
    if not mbd:
        return 0.0
    values = []
    for day_metrics in mbd:
        day_cm = day_metrics.get("confusion_matrix", None)
        if day_cm is not None:
            f1_val = compute_f1_for_class(day_cm, class_index)
            values.append(f1_val)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def compute_std_over_days_regression(metrics_dict, user_metric):
    """
    For regression (daily), we want the stdev across days of e.g. "MAE_STD" or "RMSE_STD".
    We then return that as the 'sigma' for the daily approach.
    """
    mbd = metrics_dict.get("metrics_by_day", [])
    if not mbd:
        return 0.0

    metric_upper = user_metric.upper()
    if metric_upper == "MAE":
        day_key = "MAE_STD"
    elif metric_upper == "RMSE":
        day_key = "RMSE_STD"
    else:
        return 0.0

    values = []
    for day_metrics in mbd:
        val = day_metrics.get(day_key, None)
        if val is not None:
            values.append(val)

    if len(values) < 2:
        return 0.0
    return np.mean(values)

def fix_shape(cm):
    """
    Ensure that a confusion matrix is 2D.
    If cm is a 1D array of 4 elements, reshape it to 2x2.
    """
    cm = np.array(cm)
    print("fix_shape: input shape:", cm.shape)  # Debug print
    if cm.ndim == 1 and cm.size == 4:
        cm = cm.reshape((2,2))
    print("fix_shape: output shape:", cm.shape)  # Debug print
    return cm



def main():
    args = parse_args()
    user_metric = args.metric
    task_type = identify_task_type(user_metric)

    # If user didn't specify uncertainty, use defaults
    if args.uncertainty is None:
        if task_type == "classification":
            uncertainty_key = "uncertainty"
        else:
            uncertainty_key = "confidence_intervals"
    else:
        uncertainty_key = args.uncertainty


    result_dir = args.result_dir
    output_csv = args.output_csv

    # Get a broader list of candidate files.
    pattern = os.path.join(result_dir, f"*metrics*{task_type}*threshold_*{args.threshold}*.joblib")

    print(f'PATTERN: {pattern}')

    candidate_files = glob.glob(pattern)

    # Compile a regex that matches the threshold exactly.
    # The negative lookahead (?!\d) ensures that no digit follows the specified threshold.
    regex = re.compile(r"threshold_" + re.escape(args.threshold) + r"(?!\d)")

    # Filter the candidate files.
    file_list = sorted([f for f in candidate_files if regex.search(os.path.basename(f))])

    print(f'LEN FILELIST: {len(file_list)}')

    results = {}
    support_dict = {}
    class_support_dict = {}
    all_experiments = set()
    all_channels = set()

    for fpath in file_list:
        n_ch = extract_n_channels(fpath)
        exp_name = extract_experiment_name(fpath)

        data = load(fpath)
        metrics_dict = data["metrics"]

        if task_type == "classification":
            mbd = metrics_dict.get("metrics_by_day", [])

            daily_auc = []
            for day_metrics in mbd:
                if "auc" in day_metrics:
                    daily_auc.append(day_metrics["auc"])


            daily_cms = []
            for day_metrics in mbd:
                # Get the day's confusion matrix (assumed to be a numpy array)
                day_cm = day_metrics.get("confusion_matrix", None)
                if day_cm is not None:
                    arr = np.array(day_cm)
                    # If the array is 1-dimensional and has 4 elements, reshape it to 2x2.
                    if arr.ndim == 1 and arr.size == 4:
                        arr = arr.reshape((2,2))
                    daily_cms.append(arr)

            # If no confusion matrices were found, continue to next file
            if not daily_cms:
                print(f"[WARNING] No confusion matrices found in {fpath}", file=sys.stderr)
                continue

            # Check if classes should be swapped.
            if args.swap_classes:
                swapped_daily_cms = []
                for cm in daily_cms:
                    # Reverse the rows and columns.
                    # For a 2x2 matrix, this is equivalent to flipping it upside down and left-right.
                    cm_swapped = cm[::-1, ::-1]
                    swapped_daily_cms.append(cm_swapped)
                daily_cms = swapped_daily_cms


            # Aggregate daily confusion matrices.
            global_cm = np.sum(daily_cms, axis=0)
            # Optionally print for debugging.
            print(f"[DEBUG] Aggregated confusion matrix for {fpath}:\n{global_cm}")

            # Compute global support as the total number of examples (sum of all elements)
            global_support = int(np.sum(global_cm))
            support_dict[exp_name] = global_support

            # Also compute per-class support from the aggregated confusion matrix.
            support0 = int(np.sum(global_cm[0, :]))
            support1 = int(np.sum(global_cm[1, :]))
            class_support_dict[exp_name] = (support0, support1)

            # Compute global metrics using the aggregated confusion matrix.
            if user_metric.lower() in ("f1", "f1-score"):
                global_weighted_f1 = compute_weighted_f1(global_cm)
                metric_val = global_weighted_f1
            elif user_metric.lower() == "accuracy":
                metric_val = compute_global_accuracy(global_cm)
            elif user_metric.lower() == "auc":
                overall_auc = statistics.mean(daily_auc)  # aggregated AUC as mean over days
                metric_val = overall_auc
            else:
                metric_val = 0.0  # Fallback



            per_class_results = {}
            if args.per_class:
                # Assume binary classification with classes "0" and "1".
                if args.aggregate_f1:
                    # Compute per-class F1 using the aggregated confusion matrix.
                    f1_class0 = compute_f1_for_class(global_cm, 0)
                    f1_class1 = compute_f1_for_class(global_cm, 1)
                    if uncertainty_key == "uncertainty":

                        lower0, upper0 = bootstrap_metric(daily_cms, lambda agg: debug_compute_f1(fix_shape(agg), 0))
                        lower1, upper1 = bootstrap_metric(daily_cms, lambda agg: debug_compute_f1(fix_shape(agg), 1))


                        # lower0, upper0 = bootstrap_metric(daily_cms, lambda agg: compute_f1_for_class(fix_shape(agg), 0))
                        # lower1, upper1 = bootstrap_metric(daily_cms, lambda agg: compute_f1_for_class(fix_shape(agg), 1))




                        per_class_results["0"] = f"{f1_class0:.4f} ({lower0:.4f}, {upper0:.4f})"
                        per_class_results["1"] = f"{f1_class1:.4f} ({lower1:.4f}, {upper1:.4f})"
                    elif uncertainty_key in ("confidence_interval", "CI", "ci"):
                        mbd = metrics_dict.get("metrics_by_day", [])
                        k = len(mbd)
                        sigma0 = compute_std_over_days_classification_per_class(metrics_dict, 0)
                        sigma1 = compute_std_over_days_classification_per_class(metrics_dict, 1)
                        if k <= 1:
                            unc0 = 0.0
                            unc1 = 0.0
                        else:
                            unc0 = 1.96 * sigma0 / math.sqrt(k)
                            unc1 = 1.96 * sigma1 / math.sqrt(k)
                        per_class_results["0"] = f"{f1_class0:.4f} ± {unc0:.4f}"
                        per_class_results["1"] = f"{f1_class1:.4f} ± {unc1:.4f}"
                    else:
                        per_class_results["0"] = f"{f1_class0:.4f}"
                        per_class_results["1"] = f"{f1_class1:.4f}"


                else:
                    # Using day-level averages:
                    per_class_daily_vals = {"0": [], "1": []}
                    for day_cm in daily_cms:
                        f1_0 = compute_f1_for_class(day_cm, 0)
                        f1_1 = compute_f1_for_class(day_cm, 1)
                        per_class_daily_vals["0"].append(f1_0)
                        per_class_daily_vals["1"].append(f1_1)
                    mean0 = statistics.mean(per_class_daily_vals["0"]) if per_class_daily_vals["0"] else 0.0
                    mean1 = statistics.mean(per_class_daily_vals["1"]) if per_class_daily_vals["1"] else 0.0
                    if uncertainty_key == "uncertainty":
                        lower0, upper0 = bootstrap_metric_numbers(per_class_daily_vals["0"], lambda vals: statistics.mean(vals))
                        lower1, upper1 = bootstrap_metric_numbers(per_class_daily_vals["1"], lambda vals: statistics.mean(vals))
                        per_class_results["0"] = f"{mean0:.4f} ({lower0:.4f}, {upper0:.4f})"
                        per_class_results["1"] = f"{mean1:.4f} ({lower1:.4f}, {upper1:.4f})"
                    elif uncertainty_key in ("confidence_interval", "CI", "ci"):
                        k0 = len(per_class_daily_vals["0"])
                        k1 = len(per_class_daily_vals["1"])
                        sigma0 = statistics.stdev(per_class_daily_vals["0"]) if k0 >= 2 else 0.0
                        sigma1 = statistics.stdev(per_class_daily_vals["1"]) if k1 >= 2 else 0.0
                        if k0 <= 1:
                            unc0 = 0.0
                        else:
                            unc0 = 1.96 * sigma0 / math.sqrt(k0)
                        if k1 <= 1:
                            unc1 = 0.0
                        else:
                            unc1 = 1.96 * sigma1 / math.sqrt(k1)
                        per_class_results["0"] = f"{mean0:.4f} ± {unc0:.4f}"
                        per_class_results["1"] = f"{mean1:.4f} ± {unc1:.4f}"
                    else:
                        per_class_results["0"] = f"{mean0:.4f}"
                        per_class_results["1"] = f"{mean1:.4f}"

            if uncertainty_key == "uncertainty":
                if user_metric.lower() in ("f1", "f1-score"):
                    lower, upper = bootstrap_metric(daily_cms, compute_weighted_f1)
                elif user_metric.lower() == "accuracy":
                    lower, upper = bootstrap_metric(daily_cms, compute_global_accuracy)
                elif user_metric.lower() == "auc":
                    lower, upper = bootstrap_metric_numbers(daily_auc, lambda vals: statistics.mean(vals))
                else:
                    lower, upper = (0.0, 0.0)
                uncertainty_val = (lower, upper)

                if user_metric.lower() == "auc":
                        results["auc"] = {
                            "value": metric_val,      # overall AUC
                            "ci": (lower, upper)      # its confidence interval
                        }



            elif uncertainty_key in ("confidence_interval", "CI", "ci"):
                # For CI, compute the standard deviation of the daily metric values.
                mbd = metrics_dict.get("metrics_by_day", [])
                k = len(mbd)
                sigma = compute_std_over_days_classification(metrics_dict, user_metric)
                if k <= 1:
                    uncertainty_val = 0.0
                else:
                    uncertainty_val = 1.96 * sigma / math.sqrt(k)
                # support_dict[exp_name] = k
            else:
                uncertainty_val = 0.0

        else:
            # === Regression: gather the main metric and optional CI. ===
            metric_val = (metrics_dict.get(user_metric.upper()) or
                          metrics_dict.get(user_metric.lower()) or
                          metrics_dict.get(user_metric, None))
            if metric_val is None:
                print(f"[WARNING] Could not find a valid '{user_metric}' in {fpath}", file=sys.stderr)
                continue

            # # Always compute global support for regression as the number of days (folds)
            # mbd = metrics_dict.get("metrics_by_day", [])
            # global_support = len(mbd)
            # support_dict[exp_name] = global_support

            # Compute global support for regression by summing the per-day SUPPORT values.
            mbd = metrics_dict.get("metrics_by_day", [])
            global_support = sum(day.get("SUPPORT", 0) for day in mbd)
            support_dict[exp_name] = global_support



            # === Regression: gather the main metric and optional CI. ===
            metric_val = (metrics_dict.get(user_metric.upper()) or
                          metrics_dict.get(user_metric.lower()) or
                          metrics_dict.get(user_metric, None))
            if metric_val is None:
                print(f"[WARNING] Could not find a valid '{user_metric}' in {fpath}", file=sys.stderr)
                continue

            mbd = metrics_dict.get("metrics_by_day", [])
            global_support = sum(day.get("SUPPORT", 0) for day in mbd)
            support_dict[exp_name] = global_support

            if uncertainty_key == "uncertainty":
                # First, if the metric is R2, use the daily R2 values.
                if user_metric.upper() == "R2":
                    daily_r2 = []
                    for day in mbd:
                        if "R2" in day:
                            daily_r2.append(day["R2"])
                    if daily_r2:
                        lower, upper = bootstrap_metric_numbers(daily_r2, lambda vals: statistics.mean(vals))
                        overall_r2 = statistics.mean(daily_r2)
                    else:
                        overall_r2 = 0.0
                        lower, upper = (0.0, 0.0)
                    metric_val = overall_r2
                    uncertainty_val = (lower, upper)
                # Otherwise, use the fold_summary approach for MAE, MSE, or RMSE.
                elif user_metric.upper() in {"MAE", "MSE", "RMSE"}:
                    fold_stats = []
                    for day in mbd:
                        fs = day.get("fold_summary", None)
                        if fs is not None:
                            fold_stats.append(fs)
                    if not fold_stats:
                        print(f"[WARNING] No fold_summary found in metrics_by_day for {fpath}", file=sys.stderr)
                        uncertainty_val = (0.0, 0.0)
                    else:
                        metric_upper = user_metric.upper()
                        if metric_upper == "MAE":
                            metric_key = "mean_absolute_error"
                            var_key = "var_absolute_error"
                            overall_mean, (ci_lower, ci_upper) = combine_metric(fold_stats, metric_key, var_key)
                        elif metric_upper == "MSE":
                            metric_key = "mean_squared_error"
                            var_key = "var_squared_error"
                            overall_mean, (ci_lower, ci_upper) = combine_metric(fold_stats, metric_key, var_key)
                        elif metric_upper == "RMSE":
                            metric_key = "mean_squared_error"
                            var_key = "var_squared_error"
                            overall_mean_mse, (ci_lower_mse, ci_upper_mse) = combine_metric(fold_stats, metric_key, var_key)
                            overall_mean = math.sqrt(overall_mean_mse)
                            ci_lower = math.sqrt(ci_lower_mse) if ci_lower_mse >= 0 else 0.0
                            ci_upper = math.sqrt(ci_upper_mse)
                        else:
                            print(f"[WARNING] Uncertainty for regression metric {user_metric} not implemented. Setting uncertainty to 0.", file=sys.stderr)
                            overall_mean = metric_val
                            uncertainty_val = (0.0, 0.0)
                        if user_metric.upper() in {"MAE", "MSE", "RMSE"}:
                            uncertainty_val = (ci_lower, ci_upper)
                else:
                    # If uncertainty_key is 'uncertainty' but the metric is not one of the above:
                    uncertainty_val = (0.0, 0.0)
            elif uncertainty_key in ("confidence_interval", "CI", "ci"):
                sigma = compute_std_over_days_regression(metrics_dict, user_metric)
                if global_support <= 1:
                    margin = 0.0
                else:
                    margin = 1.96 * sigma / math.sqrt(global_support)
                lower = metric_val - margin
                upper = metric_val + margin
                uncertainty_val = (lower, upper)
            else:
                uncertainty_val = 0.0

            per_class_results = None



        # Now store results: store a tuple of (metric_val, uncertainty_val, per_class_results)
        # Note: For the "uncertainty" method, uncertainty_val is a tuple (lower, upper)
        pc_info = None
        if task_type == "classification" and args.per_class:
            pc_info = per_class_results

        if exp_name not in results:
            results[exp_name] = {}
        if n_ch is not None:
            results[exp_name][n_ch] = (metric_val, uncertainty_val, pc_info)
        else:
            results[exp_name][-1] = (metric_val, uncertainty_val, pc_info)

        all_experiments.add(exp_name)
        if n_ch is not None:
            all_channels.add(n_ch)

    # ========== Build CSV ==========
    sorted_experiments = sorted(all_experiments)
    sorted_channels = sorted(all_channels)

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Experiment"] + [str(ch) for ch in sorted_channels] + ["GlobalSupport"]


        if task_type == "classification" and args.per_class:
            header += ["Class0Support", "Class1Support"]

        writer.writerow(header)

        for exp_name in sorted_experiments:
            row = [exp_name]

            for ch in sorted_channels:
                val_unc_pclass = results[exp_name].get(ch)
                if val_unc_pclass is None:
                    cell_str = ""
                else:
                    # Unpack (metric_val, uncertainty_val, per_class_results)
                    if len(val_unc_pclass) == 3:
                        val, unc, pc_info = val_unc_pclass
                    else:
                        val, unc = val_unc_pclass
                        pc_info = None

                    # Separate handling for regression vs classification:
                    if task_type == "regression":
                        # Ensure that val is a scalar float:
                        v = float(val)
                        # If uncertainty is given as a tuple (for the "uncertainty" method)
                        if isinstance(unc, tuple):
                            # Compute symmetric margin
                            margin = (float(unc[1]) - float(unc[0])) / 2.0
                        else:
                            # Otherwise, assume unc is the margin (CI method)
                            margin = float(unc)
                        cell_str = f"{v:.4f} ({v - margin:.4f}, {v + margin:.4f})"
                    else:
                        # For classification, use the existing behavior.
                        if uncertainty_key == "uncertainty":
                            if pc_info and isinstance(pc_info, dict):
                                class_str = f"class0={pc_info['0']} ; class1={pc_info['1']}"
                                cell_str = f"{val:.4f} ({unc[0]:.4f}, {unc[1]:.4f}) ; {class_str}"
                            else:
                                cell_str = f"{val:.4f} ({unc[0]:.4f}, {unc[1]:.4f})"
                        else:
                            if pc_info and isinstance(pc_info, dict):
                                class_str = f"class0={pc_info['0']} ; class1={pc_info['1']}"
                                cell_str = f"{val:.4f} ± {unc:.4f} ; {class_str}"
                            else:
                                cell_str = f"{val:.4f} ± {unc:.4f}"
                row.append(cell_str)

            # Always append the global support value.
            sup_val = support_dict.get(exp_name, 0.0)
            row.append(f"{sup_val:.0f}")

            if task_type == "classification" and args.per_class:
                c0, c1 = class_support_dict.get(exp_name, (0.0, 0.0))
                row.append(f"{c0:.0f}")
                row.append(f"{c1:.0f}")

            writer.writerow(row)


    print(f"CSV written to: {output_csv}")

if __name__ == "__main__":
    main()
