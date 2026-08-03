import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import logging

# module-level logger
logger = logging.getLogger(__name__)
_shapiro_large_sample_warning_emitted = False
from sklearn.utils import shuffle
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_1samp, shapiro, skew, kurtosis

import csv
from bisect import bisect_left
import shap


def _nearest_datetime_index(datetimes, target_dt, tolerance):
    """Return the nearest timestamp index within tolerance, preferring earlier ties."""
    insert_position = bisect_left(datetimes, target_dt)
    candidate_indices = []

    # Put the preceding timestamp first so exact ties resolve to the earlier one.
    if insert_position > 0:
        candidate_indices.append(insert_position - 1)
    if insert_position < len(datetimes):
        candidate_indices.append(insert_position)

    if not candidate_indices:
        return None

    matched_index = min(
        candidate_indices,
        key=lambda index: abs(datetimes[index] - target_dt),
    )
    if abs(datetimes[matched_index] - target_dt) > tolerance:
        return None
    return matched_index


def compute_shap_values(model, X_train, X_data, feature_names, nsamples_background=100, nsamples_explain=None):
    # Create a background sample from X_train
    background = shap.utils.sample(X_train, nsamples=nsamples_background)
    explainer = shap.Explainer(model, background, feature_names=feature_names)

    # Optionally, if X_data is huge, sample for explanation
    if nsamples_explain is not None and len(X_data) > nsamples_explain:
        X_data_sample = shap.utils.sample(X_data, nsamples=nsamples_explain)
    else:
        X_data_sample = X_data

    # Compute SHAP values
    try:
        # For newer SHAP versions
        shap_values = explainer(X_data_sample, check_additivity=False)
    except TypeError:
        # For older SHAP versions that don't support 'check_additivity'
        shap_values = explainer(X_data_sample)

    return shap_values


class TripletReducer:
    def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None,
                 join_higher_classes=True, average_signals=False, apply_log=True,
                 epsilon=1e-23, time_offset_seconds=None, use_mid_target=False,
                 sample_seconds=10, center_truth=False):
        """
        Initialize with the data to be reduced and the parameters.
        - X: List of 2D numpy arrays.
        - y: List of integers (classification) or floats (regression).
        - dt: List of utc-aware datetime objects.
        - ships: Optional ship data to reduce, list of lists (each sublist contains dictionaries with ship data).
        - n_seconds: Total number of seconds to average.
        - n_overlapping_seconds: If specified, overlap between groups in seconds.
        - join_higher_classes: If True, convert all y >= 1 to 1 (for classification).
        - average_signals: Averaging method or backward-compatible boolean.
            - "none": No averaging; concatenates samples in the time dimension.
            - "time": Average across time dimension only, keeping channel information.
            - "channel": Average across channel dimension only, keeping time information.
            - "time_channel": Average across both time and channel dimensions.
            - True: Equivalent to "time_channel" for backward compatibility.
            - False: Equivalent to "none" for backward compatibility.
        - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
        - epsilon: Small value added to X before applying log to avoid NaNs.
        - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
        - use_mid_target: If True, calculate the mean of target values in the group.
        """
        self.X = X
        self.y = y
        self.dt = dt
        self.ships = ships  # Ship data to be reduced if provided
        self.n_seconds = n_seconds
        self.n_overlapping_seconds = n_overlapping_seconds
        self.join_higher_classes = join_higher_classes
        self.average_signals = "time_channel" if average_signals is True else ("none" if average_signals is False else average_signals)
        self.apply_log = apply_log
        self.epsilon = epsilon
        self.time_offset_seconds = time_offset_seconds
        self.use_mid_target = use_mid_target
        self.sample_seconds = sample_seconds
        self.center_truth = center_truth

        # Sort the triplets chronologically before applying any offset and track the original indices
        self._sort_triplets()

        # Apply the time offset correction if necessary
        if time_offset_seconds is not None:
            self._apply_time_offset()

    # @time_it
    def _sort_triplets(self):
        """Sort the triplets (X, y, dt, ships) chronologically by dt and store the sorted values."""
        if self.ships is not None:
            triplets = sorted(zip(self.X, self.y, self.dt, self.ships), key=lambda t: t[2])
            X_sorted, y_sorted, dt_sorted, ships_sorted = zip(*triplets)
            self.X, self.y, self.dt, self.ships = list(X_sorted), list(y_sorted), list(dt_sorted), list(ships_sorted)
        else:
            triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
            X_sorted, y_sorted, dt_sorted = zip(*triplets)
            self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

    def _apply_time_offset(self):
        """Correct labels using the nearest target to dt + time_offset_seconds within a 5-second tolerance."""
        if self.time_offset_seconds is None:
            return

        # Time tolerance in seconds
        time_tolerance = timedelta(seconds=5)

        # Lists to hold the valid reduced triplets
        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []

        # Iterate through the sorted datetimes (dt)
        for i, current_dt in enumerate(self.dt):
            # Calculate the target time
            target_dt = current_dt + timedelta(seconds=self.time_offset_seconds)

            matched_index = _nearest_datetime_index(
                self.dt, target_dt, time_tolerance
            )
            if matched_index is not None:
                reduced_X.append(self.X[i])
                reduced_y.append(self.y[matched_index])
                reduced_dt.append(target_dt)
                if self.ships:
                    reduced_ships.append(self.ships[i])

        self.X = np.array(reduced_X)
        self.y = np.array(reduced_y)
        self.dt = reduced_dt
        if self.ships:
            self.ships = reduced_ships

    # @time_it
    def _apply_averaging(self, group_X):
        """Apply the specified averaging method to group_X."""
        if self.average_signals == "none":
            # Concatenate samples across time without averaging
            return np.concatenate(group_X, axis=1)
        elif self.average_signals == "time":
            # Average across time, keeping channels intact
            return np.mean(group_X, axis=0)
        elif self.average_signals == "channel":
            # Average across channels for each sample, then concatenate over time
            # Resulting in a 1D array of shape (n_samples_per_group * n_columns,)
            return np.array([np.mean(x, axis=0) for x in group_X]).flatten()
        elif self.average_signals == "time_channel":
            # Average across both time and channels
            return np.mean(group_X, axis=(0, 1))
        else:
            raise ValueError("Invalid value for average_signals. Expected 'none', 'time', 'channel', or 'time_channel'.")

    # @time_it
    def reduce_triplets(self):
        """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
        n_samples_per_group = self.n_seconds // self.sample_seconds
        # n_overlap_samples = (self.n_overlapping_seconds // self.sample_seconds) if self.n_overlapping_seconds else 0

        # Handle negative n_overlapping_seconds
        if self.n_overlapping_seconds is not None:
            if self.n_overlapping_seconds < 0:
                n_overlap_samples = n_samples_per_group + (self.n_overlapping_seconds // self.sample_seconds)
            else:
                n_overlap_samples = self.n_overlapping_seconds // self.sample_seconds
        else:
            n_overlap_samples = 0

        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
        i = 0

        while i < len(self.X):
            group_X = self.X[i:i + n_samples_per_group]
            group_y = self.y[i:i + n_samples_per_group]
            group_dt = self.dt[i:i + n_samples_per_group]
            group_ships = self.ships[i:i + n_samples_per_group] if self.ships else []

            if len(group_X) < n_samples_per_group:
                break  # Skip incomplete groups at the end

            # Apply logarithmic transformation to each sample in group_X if needed
            if self.apply_log:
                group_X = [np.log(np.maximum(sample, self.epsilon)) for sample in group_X]

            # Apply averaging method based on the selected option
            avg_X = self._apply_averaging(group_X)


            if self.join_higher_classes:
                group_y = np.clip(group_y, 0, 1)

            if self.use_mid_target:
                if self.center_truth:
                    mid_index = len(group_y) // 2
                    min_y = group_y[mid_index]
                else:
                    min_y = np.bincount(group_y).argmax()
            else:
               min_y = min(group_y)




            # Take the oldest datetime
            oldest_dt = min(group_dt)

            reduced_X.append(avg_X)
            reduced_y.append(min_y)
            reduced_dt.append(oldest_dt)

            if self.ships:
                closest_list = min(group_ships, key=lambda ship_list: min(ship['distance'] for ship in ship_list))
                reduced_ships.append(closest_list)

            i += n_samples_per_group - n_overlap_samples


        if self.ships:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
        return np.array(reduced_X), np.array(reduced_y), reduced_dt



class TripletRegressionReducer:
    def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None,
                 average_signals="none", apply_log=True, epsilon=1e-23,
                 time_offset_seconds=7200,
                 threshold=None, eliminate_within_range=None,
                 use_mid_target=False,
                 regression_target_method=None,
                 sample_seconds=10):
        """
        Initialize with the data to be reduced and the parameters.
        - X: List of 2D numpy arrays.
        - y: List of continuous values (for regression).
        - dt: List of utc-aware datetime objects.
        - ships: Optional ship data to reduce, list of lists (each sublist contains dictionaries with ship data).
        - n_seconds: Total number of seconds to average.
        - n_overlapping_seconds: If specified, overlap between groups in seconds.
        - average_signals: Averaging method ("none", "time", "channel", "time_channel").
        - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
        - epsilon: Small value added to X before applying log to avoid NaNs.
        - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
        - threshold: If provided, any y value greater than this threshold will be filtered out along with its corresponding X and dt.
        - eliminate_within_range: If provided, a tuple (low, high) to eliminate y values within that range.
        - use_mid_target: If True, selects the most central value as the target.
        """
        self.X = X
        self.y = y
        self.dt = dt
        self.ships = ships  # Ship data to be reduced if provided
        self.n_seconds = n_seconds
        self.n_overlapping_seconds = n_overlapping_seconds
        self.average_signals = "time_channel" if average_signals is True else ("none" if average_signals is False else average_signals)
        self.apply_log = apply_log
        self.epsilon = epsilon
        self.time_offset_seconds = time_offset_seconds
        self.threshold = threshold
        self.eliminate_within_range = eliminate_within_range
        self.use_mid_target = use_mid_target
        if regression_target_method is None:
            regression_target_method = "legacy" if use_mid_target else "min"
        valid_target_methods = {
            "legacy", "central_t", "first_t", "last_t", "min", "mean", "median"
        }
        if regression_target_method not in valid_target_methods:
            raise ValueError(
                "Invalid regression_target_method. Expected one of: "
                + ", ".join(sorted(valid_target_methods))
            )
        self.regression_target_method = regression_target_method
        self.sample_seconds = sample_seconds

        # Sort the triplets chronologically before applying any offset
        self._sort_triplets()

        # Apply the time offset correction if necessary
        if time_offset_seconds is not None:
            self._apply_time_offset()

        # Apply the threshold filtering if necessary
        if threshold is not None or eliminate_within_range is not None:
            self._apply_threshold()

        # Remove triplets where y is None
        self._remove_none_targets()

    # @time_it
    def _sort_triplets(self):
        """Sort the triplets (X, y, dt, ships) chronologically by dt and store the sorted values."""
        if self.ships is not None:
            triplets = sorted(zip(self.X, self.y, self.dt, self.ships), key=lambda t: t[2])
            X_sorted, y_sorted, dt_sorted, ships_sorted = zip(*triplets)
            self.X, self.y, self.dt, self.ships = list(X_sorted), list(y_sorted), list(dt_sorted), list(ships_sorted)
        else:
            triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
            X_sorted, y_sorted, dt_sorted = zip(*triplets)
            self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

    # @time_it
    def _apply_time_offset(self):
        """Correct labels using the nearest target to dt + time_offset_seconds within a 5-second tolerance."""
        if self.time_offset_seconds is None:
            return

        # Time tolerance in seconds
        time_tolerance = timedelta(seconds=5)

        # Lists to hold the valid reduced triplets
        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []

        # Iterate through the sorted datetimes (dt)
        for i, current_dt in enumerate(self.dt):
            # Calculate the target time
            target_dt = current_dt + timedelta(seconds=self.time_offset_seconds)

            matched_index = _nearest_datetime_index(
                self.dt, target_dt, time_tolerance
            )
            if matched_index is not None:
                reduced_X.append(self.X[i])
                reduced_y.append(self.y[matched_index])
                reduced_dt.append(target_dt)
                if self.ships:
                    reduced_ships.append(self.ships[i])

        self.X = np.array(reduced_X)
        self.y = np.array(reduced_y)
        self.dt = reduced_dt
        if self.ships:
            self.ships = reduced_ships

    # @time_it
    def _apply_threshold(self):
        """Remove X, y, dt, and ships entries where y is outside the specified threshold range."""
        valid_indices = []

        if self.threshold is not None:
            if isinstance(self.threshold, (tuple, list)) and len(self.threshold) == 2:
                low, high = self.threshold
                valid_indices = [
                    i for i, val in enumerate(self.y)
                    if val is not None and low <= val <= high
                ]
            else:
                valid_indices = [
                    i for i, val in enumerate(self.y)
                    if val is not None and val <= self.threshold
                ]

        if self.eliminate_within_range is not None:
            low, high = self.eliminate_within_range
            valid_indices = [
                i for i, val in enumerate(self.y)
                if val is not None and not (low <= val <= high)
            ]

        if valid_indices:
            self.X = np.array([self.X[i] for i in valid_indices])
            self.y = np.array([self.y[i] for i in valid_indices])
            self.dt = np.array([self.dt[i] for i in valid_indices])
            if self.ships is not None:
                self.ships = [self.ships[i] for i in valid_indices]

    def _remove_none_targets(self):
        """Remove triplets where y is None."""
        valid_indices = [i for i, val in enumerate(self.y) if val is not None]
        self.X = [self.X[i] for i in valid_indices]
        self.y = [self.y[i] for i in valid_indices]
        self.dt = [self.dt[i] for i in valid_indices]
        if self.ships:
            self.ships = [self.ships[i] for i in valid_indices]

    # @time_it
    def _apply_averaging(self, group_X):
        """Apply the specified averaging method to group_X."""
        if self.average_signals == "none":
            return np.concatenate(group_X, axis=1)
        elif self.average_signals == "time":
            return np.mean(group_X, axis=0)
        elif self.average_signals == "channel":
            return np.array([np.mean(x, axis=0) for x in group_X]).flatten()
        elif self.average_signals == "time_channel":
            return np.mean(group_X, axis=(0, 1))
        else:
            raise ValueError("Invalid value for average_signals. Expected 'none', 'time', 'channel', or 'time_channel'.")

    # @time_it
    def _select_target_and_datetime(self, group_y, group_dt):
        """Return the configured regression target and its representative time."""
        method = self.regression_target_method
        middle = len(group_y) // 2

        if method == "legacy":
            return group_y[middle], min(group_dt)
        if method == "central_t":
            if len(group_y) % 2:
                return group_y[middle], group_dt[middle]
            target = np.mean([group_y[middle - 1], group_y[middle]])
            timestamp = group_dt[middle - 1] + (
                group_dt[middle] - group_dt[middle - 1]
            ) / 2
            return target, timestamp
        if method == "first_t":
            return group_y[0], group_dt[0]
        if method == "last_t":
            return group_y[-1], group_dt[-1]
        if method == "min":
            target_index = int(np.argmin(group_y))
            return group_y[target_index], group_dt[target_index]

        timestamp = group_dt[0] + (group_dt[-1] - group_dt[0]) / 2
        if method == "mean":
            return np.mean(group_y), timestamp
        return np.median(group_y), timestamp

    def reduce_triplets(self):
        """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
        n_samples_per_group = self.n_seconds // self.sample_seconds
        # n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

        # Handle negative n_overlapping_seconds
        if self.n_overlapping_seconds is not None:
            if self.n_overlapping_seconds < 0:
                n_overlap_samples = n_samples_per_group + (self.n_overlapping_seconds // self.sample_seconds)
            else:
                n_overlap_samples = self.n_overlapping_seconds // self.sample_seconds
        else:
            n_overlap_samples = 0

        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
        i = 0
        while i < len(self.X):
            group_X = self.X[i:i + n_samples_per_group]
            group_y = self.y[i:i + n_samples_per_group]
            group_dt = self.dt[i:i + n_samples_per_group]
            group_ships = self.ships[i:i + n_samples_per_group] if self.ships else None

            if len(group_X) < n_samples_per_group:
                break  # Skip incomplete groups at the end

            # Apply logarithmic transformation to each sample in group_X if needed
            if self.apply_log:
                group_X = [np.log(np.maximum(sample, self.epsilon)) for sample in group_X]

            # Apply averaging method based on the selected option
            avg_X = self._apply_averaging(group_X)

            # # Apply logarithmic transformation if needed, ensuring no NaNs
            # if self.apply_log:
            #     avg_X = np.log(np.maximum(avg_X, self.epsilon))


            target_y, target_dt = self._select_target_and_datetime(
                group_y, group_dt
            )

            if group_ships:
                flattened_ships = [ship for sublist in group_ships for ship in sublist]
                min_distance_ship = min(flattened_ships, key=lambda s: s.get('distance', float('inf')))
                reduced_ships.append([min_distance_ship])

            reduced_X.append(avg_X)
            reduced_y.append(target_y)
            reduced_dt.append(target_dt)

            i += n_samples_per_group - n_overlap_samples

        if self.ships:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
        else:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt

# class TripletRegressionReducer:
#     def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None,
#                  average_signals=False, apply_log=True, epsilon=1e-10,
#                  time_offset_seconds=7200, target_method='average',
#                  threshold=None, eliminate_within_range=None):
#         """
#         Initialize with the data to be reduced and the parameters.
#         - X: List of 2D numpy arrays.
#         - y: List of continuous values (for regression).
#         - dt: List of utc-aware datetime objects.
#         - ships: Optional ship data to reduce, list of lists (each sublist contains dictionaries with ship data).
#         - n_seconds: Total number of seconds to average.
#         - n_overlapping_seconds: If specified, overlap between groups in seconds.
#         - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
#         - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
#         - epsilon: Small value added to X before applying log to avoid NaNs.
#         - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
#         - target_method: Method to determine the target ('minimum', 'average', 'median').
#         - threshold: If provided, any y value greater than this threshold will be filtered out along with its corresponding X and dt.
#         - eliminate_within_range: If provided, a tuple (low, high) to eliminate y values within that range.
#         """
#         self.X = X
#         self.y = y
#         self.dt = dt
#         self.ships = ships  # Ship data to be reduced if provided
#         self.n_seconds = n_seconds
#         self.n_overlapping_seconds = n_overlapping_seconds
#         self.average_signals = average_signals
#         self.apply_log = apply_log
#         self.epsilon = epsilon
#         self.time_offset_seconds = time_offset_seconds
#         self.target_method = target_method.lower()
#         self.threshold = threshold
#         self.eliminate_within_range = eliminate_within_range

#         # Sort the triplets chronologically before applying any offset
#         self._sort_triplets()

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

#         # Apply the threshold filtering if necessary
#         if threshold is not None or eliminate_within_range is not None:
#             self._apply_threshold()

#     def _sort_triplets(self):
#         """Sort the triplets (X, y, dt, ships) chronologically by dt and store the sorted values."""
#         if self.ships is not None:
#             triplets = sorted(zip(self.X, self.y, self.dt, self.ships), key=lambda t: t[2])
#             X_sorted, y_sorted, dt_sorted, ships_sorted = zip(*triplets)
#             self.X, self.y, self.dt, self.ships = list(X_sorted), list(y_sorted), list(dt_sorted), list(ships_sorted)
#         else:
#             triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
#             X_sorted, y_sorted, dt_sorted = zip(*triplets)
#             self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

#     def _apply_time_offset(self):
#         """Correct labels using the nearest target to dt + time_offset_seconds within a 5-second tolerance."""
#         if self.time_offset_seconds is None:
#             return

#         # Time tolerance in seconds
#         time_tolerance = timedelta(seconds=5)

#         # Lists to hold the valid reduced triplets
#         reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []

#         # Create a mapping from dt to index for quick lookup using binary search
#         for i, current_dt in enumerate(self.dt):
#             # Calculate the target time
#             target_dt = current_dt + timedelta(seconds=self.time_offset_seconds)

#             # Use binary search to find the closest index where target_dt could be inserted
#             insert_position = bisect_left(self.dt, target_dt)

#             # Check the nearest neighbors within the tolerance
#             if insert_position < len(self.dt):
#                 # Check the candidate at the insert position
#                 if abs(self.dt[insert_position] - target_dt) <= time_tolerance:
#                     reduced_X.append(self.X[i])  # Keep the current X
#                     reduced_y.append(self.y[insert_position])  # Use the corresponding y at dt + offset
#                     reduced_dt.append(target_dt)  # Keep the target_dt
#                     if self.ships:
#                         reduced_ships.append(self.ships[i])
#                 # Check the candidate just before the insert position
#                 elif insert_position > 0 and abs(self.dt[insert_position - 1] - target_dt) <= time_tolerance:
#                     reduced_X.append(self.X[i])  # Keep the current X
#                     reduced_y.append(self.y[insert_position - 1])  # Use the corresponding y at dt + offset
#                     reduced_dt.append(current_dt)  # Keep the current dt (not the shifted one)
#                     if self.ships:
#                         reduced_ships.append(self.ships[i])

#         # Replace the original X, y, dt, ships with the reduced ones
#         self.X = np.array(reduced_X)
#         self.y = np.array(reduced_y)
#         self.dt = reduced_dt
#         if self.ships:
#             self.ships = reduced_ships

#     def _apply_threshold(self):
#         """Remove X, y, dt, and ships entries where y is greater than the threshold or within the range."""
#         valid_indices = []

#         if self.threshold is not None:
#             # Create a mask for entries where y is less than or equal to the threshold
#             valid_indices = [i for i, val in enumerate(self.y) if val <= self.threshold]

#         if self.eliminate_within_range is not None:
#             low, high = self.eliminate_within_range
#             # Extend the valid_indices to exclude y within the specified range
#             valid_indices = [i for i, val in enumerate(self.y) if not (low <= val <= high)]

#         # Filter X, y, dt, and ships based on the valid indices
#         if valid_indices:
#             self.X = np.array([self.X[i] for i in valid_indices])
#             self.y = np.array([self.y[i] for i in valid_indices])
#             self.dt = np.array([self.dt[i] for i in valid_indices])
#             if self.ships is not None:
#                 self.ships = [self.ships[i] for i in valid_indices]

#     def reduce_triplets(self):
#         """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
#         i = 0
#         while i < len(self.X):
#             # Define the current group range
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
#             group_ships = self.ships[i:i + n_samples_per_group] if self.ships else None

#             if len(group_X) < n_samples_per_group:
#                 break  # Skip incomplete groups at the end

#             # Average the Xs
#             avg_X = np.mean(group_X, axis=0)

#             # Optionally average across all rows (producing a 1-D array)
#             if self.average_signals:
#                 avg_X = np.mean(avg_X, axis=0)

#             # Apply logarithmic transformation if needed, ensuring no NaNs
#             if self.apply_log:
#                 avg_X = np.log(np.maximum(avg_X, self.epsilon))

#             # Determine the y value (target) based on the target method
#             if self.target_method == 'minimum':
#                 target_y = np.min(group_y)
#             elif self.target_method == 'median':
#                 target_y = np.median(group_y)
#             else:
#                 target_y = np.mean(group_y)  # Default is 'average'

#             # Take the oldest datetime
#             oldest_dt = min(group_dt)

#             # Reduce ship data if present
#             if group_ships:
#                 # Flatten the list of lists of ships
#                 flattened_ships = [ship for sublist in group_ships for ship in sublist]

#                 # Find the ship with the minimum distance
#                 min_distance_ship = min(flattened_ships, key=lambda s: s.get('distance', float('inf')))

#                 # Append the entire list of ships containing the closest ship
#                 reduced_ships.append([min_distance_ship])

#             # Store the reduced triplet
#             reduced_X.append(avg_X)
#             reduced_y.append(target_y)
#             reduced_dt.append(oldest_dt)

#             # Move the index by the group size minus overlap
#             i += n_samples_per_group - n_overlap_samples

#         if self.ships:
#             return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
#         else:
#             return np.array(reduced_X), np.array(reduced_y), reduced_dt



class DataSplitter:
    def __init__(self, X, y, dt):
        self.X = X
        self.y = y
        self.dt = dt

    def _sort_data(self):
        triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
        X_sorted, y_sorted, dt_sorted = zip(*triplets)
        return list(X_sorted), list(y_sorted), list(dt_sorted)

    def _balance_test_set(self, X_train, X_test, y_train, y_test, dt_train, dt_test):
        test_class_counts = Counter(y_test)
        min_class_count = min(test_class_counts.values())
        X_test_min, X_test_maj = [], []
        y_test_min, y_test_maj = [], []
        dt_test_min, dt_test_maj = [], []
        for X_val, y_val, dt_val in zip(X_test, y_test, dt_test):
            if test_class_counts[y_val] == min_class_count:
                X_test_min.append(X_val)
                y_test_min.append(y_val)
                dt_test_min.append(dt_val)
            else:
                X_test_maj.append(X_val)
                y_test_maj.append(y_val)
                dt_test_maj.append(dt_val)
        X_test_min, X_test_maj = np.array(X_test_min), np.array(X_test_maj)
        y_test_min, y_test_maj = np.array(y_test_min), np.array(y_test_maj)
        dt_test_min, dt_test_maj = np.array(dt_test_min), np.array(dt_test_maj)
        excess = len(X_test_maj) - min_class_count
        if excess > 0:
            X_train = np.concatenate([X_train, X_test_maj[:excess]])
            y_train = np.concatenate([y_train, y_test_maj[:excess]])
            dt_train = np.concatenate([dt_train, dt_test_maj[:excess]])
            X_test_maj, y_test_maj, dt_test_maj = X_test_maj[excess:], y_test_maj[excess:], dt_test_maj[excess:]
        X_test_balanced = np.concatenate([X_test_min, X_test_maj])
        y_test_balanced = np.concatenate([y_test_min, y_test_maj])
        dt_test_balanced = np.concatenate([dt_test_min, dt_test_maj])
        return np.array(X_train), X_test_balanced, np.array(y_train), y_test_balanced, dt_train, dt_test_balanced

    def split_by_sklearn(self, test_size=0.2, random_state=None, shuffle=True, balance_test=False, **kwargs):
        X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
            self.X, self.y, self.dt, test_size=test_size, random_state=random_state, shuffle=shuffle, **kwargs
        )
        if balance_test:
            return self._balance_test_set(X_train, X_test, y_train, y_test, dt_train, dt_test)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test


    def split_by_day(self, test_day, balance_test=False):
        """
        Split the data such that the test set contains all data from the specified day.
        - test_day: A datetime.date object specifying which day's data should go into the test set.
        - balance_test: If True, balance the test set by undersampling the majority class.
        """
        X_sorted, y_sorted, dt_sorted = self._sort_data()

        X_train, X_test, y_train, y_test, dt_train, dt_test = [], [], [], [], [], []

        for X_val, y_val, dt_val in zip(X_sorted, y_sorted, dt_sorted):
            if dt_val.date() == test_day:
                X_test.append(X_val)
                y_test.append(y_val)
                dt_test.append(dt_val)
            else:
                X_train.append(X_val)
                y_train.append(y_val)
                dt_train.append(dt_val)

        if balance_test:
            X_train, X_test, y_train, y_test, dt_train, dt_test = self._balance_test_set(X_train, X_test, y_train, y_test, dt_train, dt_test)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(dt_train), np.array(dt_test)

    def split_by_index_range(self, index_1, index_2, balance_test=False):
        """
        Split the data by a specific range of indexes, leaving that range in the test set.
        The data will be sorted chronologically before splitting.
        - index_1: Start index for the test set.
        - index_2: End index (inclusive) for the test set.
        - balance_test: If True, balance the test set by undersampling the majority class.
        """
        X_sorted, y_sorted, dt_sorted = self._sort_data()

        X_test = X_sorted[index_1:index_2+1]
        y_test = y_sorted[index_1:index_2+1]
        dt_test = dt_sorted[index_1:index_2+1]

        X_train = X_sorted[:index_1] + X_sorted[index_2+1:]
        y_train = y_sorted[:index_1] + y_sorted[index_2+1:]
        dt_train = dt_sorted[:index_1] + dt_sorted[index_2+1:]

        if balance_test:
            X_train, X_test, y_train, y_test, dt_train, dt_test = self._balance_test_set(X_train, X_test, y_train, y_test, dt_train, dt_test)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test

    def _get_time_interval(self, dt, interval_start, interval_end):
        """Check if the datetime object dt falls within the given time interval."""
        return interval_start <= dt.time() <= interval_end

    def _balance_classes_in_test(self, X_train, X_test, y_train, y_test, dt_train, dt_test):
        """Balance the classes in the test set, ensuring both class 0 and class 1 are present."""
        class_0_indices = [i for i, y in enumerate(y_test) if y == 0]
        class_1_indices = [i for i, y in enumerate(y_test) if y == 1]

        # If there are no instances of one of the classes, move some samples from the train set
        if not class_0_indices or not class_1_indices:
            required_class = 0 if not class_0_indices else 1
            train_class_indices = [i for i, y in enumerate(y_train) if y == required_class]

            # Move enough samples to the test set from the train set to balance
            for i in range(min(len(train_class_indices), len(class_0_indices) + len(class_1_indices))):
                X_test.append(X_train[train_class_indices[i]])
                y_test.append(y_train[train_class_indices[i]])
                dt_test.append(dt_train[train_class_indices[i]])

                # Remove from train set
                X_train.pop(train_class_indices[i])
                y_train.pop(train_class_indices[i])
                dt_train.pop(train_class_indices[i])

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test

    def split_by_time_interval(self, interval_start, interval_end, test_size=0.3, balance_test=True):
        """
        Split data by time intervals within each day to ensure a more reliable test set.

        - interval_start: A datetime.time object representing the start of the interval (e.g., time(8, 0) for 8 AM).
        - interval_end: A datetime.time object representing the end of the interval (e.g., time(16, 0) for 4 PM).
        - test_size: Proportion of the data to include in the test set.
        - balance_test: If True, ensure the test set has instances from both classes.
        """
        X_sorted, y_sorted, dt_sorted = self._sort_data()

        X_train, X_test, y_train, y_test, dt_train, dt_test = [], [], [], [], [], []

        # Iterate over sorted data and assign to test if within the interval
        for X_val, y_val, dt_val in zip(X_sorted, y_sorted, dt_sorted):
            if self._get_time_interval(dt_val, interval_start, interval_end):
                X_test.append(X_val)
                y_test.append(y_val)
                dt_test.append(dt_val)
            else:
                X_train.append(X_val)
                y_train.append(y_val)
                dt_train.append(dt_val)

        # Adjust the test size if necessary (undersample if test set is too large)
        if len(X_test) > len(X_sorted) * test_size:
            X_test, y_test, dt_test = X_test[:int(len(X_sorted) * test_size)], y_test[:int(len(X_sorted) * test_size)], dt_test[:int(len(X_sorted) * test_size)]

        if balance_test:
            return self._balance_classes_in_test(X_train, X_test, y_train, y_test, dt_train, dt_test)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test


class DataRegressionSplitter:
    def __init__(self, X, y, dt):
        self.X = X
        self.y = y
        self.dt = dt

    def _sort_data(self):
        triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
        X_sorted, y_sorted, dt_sorted = zip(*triplets)
        return list(X_sorted), list(y_sorted), list(dt_sorted)

    def split_by_sklearn(self, test_size=0.2, random_state=None, shuffle=True):
        """Split the data using sklearn's train_test_split for regression."""
        X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
            self.X, self.y, self.dt, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test

    def split_by_day(self, test_day):
        """Split the data by leaving a specific day in the test set."""
        X_sorted, y_sorted, dt_sorted = self._sort_data()

        X_train, X_test, y_train, y_test, dt_train, dt_test = [], [], [], [], [], []
        for X_val, y_val, dt_val in zip(X_sorted, y_sorted, dt_sorted):
            if dt_val.date() == test_day:
                X_test.append(X_val)
                y_test.append(y_val)
                dt_test.append(dt_val)
            else:
                X_train.append(X_val)
                y_train.append(y_val)
                dt_train.append(dt_val)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test

    def split_by_time_interval(self, interval_start, interval_end):
        """Split the data by leaving a specific time interval in the test set."""
        X_sorted, y_sorted, dt_sorted = self._sort_data()

        X_train, X_test, y_train, y_test, dt_train, dt_test = [], [], [], [], [], []
        for X_val, y_val, dt_val in zip(X_sorted, y_sorted, dt_sorted):
            if interval_start <= dt_val.time() <= interval_end:
                X_test.append(X_val)
                y_test.append(y_val)
                dt_test.append(dt_val)
            else:
                X_train.append(X_val)
                y_train.append(y_val)
                dt_train.append(dt_val)

        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), dt_train, dt_test



class DayBasedCV(BaseCrossValidator):
    def __init__(self, dt):
        self.dt = np.array([d.date() for d in dt])
        self.unique_days = np.unique(self.dt)

    def split(self, X, y=None, groups=None):
        for day in self.unique_days:
            test_idx = np.where(self.dt == day)[0]
            train_idx = np.where(self.dt != day)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.unique_days)

class CrossValidator:
    def __init__(self, X_train, y_train, dt_train):
        self.X_train = X_train
        self.y_train = y_train
        self.dt_train = dt_train

    def get_cv(self, cv=None, stratified=False):
        if cv is None:
            return StratifiedKFold(n_splits=5) if stratified else KFold(n_splits=5)
        elif isinstance(cv, int):
            return StratifiedKFold(n_splits=cv) if stratified else KFold(n_splits=cv)
        elif cv == 'day_based':
            return DayBasedCV(self.dt_train)
        else:
            raise ValueError("Invalid CV option. Use None, int, or 'day_based'.")


class RegressionCrossValidator:
    def __init__(self, X_train, y_train, dt_train):
        self.X_train = X_train
        self.y_train = y_train
        self.dt_train = dt_train

    def get_cv(self, cv=None):
        """
        Return the cross-validation splitter.
        - cv: None (default 5-fold), int (number of folds), or 'day_based' for day-based splitting.
        """
        if cv is None:
            return KFold(n_splits=5)  # Default 5-fold cross-validation
        elif isinstance(cv, int):
            return KFold(n_splits=cv)  # Custom number of folds
        elif cv == 'day_based':
            return DayBasedCV(self.dt_train)  # Day-based cross-validation
        else:
            raise ValueError("Invalid CV option. Use None, int, or 'day_based'.")


class ModelEvaluator:
    """
    Evaluates a classification model and optionally computes per-day bootstrap uncertainty.
    Includes bootstrap resampling to estimate uncertainty on daily metrics.
    """

    def __init__(
        self, model, X_train, X_test, y_train, y_test, cv,
        instance_window: Optional[int] = None, dt_train=None, dt_test=None,
        compute_daywise_bootstrap: bool = False, n_bootstrap: int = 1000,
        freq_limit_joblib=None, center_ground_truth: bool = False, include_predictions: bool = True
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.instance_window = instance_window
        self.dt_train = dt_train
        self.dt_test = dt_test
        self.compute_daywise_bootstrap = compute_daywise_bootstrap
        self.n_bootstrap = n_bootstrap
        self.freq_limit_joblib = freq_limit_joblib
        self.center_ground_truth = center_ground_truth
        self.include_predictions = include_predictions

        if self.instance_window and self.instance_window > 1:
            self._sort_data()

    def _sort_data(self):
        """Sort training and test data by datetime if available."""
        if self.dt_train is not None:
            sorted_indices_train = np.argsort(self.dt_train)
            self.X_train = self.X_train[sorted_indices_train]
            self.y_train = self.y_train[sorted_indices_train]
            self.dt_train = self.dt_train[sorted_indices_train]

        if self.dt_test is not None:
            sorted_indices_test = np.argsort(self.dt_test)
            self.X_test = self.X_test[sorted_indices_test]
            self.y_test = self.y_test[sorted_indices_test]

    def evaluate_on_test_set(self):
        """
        Evaluates the model with optional uncertainty measures.
        If instance_window is set, applies majority voting across overlapping windows.
        """


        # Generate predictions normally
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]


        if self.instance_window and self.instance_window > 1:
            # Apply majority voting over instance_window-sized overlapping groups
            y_pred_windowed, y_test_windowed = self._apply_majority_voting(self.y_test, y_pred)
        else:
            y_pred_windowed, y_test_windowed = y_pred, self.y_test

        min_len = min(len(y_pred_windowed), len(y_test_windowed))
        y_pred_windowed = y_pred_windowed[:min_len]
        y_test_windowed = y_test_windowed[:min_len]
        y_pred_proba = y_pred_proba[:min_len]

        # Debug: log unique classes in y_test_windowed to diagnose potential ROC AUC errors
        unique_classes, class_counts = np.unique(y_test_windowed, return_counts=True)
        logger.info(
            "evaluate_on_test_set: unique classes and counts: %s",
            dict(zip(unique_classes, class_counts)),
        )

        # Compute base metrics
        accuracy = accuracy_score(y_test_windowed, y_pred_windowed)
        auc = roc_auc_score(y_test_windowed, y_pred_proba[:len(y_test_windowed)])  # Align length after windowing
        report = classification_report(y_test_windowed, y_pred_windowed, output_dict=True)
        confusion = confusion_matrix(y_test_windowed, y_pred_windowed)

        results = {
            "accuracy": accuracy,
            "auc": auc,
            "classification_report": report,
            "confusion_matrix": confusion
        }

        # Compute uncertainty measures using bootstrap if enabled
        if self.compute_daywise_bootstrap:
            logger.info("Computing per-day bootstrap uncertainty...")
            results["uncertainty"] = {
                "accuracy": self._compute_bootstrap_uncertainty(y_test_windowed, y_pred, accuracy_score),
                "f1_macro": self._compute_bootstrap_uncertainty(
                    y_test_windowed, y_pred, lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
                ),
                "f1_weighted": self._compute_bootstrap_uncertainty(
                    y_test_windowed, y_pred, lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
                )
            }

        # Compute SHAP values if frequency limits are provided
        if self.freq_limit_joblib:
            from joblib import load
            # import shap
            freq_limit_dict = load(self.freq_limit_joblib)
            band_limits = freq_limit_dict['band_limits']
            feature_names = [f"{round(start, 2)}-{round(end, 2)}" for start, end in band_limits]
            # Initialize SHAP explainer with training data as background and assign feature names.
            # explainer = shap.Explainer(self.model, self.X_train, feature_names=feature_names)

            # Compute SHAP values on both test and training data
            # shap_values_test = explainer(self.X_test)
            # shap_values_train = explainer(self.X_train)

            # For training data
            shap_values_train = compute_shap_values(
                model=self.model,
                X_train=self.X_train,
                X_data=self.X_train,
                feature_names=feature_names,
                nsamples_background=100,    # Adjust based on dataset and required precision
                nsamples_explain=1000       # Use a subset if X_train is huge
            )

            # Similarly for test data if needed:
            shap_values_test = compute_shap_values(
                model=self.model,
                X_train=self.X_train,
                X_data=self.X_test,
                feature_names=feature_names,
                nsamples_background=100,
                nsamples_explain=500        # Adjust accordingly
            )


            # Save only the essential parts from the Explanation objects
            results["shap"] = {
                "test_values": shap_values_test.values,
                "test_data": shap_values_test.data,
                "train_values": shap_values_train.values,
                "train_data": shap_values_train.data,
                "feature_names": feature_names,
                "base_value_test": shap_values_test.base_values,
                "base_value_train": shap_values_train.base_values,
            }


        if self.include_predictions:
            results["y_true"] = y_test_windowed
            results["y_pred"] = y_pred_windowed
            if self.dt_test is not None:
                # Adjust dt_test length to match y_test_windowed
                if self.instance_window and self.instance_window > 1:
                    num_windows = len(self.y_test) - self.instance_window + 1
                    results["datetimes"] = self.dt_test[:num_windows]
                else:
                    results["datetimes"] = self.dt_test[:len(y_test_windowed)]




        return results

    # @time_it
    # def _apply_majority_voting(self, y_true, y_pred):
    #     """
    #     Applies majority voting across overlapping windows in the test set.
    #     - Each window contains `instance_window` overlapping elements.
    #     - Majority class from y_true forms the ground truth.
    #     - Majority class from y_pred forms the model prediction.

    #     Returns:
    #         y_pred_windowed, y_true_windowed (both with the same length)
    #     """
    #     y_true_windowed, y_pred_windowed = [], []

    #     # Ensure we have enough data points to form at least one full window
    #     if len(y_true) < self.instance_window or len(y_pred) < self.instance_window:
    #         logger.warning(
    #             "Not enough data for full instance_window=%d. Skipping majority voting.",
    #             self.instance_window
    #         )
    #         return np.array(y_pred), np.array(y_true)  # Return original arrays if windowing is not feasible

    #     num_windows = len(y_true) - self.instance_window + 1  # Overlapping sliding windows

    #     for i in range(num_windows):
    #         y_true_window = y_true[i:i + self.instance_window]
    #         y_pred_window = y_pred[i:i + self.instance_window]

    #         # Compute majority vote for each window
    #         y_true_windowed.append(self._get_majority_vote(y_true_window))
    #         y_pred_windowed.append(self._get_majority_vote(y_pred_window))

    #     # Convert to numpy arrays and ensure they have the same length
    #     y_true_windowed = np.array(y_true_windowed)
    #     y_pred_windowed = np.array(y_pred_windowed)

    #     # Ensure equal lengths (in case of any unexpected misalignment)
    #     min_len = min(len(y_true_windowed), len(y_pred_windowed))
    #     return y_pred_windowed[:min_len], y_true_windowed[:min_len]

    def _apply_majority_voting(self, y_true, y_pred):
        """
        Applies majority voting across overlapping windows in the test set.
        - Each window contains `instance_window` overlapping elements.
        - Majority class from y_true forms the ground truth.
        - Majority class from y_pred forms the model prediction.

        Returns:
            y_pred_windowed, y_true_windowed (both with the same length)
        """
        y_true_windowed, y_pred_windowed = [], []

        # Ensure we have enough data points to form at least one full window
        if len(y_true) < self.instance_window or len(y_pred) < self.instance_window:
            print(f"Warning: Not enough data for full instance_window={self.instance_window}. Skipping majority voting.")
            return np.array(y_pred), np.array(y_true)  # Return original arrays if windowing is not feasible

        num_windows = len(y_true) - self.instance_window + 1  # Overlapping sliding windows

        differences_count = 0
        total_windows = 0

        for i in range(num_windows):
            y_true_window = y_true[i:i + self.instance_window]
            y_pred_window = y_pred[i:i + self.instance_window]

            center_index = self.instance_window // 2
            center_value = y_true_window[center_index]
            majority_value = self._get_majority_vote(y_true_window)

            if center_value != majority_value:
                differences_count += 1

            total_windows += 1

            # Compute majority vote for each window
            if self.center_ground_truth:
                y_true_windowed.append(center_value)
            else:
                y_true_windowed.append(majority_value)
            y_pred_windowed.append(self._get_majority_vote(y_pred_window))

        if self.center_ground_truth:
            print(f"DEBUG: Total windows processed: {total_windows}")
            print(f"DEBUG: Number of windows where center value != majority vote: {differences_count}")

        # Convert to numpy arrays and ensure they have the same length
        y_true_windowed = np.array(y_true_windowed)
        y_pred_windowed = np.array(y_pred_windowed)

        # Ensure equal lengths (in case of any unexpected misalignment)
        min_len = min(len(y_true_windowed), len(y_pred_windowed))
        return y_pred_windowed[:min_len], y_true_windowed[:min_len]

    # @time_it
    def _get_majority_vote(self, values):
        """
        Returns the most frequent value in a NumPy array of integer labels.
        """
        values = np.array(values)  # Ensure it's a NumPy array
        return np.bincount(values).argmax()

    def _compute_bootstrap_uncertainty(self, y_true, y_pred, metric_func):
        """
        Computes bootstrap uncertainty for a given metric.
        """
        boot_results = []
        n_samples = len(y_true)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            if len(np.unique(y_true_boot)) < 2:
                continue
            boot_results.append(metric_func(y_true_boot, y_pred_boot))

        if not boot_results:
            return {"mean": 0.0, "std": 0.0, "lower_bound": 0.0, "upper_bound": 0.0}

        boot_results = np.array(boot_results)
        lower_bound = np.percentile(boot_results, 2.5)
        upper_bound = np.percentile(boot_results, 97.5)

        return {
            "mean": np.mean(boot_results),
            "std": np.std(boot_results),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }



class ModelRegressionEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test, cv,
                 y_threshold=None, instance_window=None, dt_train=None, dt_test=None,
                 print_results=False, compute_daywise_bootstrap: bool = False, freq_limit_joblib=None,
                 center_ground_truth: bool = False, include_predictions: bool = True,
                 regression_evaluation_threshold=None):
        """
        Initialize the evaluator with the model and data.
        - model: The regression model (e.g., XGBRegressor).
        - X_train, X_test, y_train, y_test: Training and test data.
        - cv: Cross-validation splitter.
        - y_threshold: Optional numeric value to filter y_test. Only evaluate on y_test < y_threshold.
        - instance_window: Optional window size to aggregate predictions.
        - dt_train, dt_test: Optional datetime arrays for sorting training and test data by time.
        - print_results: Whether to print the evaluation results.
        - compute_daywise_bootstrap: If True, compute per-day bootstrap uncertainty on regression metrics.
        - freq_limit_joblib: Path to joblib file containing frequency band limits for SHAP computation.
        - include_predictions: If True, include true y, predicted y, and datetimes in the final metrics.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.y_threshold = y_threshold
        self.instance_window = instance_window
        self.dt_train = np.array(dt_train) if dt_train is not None else None
        self.dt_test = np.array(dt_test) if dt_test is not None else None
        self.print_results = print_results
        self.compute_daywise_bootstrap = compute_daywise_bootstrap
        self.freq_limit_joblib = freq_limit_joblib
        self.center_ground_truth = center_ground_truth
        self.include_predictions = include_predictions
        self.regression_evaluation_threshold = regression_evaluation_threshold

        # Apply the y_threshold filter to the test data if provided
        if self.y_threshold is not None:
            self._apply_y_threshold_filter()

        # Sort the data by datetime if instance_window is used and datetime info is provided
        if self.instance_window and self.instance_window > 1:
            self._sort_by_datetime()

    def _apply_y_threshold_filter(self):
        """Filter X_test and y_test where y_test is less than the specified y_threshold."""
        filter_mask = self.y_test < self.y_threshold
        self.X_test = self.X_test[filter_mask]
        self.y_test = self.y_test[filter_mask]
        self.dt_test = self.dt_test[filter_mask]

    def _sort_by_datetime(self):
        """Sort the training and test data by their respective datetime objects."""
        if self.dt_train is not None:
            sorted_indices_train = np.argsort(self.dt_train)
            if len(sorted_indices_train) == len(self.X_train):
                self.X_train = self.X_train[sorted_indices_train]
                self.y_train = self.y_train[sorted_indices_train]
                self.dt_train = self.dt_train[sorted_indices_train]
            else:
                raise ValueError("Mismatch between the number of dt_train and X_train samples.")

        if self.dt_test is not None:
            sorted_indices_test = np.argsort(self.dt_test)
            if len(sorted_indices_test) == len(self.X_test):
                self.X_test = self.X_test[sorted_indices_test]
                self.y_test = self.y_test[sorted_indices_test]
                self.dt_test = self.dt_test[sorted_indices_test]
            else:
                raise ValueError("Mismatch between the number of dt_test and X_test samples.")

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set and compute additional metrics.
        Instead of doing a bootstrap within this method, we now compute and save
        fold-level summary statistics for:
          - error (residual)
          - squared error
          - absolute error
        These summaries can later be aggregated across folds to estimate overall
        confidence intervals.
        """


        # Predict on test set
        y_pred = self.model.predict(self.X_test)

        if self.instance_window and self.instance_window > 1:
            # Apply mean aggregation across instance_window-sized overlapping groups
            y_pred_windowed, y_test_windowed = self._apply_mean_grouping(self.y_test, y_pred)
        else:
            y_pred_windowed, y_test_windowed = y_pred, self.y_test

        # Compute base regression metrics
        mse = mean_squared_error(y_test_windowed, y_pred_windowed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_windowed, y_pred_windowed)
        r2 = r2_score(y_test_windowed, y_pred_windowed)

        # Additional metrics
        residuals = y_test_windowed - y_pred_windowed
        mae_std = np.std(np.abs(residuals))
        rmse_std = np.std(residuals ** 2)
        t_stat_mae, two_sided_p_value_mae = ttest_1samp(np.abs(residuals), popmean=0)
        t_stat_rmse, two_sided_p_value_rmse = ttest_1samp(residuals ** 2, popmean=0)

        # One-sided t-tests for MAE
        p_value_mae_less = two_sided_p_value_mae / 2 if t_stat_mae < 0 else 1 - (two_sided_p_value_mae / 2)
        p_value_mae_greater = 1 - p_value_mae_less
        # One-sided t-tests for RMSE
        p_value_rmse_less = two_sided_p_value_rmse / 2 if t_stat_rmse < 0 else 1 - (two_sided_p_value_rmse / 2)
        p_value_rmse_greater = 1 - p_value_rmse_less

        # Normality and shape metrics for residuals. SciPy documents the
        # Shapiro p-value as potentially inaccurate above 5,000 observations.
        # Preserve the result keys but skip the calculation in that case.
        global _shapiro_large_sample_warning_emitted
        if len(residuals) > 5000:
            shapiro_stat, shapiro_p_value = np.nan, np.nan
            if not _shapiro_large_sample_warning_emitted:
                logger.warning(
                    "Skipping Shapiro-Wilk residual normality test because "
                    "the residual count (%d) exceeds 5000; SciPy documents "
                    "the p-value as potentially inaccurate for larger samples.",
                    len(residuals),
                )
                _shapiro_large_sample_warning_emitted = True
        else:
            shapiro_stat, shapiro_p_value = shapiro(residuals)
        residual_skewness = skew(residuals)
        residual_kurtosis = kurtosis(residuals)

        # -------------------------------
        # Instead of bootstrapping all errors here,
        # compute fold-level summary statistics for later aggregation.
        # -------------------------------
        fold_summary = {
            'mean_error': np.mean(residuals),
            'var_error': np.var(residuals, ddof=1),
            'mean_squared_error': np.mean(residuals ** 2),
            'var_squared_error': np.var(residuals ** 2, ddof=1),
            'mean_absolute_error': np.mean(np.abs(residuals)),
            'var_absolute_error': np.var(np.abs(residuals), ddof=1),
            'n': len(residuals)
        }

        # -------------------------------
        # Aggregate all metrics in a dictionary
        # -------------------------------
        metrics = {
            "MEAN_TARGET": np.mean(y_test_windowed),
            "SUPPORT": len(y_test_windowed),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAE_STD": mae_std,
            "RMSE_STD": rmse_std,
            "T_STAT_MAE": t_stat_mae,
            "P_VALUE_MAE_TWO_SIDED": two_sided_p_value_mae,
            "P_VALUE_MAE_LESS": p_value_mae_less,
            "P_VALUE_MAE_GREATER": p_value_mae_greater,
            "T_STAT_RMSE": t_stat_rmse,
            "P_VALUE_RMSE_TWO_SIDED": two_sided_p_value_rmse,
            "P_VALUE_RMSE_LESS": p_value_rmse_less,
            "P_VALUE_RMSE_GREATER": p_value_rmse_greater,
            "SHAPIRO_STAT": shapiro_stat,
            "SHAPIRO_P_VALUE": shapiro_p_value,
            "RESIDUAL_SKEWNESS": residual_skewness,
            "RESIDUAL_KURTOSIS": residual_kurtosis,
            # Save fold-level summary statistics for later aggregation
            "fold_summary": fold_summary
        }
        # also keep fold residuals for bootstrap-based CI
        metrics['residuals'] = residuals

        # This optional view is deliberately computed from the already-generated
        # predictions, so it cannot change training or the existing overall metrics.
        if self.regression_evaluation_threshold is not None:
            threshold = float(self.regression_evaluation_threshold)
            threshold_mask = y_test_windowed <= threshold
            threshold_y_true = y_test_windowed[threshold_mask]
            threshold_y_pred = y_pred_windowed[threshold_mask]
            threshold_metrics = {
                "threshold": threshold,
                "SUPPORT": int(len(threshold_y_true)),
            }
            if len(threshold_y_true) == 0:
                logger.warning(
                    "No regression evaluation samples have true distance <= %s; "
                    "threshold metrics are unavailable.",
                    threshold,
                )
                for key in ("MAE", "RMSE", "MSE", "R2", "MAE_STD", "RMSE_STD"):
                    threshold_metrics[key] = np.nan
                threshold_metrics["residuals"] = np.array([])
                threshold_metrics["y_true"] = np.array([])
                threshold_metrics["y_pred"] = np.array([])
            else:
                threshold_residuals = threshold_y_true - threshold_y_pred
                threshold_mse = mean_squared_error(threshold_y_true, threshold_y_pred)
                threshold_metrics.update({
                    "MEAN_TARGET": float(np.mean(threshold_y_true)),
                    "MSE": float(threshold_mse),
                    "RMSE": float(np.sqrt(threshold_mse)),
                    "MAE": float(mean_absolute_error(threshold_y_true, threshold_y_pred)),
                    "R2": (
                        float(r2_score(threshold_y_true, threshold_y_pred))
                        if len(threshold_y_true) >= 2 else np.nan
                    ),
                    "MAE_STD": float(np.std(np.abs(threshold_residuals))),
                    "RMSE_STD": float(np.std(threshold_residuals ** 2)),
                    "residuals": threshold_residuals,
                    "y_true": threshold_y_true,
                    "y_pred": threshold_y_pred,
                })
            metrics["regression_threshold_evaluation"] = threshold_metrics

        if self.freq_limit_joblib:
            from joblib import load
            # import shap
            freq_limit_dict = load(self.freq_limit_joblib)
            band_limits = freq_limit_dict['band_limits']
            feature_names = [f"{round(start, 2)}-{round(end, 2)}" for start, end in band_limits]
            # Initialize the SHAP explainer with the training data as background and set feature names.
            # explainer = shap.Explainer(self.model, self.X_train, feature_names=feature_names)

            # Compute SHAP values for both test and training sets
            # shap_values_test = explainer(self.X_test)
            # shap_values_train = explainer(self.X_train)

            # For training data
            shap_values_train = compute_shap_values(
                model=self.model,
                X_train=self.X_train,
                X_data=self.X_train,
                feature_names=feature_names,
                nsamples_background=100,    # Adjust based on dataset and required precision
                nsamples_explain=1000       # Use a subset if X_train is huge
            )

            # Similarly for test data if needed:
            shap_values_test = compute_shap_values(
                model=self.model,
                X_train=self.X_train,
                X_data=self.X_test,
                feature_names=feature_names,
                nsamples_background=100,
                nsamples_explain=500        # Adjust accordingly
            )

            # Save only essentials (no need for the full Explanation object)
            metrics["shap"] = {
                "test_values": shap_values_test.values,
                "test_data": shap_values_test.data,
                "train_values": shap_values_train.values,
                "train_data": shap_values_train.data,
                "feature_names": feature_names,
                "base_value_test": shap_values_test.base_values,
                "base_value_train": shap_values_train.base_values,
            }

        if self.include_predictions:
            metrics["y_true"] = y_test_windowed
            metrics["y_pred"] = y_pred_windowed
            if self.dt_test is not None:
                # Adjust dt_test length to match y_test_windowed
                if self.instance_window and self.instance_window > 1:
                    num_windows = len(self.y_test) - self.instance_window + 1
                    metrics["datetimes"] = self.dt_test[:num_windows]
                else:
                    metrics["datetimes"] = self.dt_test[:len(y_test_windowed)]


        return metrics


    def _apply_mean_grouping(self, y_true, y_pred):
        """
        Applies mean value aggregation across overlapping instance_window-sized groups in the test set.
        """
        y_true_windowed, y_pred_windowed = [], []
        num_windows = len(y_true) - self.instance_window + 1  # Overlapping sliding windows

        # for i in range(num_windows):
        #     y_true_window = y_true[i:i + self.instance_window]
        #     y_pred_window = y_pred[i:i + self.instance_window]

        #     # Compute mean value for each window
        #     y_true_windowed.append(np.mean(y_true_window))
        #     y_pred_windowed.append(np.mean(y_pred_window))

        for i in range(num_windows):
            y_true_window = y_true[i:i + self.instance_window]
            y_pred_window = y_pred[i:i + self.instance_window]

            # Compute mean value for each window
            if self.center_ground_truth:
                center_index = self.instance_window // 2
                y_true_windowed.append(y_true_window[center_index])
            else:
                y_true_windowed.append(np.mean(y_true_window))
            y_pred_windowed.append(np.mean(y_pred_window))

        return np.array(y_pred_windowed), np.array(y_true_windowed)


    def evaluate_with_cross_validation(self):
        """Evaluate the model using cross-validation on the training set for RMSE and MAE."""

        if self.instance_window and self.instance_window > 1:
            X_train_windowed, y_train_windowed = self._apply_windowing(self.X_train, self.y_train)
            X_data, y_data = X_train_windowed, y_train_windowed
        else:
            X_data, y_data = self.X_train, self.y_train

        # Cross-validation for MSE
        cv_mse_scores = cross_val_score(
            self.model, X_data, y_data,
            cv=self.cv, scoring='neg_mean_squared_error'
        )
        mse_scores = -cv_mse_scores  # Convert negative MSE to positive
        rmse_scores = np.sqrt(mse_scores)  # Calculate RMSE

        # Cross-validation for MAE
        cv_mae_scores = cross_val_score(
            self.model, X_data, y_data,
            cv=self.cv, scoring='neg_mean_absolute_error'
        )
        mae_scores = -cv_mae_scores  # Convert negative MAE to positive

        # Log cross-validation results
        logger.info("Cross-validation Results:")
        logger.info("  MSE Scores: %s", mse_scores)
        logger.info("  RMSE Scores: %s", rmse_scores)
        logger.info("  MAE Scores: %s", mae_scores)
        logger.info("  Mean RMSE: %.4f", np.mean(rmse_scores))
        logger.info("  Mean MAE: %.4f", np.mean(mae_scores))

        # Return the metrics as a dictionary
        return {
            "CV_MSE_Scores": mse_scores,
            "CV_RMSE_Scores": rmse_scores,
            "CV_MAE_Scores": mae_scores,
            "Mean_CV_RMSE": np.mean(rmse_scores),
            "Mean_CV_MAE": np.mean(mae_scores)
        }


class ShipLabeler:
    def __init__(self, data):
        """
        Initialize with the data.
        - data: A list of lists, where each sublist contains dictionaries with 'mmsi', 'distance', and other information.
        """
        self.data = data


    def _get_closest_distance(self, ships):
        """Returns the minimum distance from a list of ship dictionaries."""
        return min(ship['distance'] for ship in ships)

    def _get_closest_distance_with_mmsi(self, ships, valid_mmsi=None, invalid_mmsi=None, use_invalid_mmsi=False):
        """Returns the minimum distance from ships considering either valid or invalid MMSI based on the flag."""
        if use_invalid_mmsi:
            filtered_ships = [ship for ship in ships if ship['mmsi'] not in invalid_mmsi]
        else:
            filtered_ships = [ship for ship in ships if ship['mmsi'] in valid_mmsi]

        if not filtered_ships:
            return float('inf')  # No valid ships, return a large number
        return min(ship['distance'] for ship in filtered_ships)


    def _get_n_ships_below_threshold(self, ships, threshold):
        """Returns the number of ships with a distance less than or equal to the threshold."""
        return sum(1 for ship in ships if ship['distance'] <= threshold)


    def _get_n_ships_below_threshold_with_mmsi(self, ships, threshold, valid_mmsi=None, invalid_mmsi=None, use_invalid_mmsi=False):
        """Returns the number of ships with a distance below the threshold considering valid or invalid MMSI."""
        if use_invalid_mmsi:
            return sum(1 for ship in ships if ship['distance'] <= threshold and ship['mmsi'] not in invalid_mmsi)
        else:
            return sum(1 for ship in ships if ship['distance'] <= threshold and ship['mmsi'] in valid_mmsi)

    def label_by_closest_distance(self, thresholds):
        """
        Label each list of ship dictionaries according to the closest distance and given thresholds.
        - thresholds: List of distance thresholds.
        """
        labels = []
        for ships in self.data:
            closest_distance = self._get_closest_distance(ships)
            label = self._assign_label_by_threshold(closest_distance, thresholds)
            labels.append(label)
        return labels

    def label_by_n_ships_below_threshold(self, thresholds, n):
        """
        Label each list of ship dictionaries according to whether at least n ships are below the given thresholds.
        - thresholds: List of distance thresholds.
        - n: Number of ships required to be below each threshold.
        """
        labels = []
        for ships in self.data:
            for i, threshold in enumerate(thresholds):
                if self._get_n_ships_below_threshold(ships, threshold) >= n:
                    labels.append(i)
                    break
            else:
                labels.append(len(thresholds))
        return labels

    def label_by_closest_distance_with_mmsi(self, thresholds, valid_mmsi=None, invalid_mmsi=None, use_invalid_mmsi=False):
        """
        Label each list of ship dictionaries according to the closest distance for ships in valid or invalid mmsi.
        - thresholds: List of distance thresholds.
        - valid_mmsi: List of valid MMSI numbers to consider.
        - invalid_mmsi: List of invalid MMSI numbers to exclude if use_invalid_mmsi is True.
        - use_invalid_mmsi: If True, exclude ships with invalid_mmsi instead of considering only valid_mmsi.
        """
        labels = []
        for ships in self.data:
            closest_distance = self._get_closest_distance_with_mmsi(ships, valid_mmsi, invalid_mmsi, use_invalid_mmsi)
            label = self._assign_label_by_threshold(closest_distance, thresholds)
            labels.append(label)
        return labels

    def label_by_n_ships_below_threshold_with_mmsi(self, thresholds, n, valid_mmsi=None, invalid_mmsi=None, use_invalid_mmsi=False):
        """
        Label each list of ship dictionaries according to whether at least n ships with valid or invalid MMSI are below thresholds.
        - thresholds: List of distance thresholds.
        - n: Number of ships required to be below each threshold.
        - valid_mmsi: List of valid MMSI numbers to consider.
        - invalid_mmsi: List of invalid MMSI numbers to exclude if use_invalid_mmsi is True.
        - use_invalid_mmsi: If True, exclude ships with invalid_mmsi instead of considering only valid_mmsi.
        """
        labels = []
        for ships in self.data:
            for i, threshold in enumerate(thresholds):
                if self._get_n_ships_below_threshold_with_mmsi(ships, threshold, valid_mmsi, invalid_mmsi, use_invalid_mmsi) >= n:
                    labels.append(i)
                    break
            else:
                labels.append(len(thresholds))
        return labels

    def _assign_label_by_threshold(self, distance, thresholds):
        """Assign a label based on the distance and the threshold list."""
        for i, threshold in enumerate(thresholds):
            if distance <= threshold:
                return i
        return len(thresholds)  # If the distance exceeds all thresholds, assign the highest label


class ShipDistanceTargetGenerator:
    def __init__(self, data, target_method='average', saturation_threshold=None,
                 valid_mmsi=None, invalid_mmsi=None, use_invalid_mmsi=False):
        """
        Initialize with the data, target method, and optional MMSI filters.
        - data: A list of lists, where each sublist contains dictionaries with 'mmsi', 'distance', and other information.
        - target_method: Method to determine the target ('minimum', 'average', or 'median').
        - saturation_threshold: Maximum distance threshold, beyond which values are clipped.
        - valid_mmsi: List of valid MMSI numbers to include.
        - invalid_mmsi: List of invalid MMSI numbers to exclude if use_invalid_mmsi is True.
        - use_invalid_mmsi: If True, the invalid_mmsi list will be used to filter out ships.
        """
        self.data = data
        self.target_method = target_method.lower()  # Ensure it's lowercase for consistency
        self.saturation_threshold = saturation_threshold  # Optional threshold for clipping
        self.valid_mmsi = valid_mmsi  # List of valid MMSI numbers (optional)
        self.invalid_mmsi = invalid_mmsi  # List of invalid MMSI numbers (optional)
        self.use_invalid_mmsi = use_invalid_mmsi  # Flag to switch between valid and invalid MMSI

    def generate_targets(self):
        """
        Generate targets for regression based on the specified target method and saturation threshold.
        Returns a list of distances representing the calculated target for each element in the data.
        """
        targets = []
        for ships in self.data:
            # Filter ships based on valid or invalid MMSI
            filtered_ships = self._filter_ships(ships)

            # Get the target distance based on the filtered ships
            target_distance = self._get_target_distance(filtered_ships)

            # Apply saturation threshold if specified
            if self.saturation_threshold is not None:
                target_distance = min(target_distance, self.saturation_threshold)

            targets.append(target_distance)
        return targets

    def _filter_ships(self, ships):
        """
        Filters ships based on valid or invalid MMSI.
        Returns the list of filtered ships.
        """
        if self.use_invalid_mmsi:
            # Exclude ships with MMSI in invalid_mmsi
            filtered_ships = [ship for ship in ships if ship['mmsi'] not in self.invalid_mmsi]
        elif self.valid_mmsi is not None:
            # Include only ships with MMSI in valid_mmsi
            filtered_ships = [ship for ship in ships if ship['mmsi'] in self.valid_mmsi]
        else:
            # No filtering, use all ships
            filtered_ships = ships

        return filtered_ships

    def _get_target_distance(self, ships):
        """
        Returns the target distance based on the specified method (minimum, average, or median).
        """
        distances = np.array([ship['distance'] for ship in ships])

        # Filter out invalid values (inf, NaN)
        valid_distances = distances[np.isfinite(distances)]

        # Handle the case where all values are invalid
        if len(valid_distances) == 0:
            return np.nan  # Return NaN if no valid distances are available

        # Apply the specified target method
        if self.target_method == 'minimum':
            return np.min(valid_distances)
        elif self.target_method == 'median':
            return np.median(valid_distances)
        else:  # Default is 'average'
            return np.mean(valid_distances)



class DataDateTimeBalancer:
    def __init__(self, X, y, dt):
        """
        Initialize the DataDateTimeBalancer class with training data and datetime list.

        Parameters:
        - X (array-like): Feature matrix
        - y (array-like): Target vector
        - dt (list of datetime): List of datetime objects corresponding to X and y
        """
        self.X = X
        self.y = y
        self.dt = np.array(dt)
        # Ensure we have datetime objects for operations that need timedelta arithmetic
        self.dt_datetime = self._ensure_datetime_array(self.dt)

    @staticmethod
    def _to_datetime(val):
        """
        Convert a value that may be a datetime, numpy datetime64, or string into
        a Python datetime object. Falls back to the Unix epoch if parsing fails.
        """
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except Exception:
            return datetime.fromtimestamp(0)

    def _ensure_datetime_array(self, dt_array):
        """Return a numpy array of datetime objects for safe arithmetic."""
        return np.array([self._to_datetime(v) for v in dt_array], dtype=object)

    def undersample(self, random_state=42, sampling_strategy='auto'):
        """
        Perform undersampling on the majority class and adjust the datetime list accordingly.

        Returns:
        - X_res (array-like): Resampled feature matrix
        - y_res (array-like): Resampled target vector
        - dt_res (array-like): Adjusted list of datetime objects
        """
        rus = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, y_res = rus.fit_resample(self.X, self.y)
        indices = rus.sample_indices_
        dt_res = self.dt[indices]
        # Shuffle data to avoid any bias issues related to ordering
        X_res, y_res, dt_res = shuffle(X_res, y_res, dt_res, random_state=random_state)
        return X_res, y_res, dt_res



    def oversample_smote(self, random_state=42, k_neighbors=5, sampling_strategy='auto'):
        """
        Perform oversampling on the minority class using SMOTE and adjust the datetime list by repeating datetimes.
        Synthetic datetimes are generated by slightly modifying the datetime of an existing entry.

        Returns:
        - X_res (array-like): Resampled feature matrix
        - y_res (array-like): Resampled target vector
        - dt_res (array-like): Adjusted list of datetime objects
        """
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        X_res, y_res = smote.fit_resample(self.X, self.y)
        dt_res = np.copy(self.dt_datetime)  # Start with datetime-safe copies


        # Extend the datetime array with modified datetimes of the original closest samples
        for i in range(len(self.y), len(y_res)):
            # Randomly pick a datetime from the original datetimes to modify
            original_datetime = np.random.choice(self.dt_datetime)
            # Adjust datetime slightly to avoid identical times for different classes
            adjustment = timedelta(minutes=1) if np.random.rand() > 0.5 else timedelta(minutes=-1)
            new_datetime = original_datetime + adjustment
            dt_res = np.append(dt_res, new_datetime)

        # Shuffle to ensure data structure is suitable for training
        X_res, y_res, dt_res = shuffle(X_res, y_res, dt_res, random_state=random_state)
        return X_res, y_res, dt_res


    def naive_oversample(self, random_state=42, sampling_strategy='auto'):
        """
        Perform naive random oversampling and adjust the datetime list by duplicating datetimes.

        Returns:
        - X_res (array-like): Resampled feature matrix
        - y_res (array-like): Resampled target vector
        - dt_res (array-like): Adjusted list of datetime objects
        """
        ros = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, y_res = ros.fit_resample(self.X, self.y)
        # Determine which indices were added to balance the classes
        added_indices = [i for i in range(len(y_res)) if i >= len(self.y)]
        # Duplicate the datetime for each added sample
        dt_res = np.copy(self.dt)
        for index in added_indices:
            dt_res = np.append(dt_res, self.dt[index % len(self.y)])

        # Shuffle to ensure data structure is suitable for training
        X_res, y_res, dt_res = shuffle(X_res, y_res, dt_res, random_state=random_state)
        return X_res, y_res, dt_res



    def oversample_adasyn(self, random_state=42, n_neighbors=5, sampling_strategy='auto'):
        """
        Perform ADASYN oversampling to generate synthetic samples proportionally to the number of nearby misclassifications,
        and adjust the datetime list by generating new datetimes based on the original datetimes.

        Returns:
        - X_res (array-like): Resampled feature matrix
        - y_res (array-like): Resampled target vector
        - dt_res (array-like): Adjusted list of datetime objects
        """
        adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors, sampling_strategy=sampling_strategy)
        X_res, y_res = adasyn.fit_resample(self.X, self.y)
        dt_res = np.copy(self.dt_datetime)  # Start with datetime-safe copies
        for i in range(len(self.y), len(y_res)):
            original_datetime = np.random.choice(self.dt_datetime)
            adjustment = timedelta(minutes=1) if np.random.rand() > 0.5 else timedelta(minutes=-1)
            new_datetime = original_datetime + adjustment
            dt_res = np.append(dt_res, new_datetime)
        X_res, y_res, dt_res = shuffle(X_res, y_res, dt_res, random_state=random_state)
        return X_res, y_res, dt_res



class ShipFilter:
    def __init__(self, ships):
        """
        Initialize with a list of ships where each ship is a dictionary.
        """
        self.ships = ships

    def filter_ships(self, condition):
        """
        Filters ships based on a condition.

        Args:
        - condition: A lambda or function that accepts a ship's dictionary and returns True if the ship meets the condition, False otherwise.

        Returns:
        - Two lists: one with mmsi numbers of ships that meet the condition and another with their indexes in the original list.
        """
        matching_mmsi = []
        matching_indexes = []

        for index, ship in enumerate(self.ships):
            if condition(ship):
                matching_mmsi.append(ship.get('mmsi'))
                matching_indexes.append(index)

        return matching_mmsi, matching_indexes



class ShipImpactAnalyzer:
    def __init__(self, X_reduced, y_reduced, dt_reduced, ships_reduced, classifier, zero_class_positive=True):
        """
        Initialize the analyzer with reduced data and a classifier.
        - zero_class_positive: If True, class 0 is considered positive and class > 0 as negative.
        """
        self.X_reduced = X_reduced
        self.y_reduced = y_reduced
        self.dt_reduced = dt_reduced
        self.ships_reduced = ships_reduced
        self.classifier = classifier
        self.zero_class_positive = zero_class_positive

    def analyze(self):
        """
        Analyze the impact of ships on classification results. If zero_class_positive is True,
        class 0 is considered the positive class, otherwise, class 1 is considered positive.
        """
        y_pred = self.classifier.predict(self.X_reduced)

        ship_stats = {
            'true_positives': {},
            'true_negatives': {},
            'false_positives': {},
            'false_negatives': {}
        }

        for i, (pred, actual, dt, ships) in enumerate(zip(y_pred, self.y_reduced, self.dt_reduced, self.ships_reduced)):
            nearest_ship = min(ships, key=lambda ship: ship['distance'])
            mmsi = nearest_ship['mmsi']

            if self.zero_class_positive:
                # Treat 0 as positive and class > 0 as negative
                if pred == 0 and actual == 0:
                    category = 'true_positives'
                elif pred > 0 and actual > 0:
                    category = 'true_negatives'
                elif pred == 0 and actual > 0:
                    category = 'false_positives'
                else:
                    category = 'false_negatives'
            else:
                # Treat class 1 as positive and class 0 as negative
                if pred == 1 and actual == 1:
                    category = 'true_positives'
                elif pred == 0 and actual == 0:
                    category = 'true_negatives'
                elif pred == 1 and actual == 0:
                    category = 'false_positives'
                else:
                    category = 'false_negatives'

            if mmsi not in ship_stats[category]:
                ship_stats[category][mmsi] = {'count': 0, 'times': []}
            ship_stats[category][mmsi]['count'] += 1
            ship_stats[category][mmsi]['times'].append(dt)

        sorted_stats = {
            category: sorted(mmsi_info.items(), key=lambda item: item[1]['count'], reverse=True)
            for category, mmsi_info in ship_stats.items()
        }

        return sorted_stats

    def save_to_csv(self, sorted_stats, file_name):
        """
        Save the sorted results to a CSV file.
        Each row will correspond to an MMSI, and each column to a category (TP, TN, FP, FN).
        """
        # Create a dictionary to hold all MMSI ships and their respective category counts
        all_mmsi_stats = {}

        # Iterate over each category and populate the dictionary
        for category, ships in sorted_stats.items():
            for mmsi, info in ships:
                if mmsi not in all_mmsi_stats:
                    all_mmsi_stats[mmsi] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

                if category == 'true_positives':
                    all_mmsi_stats[mmsi]['TP'] = info['count']
                elif category == 'true_negatives':
                    all_mmsi_stats[mmsi]['TN'] = info['count']
                elif category == 'false_positives':
                    all_mmsi_stats[mmsi]['FP'] = info['count']
                elif category == 'false_negatives':
                    all_mmsi_stats[mmsi]['FN'] = info['count']

        # Write the output to a CSV file
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(["MMSI", "True Positives", "True Negatives", "False Positives", "False Negatives"])

            # Write each ship's results
            for mmsi, counts in all_mmsi_stats.items():
                writer.writerow([mmsi, counts['TP'], counts['TN'], counts['FP'], counts['FN']])

        logger.info("Results successfully saved to %s", file_name)


class ShipPerformanceEvaluator:
    def __init__(self, sorted_stats):
        """
        Initialize with the sorted stats dictionary.
        - sorted_stats: Dictionary with keys ('true_positives', 'true_negatives', 'false_positives', 'false_negatives')
                        and values as lists of tuples, where each tuple contains (MMSI, {'count': int, 'times': list}).
        """
        self.sorted_stats = sorted_stats
        self.metrics = {}

    def _compute_metrics(self):
        """
        Compute metrics for each ship based on TP, TN, FP, and FN, and also calculate the number of occurrences.
        """
        for mmsi in self._get_all_mmsi():
            tp = self._get_count('true_positives', mmsi)
            tn = self._get_count('true_negatives', mmsi)
            fp = self._get_count('false_positives', mmsi)
            fn = self._get_count('false_negatives', mmsi)

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            occurrences = tp + tn + fp + fn  # Total occurrences

            self.metrics[mmsi] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'occurrences': occurrences
            }

    def _get_all_mmsi(self):
        """
        Get the set of all MMSI numbers from all categories (TP, TN, FP, FN).
        """
        mmsi_set = set()

        for category in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            for mmsi, _ in self.sorted_stats[category]:
                mmsi_set.add(mmsi)

        return mmsi_set

    def _get_count(self, category, mmsi):
        """
        Retrieve the count for a specific MMSI from the specified category.
        If the MMSI is not found in the category, return 0.
        """
        for mmsi_val, info in self.sorted_stats[category]:
            if mmsi_val == mmsi:
                return info['count']
        return 0

    def sort_ships(self, metric='accuracy'):
        """
        Sort the ships based on the specified metric (accuracy, precision, recall, f1).
        - metric: The metric to use for sorting ('accuracy', 'precision', 'recall', 'f1').
        Returns a sorted list of MMSI numbers from best to worst.
        """
        if metric not in ['accuracy', 'precision', 'recall', 'f1']:
            raise ValueError("Invalid metric. Choose from 'accuracy', 'precision', 'recall', or 'f1'.")

        self._compute_metrics()
        return sorted(self.metrics.items(), key=lambda item: item[1][metric], reverse=True)

    def get_sorted_mmsi(self, metric='accuracy'):
        """
        Get a list of MMSI sorted from highest to lowest based on the chosen metric.
        - metric: The metric to use for sorting ('accuracy', 'precision', 'recall', 'f1').
        Returns a sorted list of MMSI from best to worst.
        """
        if metric not in ['accuracy', 'precision', 'recall', 'f1']:
            raise ValueError("Invalid metric. Choose from 'accuracy', 'precision', 'recall', or 'f1'.")

        self._compute_metrics()

        # Sort MMSI based on the chosen metric
        sorted_mmsi = sorted(self.metrics.items(), key=lambda item: item[1][metric], reverse=True)

        # Return only the MMSI numbers
        return [mmsi for mmsi, _ in sorted_mmsi]

    def get_metrics(self):
        """
        Return the computed metrics for each ship.
        """
        self._compute_metrics()
        return self.metrics

    def save_metrics_to_csv(self, file_name):
        """
        Save the computed metrics to a CSV file.
        Each row will correspond to a ship (MMSI), and each column will show a metric (accuracy, precision, recall, F1-score).
        """
        # Ensure metrics are computed
        self._compute_metrics()

        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header, including the occurrences column
            writer.writerow(["MMSI", "Accuracy", "Precision", "Recall", "F1-score", "Occurrences"])

            # Write each ship's metrics
            for mmsi, metrics in self.metrics.items():
                writer.writerow([mmsi, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['occurrences']])

        logger.info("Metrics successfully saved to %s", file_name)

class VesselDateTimeInserter:
    def __init__(self, vessel_lists, dt):
        """
        Initialize with vessel lists and corresponding datetimes.
        - vessel_lists: A list of lists, where each sublist contains dictionaries representing ships.
        - dt: A list of utc-aware datetime objects corresponding to vessel_lists.
        """
        if len(vessel_lists) != len(dt):
            raise ValueError("The length of vessel_lists and dt must be the same.")
        self.vessel_lists = vessel_lists
        self.dt = dt

    def insert_datetimes(self):
        """
        Insert the corresponding datetime into each ship dictionary in vessel_lists.
        Adds a 'datetime' key with the value from dt to each ship dictionary.
        """
        for vessels, current_dt in zip(self.vessel_lists, self.dt):
            for ship in vessels:
                ship['datetime'] = current_dt  # Add the datetime to each ship's dictionary

        return self.vessel_lists

class VesselListReducerByDatetime:
    def __init__(self, vessel_lists, dt):
        """
        Initialize with vessel lists and a smaller list of datetimes.
        - vessel_lists: A list of lists, where each sublist contains dictionaries representing ships.
        Each sublist has the same datetime for all the ships.
        - dt: A smaller list of utc-aware datetime objects used for reducing vessel_lists.
        """
        self.vessel_lists = vessel_lists
        self.dt = dt

        # Preprocess vessel lists by mapping each sublist's datetime to its list of ships
        self.datetime_to_vessels = self._create_datetime_vessel_map()

    def _create_datetime_vessel_map(self):
        """
        Create a dictionary that maps each datetime to the corresponding sublist of ships.
        Only check the first ship's 'datetime' in each sublist since all ships in the sublist share the same datetime.
        """
        datetime_to_vessels = {}

        # Map the datetime of the first ship in each sublist to the entire sublist of vessels
        for vessels in self.vessel_lists:
            if vessels:  # Ensure there is at least one vessel in the sublist
                first_datetime = vessels[0]['datetime']
                datetime_to_vessels[first_datetime] = vessels

        return datetime_to_vessels

    def reduce_by_datetimes(self):
        """
        Reduce vessel_lists by retaining only the lists of ships whose 'datetime' matches any in dt.
        The order of the reduced vessel lists will follow the order of dt.
        """
        reduced_vessel_lists = []

        # For each datetime in dt, find the matching vessels using the precomputed dictionary
        for target_datetime in self.dt:
            matching_vessels = self.datetime_to_vessels.get(target_datetime, [])
            if matching_vessels:
                reduced_vessel_lists.append(matching_vessels)

        return reduced_vessel_lists



class TimeOffsetApplier:
    def __init__(self, X, y, dt, closest_distances_list=None, time_offset_seconds=7200):
        """
        Initialize with the features (X), labels (y), datetimes (dt), and optional closest_distances.
        - X: List or np.array of features (2D or 3D array).
        - y: List or np.array of labels.
        - dt: List of utc-aware datetime objects.
        - closest_distances_list: Optional list of distances corresponding to each entry in X.
        - time_offset_seconds: Time offset to apply in seconds (default is 7200 seconds).
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.dt = dt
        self.closest_distances_list = closest_distances_list
        self.time_offset_seconds = time_offset_seconds

    def apply_time_offset(self):
        """
        Correct features (X), labels (y), datetimes (dt), and optional closest distances by shifting them based on the time offset in seconds.
        Only keep entries where the offset is valid.
        """
        offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
        n_samples = len(self.X)

        if offset_samples >= n_samples:
            raise ValueError("Time offset too large, resulting in loss of all data")

        # Apply the offset and discard the first offset_samples entries
        self.X = self.X[offset_samples:n_samples]
        self.y = self.y[offset_samples:n_samples]
        self.dt = self.dt[offset_samples:n_samples]

        if self.closest_distances_list is not None:
            self.closest_distances_list = self.closest_distances_list[offset_samples:n_samples]

    def get_data(self):
        """
        Returns the adjusted features, labels, datetimes, and optionally closest distances.
        """
        if self.closest_distances_list is not None:
            return self.X, self.y, self.dt, self.closest_distances_list
        return self.X, self.y, self.dt
