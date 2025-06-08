import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from pprint import pprint
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.utils import shuffle
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

import csv
from bisect import bisect_left


class TripletReducer:
    def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None,
                 join_higher_classes=True, average_signals=False, apply_log=True, 
                 epsilon=1e-10, time_offset_seconds=7200):
        """
        Initialize with the data to be reduced and the parameters.
        - X: List of 2D numpy arrays.
        - y: List of integers (classification) or floats (regression).
        - dt: List of utc-aware datetime objects.
        - ships: Optional ship data to reduce, list of lists (each sublist contains dictionaries with ship data).
        - n_seconds: Total number of seconds to average.
        - n_overlapping_seconds: If specified, overlap between groups in seconds.
        - join_higher_classes: If True, convert all y >= 1 to 1 (for classification).
        - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
        - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
        - epsilon: Small value added to X before applying log to avoid NaNs.
        - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
        """
        self.X = X
        self.y = y
        self.dt = dt
        self.ships = ships  # Ship data to be reduced if provided
        self.n_seconds = n_seconds
        self.n_overlapping_seconds = n_overlapping_seconds
        self.join_higher_classes = join_higher_classes
        self.average_signals = average_signals
        self.apply_log = apply_log
        self.epsilon = epsilon
        self.time_offset_seconds = time_offset_seconds

        # Sort the triplets chronologically before applying any offset and track the original indices
        self._sort_triplets()

        # Apply the time offset correction if necessary
        if time_offset_seconds is not None:
            self._apply_time_offset()

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
        """Efficiently correct the labels by checking for the exact target dt + time_offset_seconds within a 5-second tolerance."""
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
    
            # Use binary search to find the closest index where target_dt could be inserted
            insert_position = bisect_left(self.dt, target_dt)
    
            # Check the nearest neighbors within the tolerance
            if insert_position < len(self.dt):
                # Check the candidate at the insert position
                if abs(self.dt[insert_position] - target_dt) <= time_tolerance:
                    reduced_X.append(self.X[i])  # Keep the current X
                    reduced_y.append(self.y[insert_position])  # Use the corresponding y at dt + offset
                    reduced_dt.append(target_dt)  # Keep the target_dt 
                    if self.ships:
                        reduced_ships.append(self.ships[i])
                # Check the candidate just before the insert position
                elif insert_position > 0 and abs(self.dt[insert_position - 1] - target_dt) <= time_tolerance:
                    reduced_X.append(self.X[i])  # Keep the current X
                    reduced_y.append(self.y[insert_position - 1])  # Use the corresponding y at dt + offset
                    reduced_dt.append(current_dt)  # Keep the current dt (not the shifted one)
                    if self.ships:
                        reduced_ships.append(self.ships[i])
    
        # Replace the original X, y, dt, ships with the reduced ones
        self.X = np.array(reduced_X)
        self.y = np.array(reduced_y)
        self.dt = reduced_dt
        if self.ships:
            self.ships = reduced_ships

            

    # def _apply_time_offset(self):
    #     """Correct the labels by checking for the exact target dt + time_offset_seconds within a 5-second tolerance."""
    #     if self.time_offset_seconds is None:
    #         return
    
    #     # Time tolerance in seconds
    #     time_tolerance = timedelta(seconds=5)
        
    #     # Create a mapping from dt to index for quick lookup
    #     dt_to_index = {dt: idx for idx, dt in enumerate(self.dt)}
    
    #     # Lists to hold the valid reduced triplets
    #     reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
    
    #     # Iterate through the sorted datetimes (dt)
    #     for i, current_dt in enumerate(self.dt):
    #         # Calculate the target time
    #         target_dt = current_dt + timedelta(seconds=self.time_offset_seconds)
    
    #         # Find the closest dt within the tolerance
    #         closest_dt = min(dt_to_index.keys(), key=lambda dt: abs(dt - target_dt))
    
    #         # If the closest dt is within the 5-second tolerance, keep the triplet
    #         if abs(closest_dt - target_dt) <= time_tolerance:
    #             j = dt_to_index[closest_dt]  # Get the corresponding index for the target dt
    #             reduced_X.append(self.X[i])  # Keep the current X
    #             reduced_y.append(self.y[j])  # Use the corresponding y at dt + offset
    #             reduced_dt.append(current_dt)  # Keep the current dt (not the shifted one)
    
    #             # If ships are provided, reduce the ships list as well
    #             if self.ships:
    #                 reduced_ships.append(self.ships[i])
    
    #     # Replace the original X, y, dt, ships with the reduced ones
    #     self.X = np.array(reduced_X)
    #     self.y = np.array(reduced_y)
    #     self.dt = reduced_dt
    #     if self.ships:
    #         self.ships = reduced_ships
    

    # def _apply_time_offset(self):
    #     """Correct the labels by shifting them according to the time offset in seconds."""
    #     offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
    #     n_samples = len(self.X)
        
    #     if offset_samples < n_samples:
    #         # Shift X, y, dt according to the time offset, discarding the first offset_samples entries from y and dt
    #         # Keep ships if provided, and apply the same logic.
    #         self.X = self.X[:n_samples - offset_samples]
    #         self.y = self.y[offset_samples:n_samples]
    #         self.dt = self.dt[offset_samples:n_samples]
    #         if self.ships is not None:
    #             self.ships = self.ships[offset_samples:n_samples]
    #     else:
    #         raise ValueError("Time offset too large, resulting in loss of all data")

    def reduce_triplets(self):
        """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
        n_samples_per_group = self.n_seconds // 10
        n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
        i = 0
        while i < len(self.X):
            # Define the current group range
            group_X = self.X[i:i + n_samples_per_group]
            group_y = self.y[i:i + n_samples_per_group]
            group_dt = self.dt[i:i + n_samples_per_group]
            group_ships = self.ships[i:i + n_samples_per_group] if self.ships else []

            if len(group_X) < n_samples_per_group:
                break  # Skip incomplete groups at the end

            # Average the Xs
            avg_X = np.mean(group_X, axis=0)

            # Optionally average across all rows (producing a 1-D array)
            if self.average_signals:
                avg_X = np.mean(avg_X, axis=0)

            # Apply logarithmic transformation if needed, ensuring no NaNs
            if self.apply_log:
                avg_X = np.log(np.maximum(avg_X, self.epsilon))

            # Take the lowest y value, and apply join_higher_classes if needed
            min_y = min(group_y)
            if self.join_higher_classes:
                min_y = 1 if min_y >= 1 else 0

            # Take the oldest datetime
            oldest_dt = min(group_dt)

            # Store the reduced triplet
            reduced_X.append(avg_X)
            reduced_y.append(min_y)
            reduced_dt.append(oldest_dt)

            # If ships data is provided, reduce it
            if self.ships:
                # Find the list that contains the closest ship based on 'distance'
                closest_list = min(group_ships, key=lambda ship_list: min(ship['distance'] for ship in ship_list))
                reduced_ships.append(closest_list)


            # Move the index by the group size minus overlap
            i += n_samples_per_group - n_overlap_samples

        if self.ships:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
        return np.array(reduced_X), np.array(reduced_y), reduced_dt



# class TripletReducer:
#     def __init__(self, X, y, dt, n_seconds, n_overlapping_seconds=None, 
#                   join_higher_classes=True, average_signals=False, 
#                   apply_log=True, epsilon=1e-10, time_offset_seconds=2700):
#         """
#         Initialize with the data to be reduced and the parameters.
#         - X: List of 2D numpy arrays.
#         - y: List of integers.
#         - dt: List of utc-aware datetime objects.
#         - n_seconds: Total number of seconds to average.
#         - n_overlapping_seconds: If specified, overlap between groups in seconds.
#         - join_higher_classes: If True, convert all y >= 1 to 1.
#         - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
#         - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
#         - epsilon: Small value added to X before applying log to avoid NaNs.
#         - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
#         """
#         self.X = X
#         self.y = y
#         self.dt = dt
#         self.n_seconds = n_seconds
#         self.n_overlapping_seconds = n_overlapping_seconds
#         self.join_higher_classes = join_higher_classes
#         self.average_signals = average_signals
#         self.apply_log = apply_log
#         self.epsilon = epsilon
#         self.time_offset_seconds = time_offset_seconds

#         # Sort the triplets chronologically before applying any offset
#         self._sort_triplets()

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

#     def _sort_triplets(self):
#         """Sort the triplets (X, y, dt) chronologically by dt."""
#         triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
#         X_sorted, y_sorted, dt_sorted = zip(*triplets)
#         self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

#     def _apply_time_offset(self):
#         """Correct the labels by shifting them according to the time offset in seconds."""
#         # Each sample represents 10 seconds, so we calculate how many samples correspond to the time offset
#         offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
#         n_samples = len(self.X)
        
#         if offset_samples < n_samples:
#             # Shift X, y, dt according to the time offset, discarding the first offset_samples entries from y and dt
#             # And keeping the first nb - offset_samples entries of X
#             self.X = np.array(self.X[:n_samples - offset_samples])
#             self.y = np.array(self.y[offset_samples:n_samples])
#             self.dt = np.array(self.dt[offset_samples:n_samples])
#         else:
#             raise ValueError("Time offset too large, resulting in loss of all data")

#     def reduce_triplets(self):
#         """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt = [], [], []
#         i = 0
#         while i < len(self.X):
#             # Define the current group range
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
            
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

#             # Take the lowest y value, and apply join_higher_classes if needed
#             min_y = min(group_y)
#             if self.join_higher_classes:
#                 min_y = 1 if min_y >= 1 else 0

#             # Take the oldest datetime
#             oldest_dt = min(group_dt)

#             # Store the reduced triplet
#             reduced_X.append(avg_X)
#             reduced_y.append(min_y)
#             reduced_dt.append(oldest_dt)

#             # Move the index by the group size minus overlap
#             i += n_samples_per_group - n_overlap_samples

#         # Convert the reduced X and y to numpy arrays
#         return np.array(reduced_X), np.array(reduced_y), reduced_dt


class TripletRegressionReducer:
    def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None, 
                 average_signals=False, apply_log=True, epsilon=1e-10, 
                 time_offset_seconds=7200, target_method='average', 
                 threshold=None, eliminate_within_range=None):
        """
        Initialize with the data to be reduced and the parameters.
        - X: List of 2D numpy arrays.
        - y: List of continuous values (for regression).
        - dt: List of utc-aware datetime objects.
        - ships: Optional ship data to reduce, list of lists (each sublist contains dictionaries with ship data).
        - n_seconds: Total number of seconds to average.
        - n_overlapping_seconds: If specified, overlap between groups in seconds.
        - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
        - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
        - epsilon: Small value added to X before applying log to avoid NaNs.
        - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
        - target_method: Method to determine the target ('minimum', 'average', 'median').
        - threshold: If provided, any y value greater than this threshold will be filtered out along with its corresponding X and dt.
        - eliminate_within_range: If provided, a tuple (low, high) to eliminate y values within that range.
        """
        self.X = X
        self.y = y
        self.dt = dt
        self.ships = ships  # Ship data to be reduced if provided
        self.n_seconds = n_seconds
        self.n_overlapping_seconds = n_overlapping_seconds
        self.average_signals = average_signals
        self.apply_log = apply_log
        self.epsilon = epsilon
        self.time_offset_seconds = time_offset_seconds
        self.target_method = target_method.lower()
        self.threshold = threshold
        self.eliminate_within_range = eliminate_within_range

        # Sort the triplets chronologically before applying any offset
        self._sort_triplets()

        # Apply the time offset correction if necessary
        if time_offset_seconds is not None:
            self._apply_time_offset()

        # Apply the threshold filtering if necessary
        if threshold is not None or eliminate_within_range is not None:
            self._apply_threshold()

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
        """Efficiently correct the labels by checking for the exact target dt + time_offset_seconds within a 5-second tolerance."""
        if self.time_offset_seconds is None:
            return
    
        # Time tolerance in seconds
        time_tolerance = timedelta(seconds=5)
    
        # Lists to hold the valid reduced triplets
        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
    
        # Create a mapping from dt to index for quick lookup using binary search
        for i, current_dt in enumerate(self.dt):
            # Calculate the target time
            target_dt = current_dt + timedelta(seconds=self.time_offset_seconds)
    
            # Use binary search to find the closest index where target_dt could be inserted
            insert_position = bisect_left(self.dt, target_dt)
    
            # Check the nearest neighbors within the tolerance
            if insert_position < len(self.dt):
                # Check the candidate at the insert position
                if abs(self.dt[insert_position] - target_dt) <= time_tolerance:
                    reduced_X.append(self.X[i])  # Keep the current X
                    reduced_y.append(self.y[insert_position])  # Use the corresponding y at dt + offset
                    reduced_dt.append(target_dt)  # Keep the target_dt 
                    if self.ships:
                        reduced_ships.append(self.ships[i])
                # Check the candidate just before the insert position
                elif insert_position > 0 and abs(self.dt[insert_position - 1] - target_dt) <= time_tolerance:
                    reduced_X.append(self.X[i])  # Keep the current X
                    reduced_y.append(self.y[insert_position - 1])  # Use the corresponding y at dt + offset
                    reduced_dt.append(current_dt)  # Keep the current dt (not the shifted one)
                    if self.ships:
                        reduced_ships.append(self.ships[i])
    
        # Replace the original X, y, dt, ships with the reduced ones
        self.X = np.array(reduced_X)
        self.y = np.array(reduced_y)
        self.dt = reduced_dt
        if self.ships:
            self.ships = reduced_ships

    def _apply_threshold(self):
        """Remove X, y, dt, and ships entries where y is greater than the threshold or within the range."""
        valid_indices = []
        
        if self.threshold is not None:
            # Create a mask for entries where y is less than or equal to the threshold
            valid_indices = [i for i, val in enumerate(self.y) if val <= self.threshold]
        
        if self.eliminate_within_range is not None:
            low, high = self.eliminate_within_range
            # Extend the valid_indices to exclude y within the specified range
            valid_indices = [i for i, val in enumerate(self.y) if not (low <= val <= high)]

        # Filter X, y, dt, and ships based on the valid indices
        if valid_indices:
            self.X = np.array([self.X[i] for i in valid_indices])
            self.y = np.array([self.y[i] for i in valid_indices])
            self.dt = np.array([self.dt[i] for i in valid_indices])
            if self.ships is not None:
                self.ships = [self.ships[i] for i in valid_indices]

    def reduce_triplets(self):
        """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
        n_samples_per_group = self.n_seconds // 10
        n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0
    
        reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
        i = 0
        while i < len(self.X):
            # Define the current group range
            group_X = self.X[i:i + n_samples_per_group]
            group_y = self.y[i:i + n_samples_per_group]
            group_dt = self.dt[i:i + n_samples_per_group]
            group_ships = self.ships[i:i + n_samples_per_group] if self.ships else None
            
            if len(group_X) < n_samples_per_group:
                break  # Skip incomplete groups at the end
    
            # Average the Xs
            avg_X = np.mean(group_X, axis=0)
    
            # Optionally average across all rows (producing a 1-D array)
            if self.average_signals:
                avg_X = np.mean(avg_X, axis=0)
    
            # Apply logarithmic transformation if needed, ensuring no NaNs
            if self.apply_log:
                avg_X = np.log(np.maximum(avg_X, self.epsilon))
    
            # Determine the y value (target) based on the target method
            if self.target_method == 'minimum':
                target_y = np.min(group_y)
            elif self.target_method == 'median':
                target_y = np.median(group_y)
            else:
                target_y = np.mean(group_y)  # Default is 'average'
    
            # Take the oldest datetime
            oldest_dt = min(group_dt)
    
            # Reduce ship data if present
            if group_ships:
                # Flatten the list of lists of ships
                flattened_ships = [ship for sublist in group_ships for ship in sublist]
                
                # Find the ship with the minimum distance
                min_distance_ship = min(flattened_ships, key=lambda s: s.get('distance', float('inf')))
                
                # Append the entire list of ships containing the closest ship
                reduced_ships.append([min_distance_ship])
    
            # Store the reduced triplet
            reduced_X.append(avg_X)
            reduced_y.append(target_y)
            reduced_dt.append(oldest_dt)
    
            # Move the index by the group size minus overlap
            i += n_samples_per_group - n_overlap_samples
    
        if self.ships:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
        else:
            return np.array(reduced_X), np.array(reduced_y), reduced_dt



# class TripletRegressionReducer:
#     def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None, 
#                  average_signals=False, apply_log=True, epsilon=1e-10, 
#                  time_offset_seconds=2700, target_method='average', threshold=None):
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

#         # Sort the triplets chronologically before applying any offset
#         self._sort_triplets()

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

#         # Apply the threshold filtering if necessary
#         if threshold is not None:
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
#         """Efficiently correct the labels by checking for the exact target dt + time_offset_seconds within a 5-second tolerance."""
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


#     # def _apply_time_offset(self):
#     #     """Correct the labels by shifting them according to the time offset in seconds."""
#     #     offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
#     #     n_samples = len(self.X)
        
#     #     if offset_samples < n_samples:
#     #         # Shift X, y, dt according to the time offset, discarding the first offset_samples entries from y and dt
#     #         self.X = np.array(self.X[:n_samples - offset_samples])
#     #         self.y = np.array(self.y[offset_samples:n_samples])
#     #         self.dt = np.array(self.dt[offset_samples:n_samples])
#     #         if self.ships is not None:
#     #             # Ships are handled as a regular list (not a NumPy array) to accommodate varying list sizes
#     #             self.ships = self.ships[offset_samples:n_samples]
#     #     else:
#     #         raise ValueError("Time offset too large, resulting in loss of all data")

#     def _apply_threshold(self):
#         """Remove X, y, dt, and ships entries where y is greater than the threshold."""
#         # Create a mask for entries where y is less than or equal to the threshold
#         valid_indices = [i for i, val in enumerate(self.y) if val <= self.threshold]
        
#         # Filter X, y, dt, and ships based on the valid indices
#         self.X = np.array([self.X[i] for i in valid_indices])
#         self.y = np.array([self.y[i] for i in valid_indices])
#         self.dt = np.array([self.dt[i] for i in valid_indices])
#         if self.ships is not None:
#             self.ships = [self.ships[i] for i in valid_indices]

    # def reduce_triplets(self):
    #     """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
    #     n_samples_per_group = self.n_seconds // 10
    #     n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0
    
    #     reduced_X, reduced_y, reduced_dt, reduced_ships = [], [], [], []
    #     i = 0
    #     while i < len(self.X):
    #         # Define the current group range
    #         group_X = self.X[i:i + n_samples_per_group]
    #         group_y = self.y[i:i + n_samples_per_group]
    #         group_dt = self.dt[i:i + n_samples_per_group]
    #         group_ships = self.ships[i:i + n_samples_per_group] if self.ships else None
            
    #         if len(group_X) < n_samples_per_group:
    #             break  # Skip incomplete groups at the end
    
    #         # Average the Xs
    #         avg_X = np.mean(group_X, axis=0)
    
    #         # Optionally average across all rows (producing a 1-D array)
    #         if self.average_signals:
    #             avg_X = np.mean(avg_X, axis=0)
    
    #         # Apply logarithmic transformation if needed, ensuring no NaNs
    #         if self.apply_log:
    #             avg_X = np.log(np.maximum(avg_X, self.epsilon))
    
    #         # Determine the y value (target) based on the target method
    #         if self.target_method == 'minimum':
    #             target_y = np.min(group_y)
    #         elif self.target_method == 'median':
    #             target_y = np.median(group_y)
    #         else:
    #             target_y = np.mean(group_y)  # Default is 'average'
    
    #         # Take the oldest datetime
    #         oldest_dt = min(group_dt)
    
    #         # Reduce ship data if present
    #         if group_ships:
    #             # Flatten the list of lists of ships
    #             flattened_ships = [ship for sublist in group_ships for ship in sublist]
                
    #             # Find the ship with the minimum distance
    #             min_distance_ship = min(flattened_ships, key=lambda s: s.get('distance', float('inf')))
                
    #             # Append the entire list of ships containing the closest ship
    #             reduced_ships.append([min_distance_ship])
    
    #         # Store the reduced triplet
    #         reduced_X.append(avg_X)
    #         reduced_y.append(target_y)
    #         reduced_dt.append(oldest_dt)
    
    #         # Move the index by the group size minus overlap
    #         i += n_samples_per_group - n_overlap_samples
    
    #     if self.ships:
    #         return np.array(reduced_X), np.array(reduced_y), reduced_dt, reduced_ships
    #     return np.array(reduced_X), np.array(reduced_y), reduced_dt


# class TripletRegressionReducer:
#     def __init__(self, X, y, dt, ships=None, n_seconds=10, n_overlapping_seconds=None, 
#                  average_signals=False, apply_log=True, epsilon=1e-10, 
#                  time_offset_seconds=2700, target_method='average'):
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

#         # Sort the triplets chronologically before applying any offset
#         self._sort_triplets()

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

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
#         """Correct the labels by shifting them according to the time offset in seconds."""
#         offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
#         n_samples = len(self.X)
        
#         if offset_samples < n_samples:
#             # Shift X, y, dt according to the time offset, discarding the first offset_samples entries from y and dt
#             self.X = np.array(self.X[:n_samples - offset_samples])
#             self.y = np.array(self.y[offset_samples:n_samples])
#             self.dt = np.array(self.dt[offset_samples:n_samples])
#             if self.ships is not None:
#                 # Ships are handled as a regular list (not a NumPy array) to accommodate varying list sizes
#                 self.ships = self.ships[offset_samples:n_samples]
#         else:
#             raise ValueError("Time offset too large, resulting in loss of all data")

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
#         return np.array(reduced_X), np.array(reduced_y), reduced_dt


#     def reduce_triplets(self):
#         """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt = [], [], []
#         i = 0
#         while i < len(self.X):
#             # Define the current group range
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
            
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

#             # Store the reduced triplet
#             reduced_X.append(avg_X)
#             reduced_y.append(target_y)
#             reduced_dt.append(oldest_dt)

#             # Move the index by the group size minus overlap
#             i += n_samples_per_group - n_overlap_samples

#         # Convert the reduced X and y to numpy arrays
#         return np.array(reduced_X), np.array(reduced_y), reduced_dt

# class TripletRegressionReducer:
#     def __init__(self, X, y, dt, n_seconds, n_overlapping_seconds=None, 
#                  average_signals=False, apply_log=True, epsilon=1e-10, 
#                  time_offset_seconds=2700, target_method='average'):
#         """
#         Initialize with the data to be reduced and the parameters.
#         - X: List of 2D numpy arrays.
#         - y: List of continuous values (for regression).
#         - dt: List of utc-aware datetime objects.
#         - n_seconds: Total number of seconds to average.
#         - n_overlapping_seconds: If specified, overlap between groups in seconds.
#         - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
#         - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
#         - epsilon: Small value added to X before applying log to avoid NaNs.
#         - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
#         - target_method: Method to determine the target ('minimum', 'average', 'median').
#         """
#         self.X = X
#         self.y = y
#         self.dt = dt
#         self.n_seconds = n_seconds
#         self.n_overlapping_seconds = n_overlapping_seconds
#         self.average_signals = average_signals
#         self.apply_log = apply_log
#         self.epsilon = epsilon
#         self.time_offset_seconds = time_offset_seconds
#         self.target_method = target_method.lower()

#         # Sort the triplets chronologically before applying any offset
#         self._sort_triplets()

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

#     def _sort_triplets(self):
#         """Sort the triplets (X, y, dt) chronologically by dt."""
#         triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
#         X_sorted, y_sorted, dt_sorted = zip(*triplets)
#         self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

#     def _apply_time_offset(self):
#         """Correct the labels by shifting them according to the time offset in seconds."""
#         # Each sample represents 10 seconds, so we calculate how many samples correspond to the time offset
#         offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
#         n_samples = len(self.X)
        
#         if offset_samples < n_samples:
#             # Shift X, y, dt according to the time offset, discarding the first offset_samples entries from y and dt
#             # And keeping the first nb - offset_samples entries of X
#             self.X = np.array(self.X[:n_samples - offset_samples])
#             self.y = np.array(self.y[offset_samples:n_samples])
#             self.dt = np.array(self.dt[offset_samples:n_samples])
#         else:
#             raise ValueError("Time offset too large, resulting in loss of all data")

#     def reduce_triplets(self):
#         """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt = [], [], []
#         i = 0
#         while i < len(self.X):
#             # Define the current group range
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
            
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

#             # Store the reduced triplet
#             reduced_X.append(avg_X)
#             reduced_y.append(target_y)
#             reduced_dt.append(oldest_dt)

#             # Move the index by the group size minus overlap
#             i += n_samples_per_group - n_overlap_samples

#         # Convert the reduced X and y to numpy arrays
#         return np.array(reduced_X), np.array(reduced_y), reduced_dt


# class TripletReducer:
#     def __init__(self, X, y, dt, n_seconds, n_overlapping_seconds=None, 
#                   join_higher_classes=True, average_signals=False, 
#                   apply_log=True, epsilon=1e-10, time_offset_seconds=7200):
#         """
#         Initialize with the data to be reduced and the parameters.
#         - X: List of 2D numpy arrays.
#         - y: List of integers.
#         - dt: List of utc-aware datetime objects.
#         - n_seconds: Total number of seconds to average.
#         - n_overlapping_seconds: If specified, overlap between groups in seconds.
#         - join_higher_classes: If True, convert all y >= 1 to 1.
#         - average_signals: If True, reduce groups of X to 1-D arrays by averaging over both axes.
#         - apply_log: If True, apply np.log to the reduced X while avoiding NaNs.
#         - epsilon: Small value added to X before applying log to avoid NaNs.
#         - time_offset_seconds: If provided, shift the labels (y) by this time offset in seconds.
#         """
#         self.X = X
#         self.y = y
#         self.dt = dt
#         self.n_seconds = n_seconds
#         self.n_overlapping_seconds = n_overlapping_seconds
#         self.join_higher_classes = join_higher_classes
#         self.average_signals = average_signals
#         self.apply_log = apply_log
#         self.epsilon = epsilon
#         self.time_offset_seconds = time_offset_seconds

#         # Apply the time offset correction if necessary
#         if time_offset_seconds is not None:
#             self._apply_time_offset()

#     def _apply_time_offset(self):
#         """Correct the labels by shifting them according to the time offset in seconds."""
#         offset_samples = self.time_offset_seconds // 10  # Each sample represents 10 seconds
#         n_samples = len(self.X)
        
#         if offset_samples < n_samples:
#             # Shift X, y, dt according to the time offset, discarding the offset data
#             self.X = self.X[:n_samples - offset_samples]
#             self.y = self.y[offset_samples:n_samples]
#             self.dt = self.dt[offset_samples:n_samples]
#         else:
#             raise ValueError("Time offset too large, resulting in loss of all data")

#     def sort_triplets(self):
#         """Sort the triplets (X, y, dt) chronologically by dt."""
#         triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
#         X_sorted, y_sorted, dt_sorted = zip(*triplets)
#         self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

#     def reduce_triplets(self):
#         """Performs the reduction of triplets based on n_seconds and n_overlapping_seconds."""
#         self.sort_triplets()
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt = [], [], []
#         i = 0
#         while i < len(self.X):
#             # Define the current group range
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
            
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

#             # Take the lowest y value, and apply join_higher_classes if needed
#             min_y = min(group_y)
#             if self.join_higher_classes:
#                 min_y = 1 if min_y >= 1 else 0

#             # Take the oldest datetime
#             oldest_dt = min(group_dt)

#             # Store the reduced triplet
#             reduced_X.append(avg_X)
#             reduced_y.append(min_y)
#             reduced_dt.append(oldest_dt)

#             # Move the index by the group size minus overlap
#             i += n_samples_per_group - n_overlap_samples

#         # Convert the reduced X and y to numpy arrays
#         return np.array(reduced_X), np.array(reduced_y), reduced_dt



# class TripletReducer:
#     def __init__(self, X, y, dt, n_seconds, n_overlapping_seconds=None, 
#                   join_higher_classes=True, average_signals=False, 
#                   apply_log=True, epsilon=1e-10):
#         self.X = X
#         self.y = y
#         self.dt = dt
#         self.n_seconds = n_seconds
#         self.n_overlapping_seconds = n_overlapping_seconds
#         self.join_higher_classes = join_higher_classes
#         self.average_signals = average_signals
#         self.apply_log = apply_log
#         self.epsilon = epsilon

#     def sort_triplets(self):
#         triplets = sorted(zip(self.X, self.y, self.dt), key=lambda t: t[2])
#         X_sorted, y_sorted, dt_sorted = zip(*triplets)
#         self.X, self.y, self.dt = list(X_sorted), list(y_sorted), list(dt_sorted)

#     def reduce_triplets(self):
#         self.sort_triplets()
#         n_samples_per_group = self.n_seconds // 10
#         n_overlap_samples = (self.n_overlapping_seconds // 10) if self.n_overlapping_seconds else 0

#         reduced_X, reduced_y, reduced_dt = [], [], []
#         i = 0
#         while i < len(self.X):
#             group_X = self.X[i:i + n_samples_per_group]
#             group_y = self.y[i:i + n_samples_per_group]
#             group_dt = self.dt[i:i + n_samples_per_group]
#             if len(group_X) < n_samples_per_group:
#                 break
#             avg_X = np.mean(group_X, axis=0)
#             if self.average_signals:
#                 avg_X = np.mean(avg_X, axis=0)
#             if self.apply_log:
#                 avg_X = np.log(np.maximum(avg_X, self.epsilon))
#             min_y = min(group_y)
#             if self.join_higher_classes:
#                 min_y = 1 if min_y >= 1 else 0
#             oldest_dt = min(group_dt)
#             reduced_X.append(avg_X)
#             reduced_y.append(min_y)
#             reduced_dt.append(oldest_dt)
#             i += n_samples_per_group - n_overlap_samples

#         return np.array(reduced_X), np.array(reduced_y), reduced_dt
    
    
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
    def __init__(self, model, X_train, X_test, y_train, y_test, cv, instance_window=None, dt_train=None, dt_test=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.instance_window = instance_window
        self.dt_train = dt_train
        self.dt_test = dt_test

        if self.instance_window is not None and self.instance_window > 1:
            # Sort the training and test data by their respective datetime objects if provided
            if self.dt_train is not None:
                sorted_indices_train = np.argsort(self.dt_train)
                self.X_train = self.X_train[sorted_indices_train]
                self.y_train = self.y_train[sorted_indices_train]
                self.dt_train = self.dt_train[sorted_indices_train]

            if self.dt_test is not None:
                sorted_indices_test = np.argsort(self.dt_test)
                self.X_test = self.X_test[sorted_indices_test]
                self.y_test = self.y_test[sorted_indices_test]
                self.dt_test = self.dt_test[sorted_indices_test]

    def _get_majority_vote(self, values):
        """Helper function to calculate the majority vote in a given list or array of values."""
        return max(set(values), key=values.count)

    def _apply_windowing(self, X, y):
        """Apply windowing to the X and y data with instance_window."""
        X_windowed, y_windowed = [], []
        for i in range(0, len(X) - self.instance_window + 1, self.instance_window):
            X_window = X[i:i + self.instance_window]
            y_window = y[i:i + self.instance_window]
            X_windowed.append(np.mean(X_window, axis=0))  # Average the features within the window
            y_windowed.append(self._get_majority_vote(list(y_window)))  # Majority vote for labels
        return np.array(X_windowed), np.array(y_windowed)

    def evaluate_on_test_set(self):
        """
        Evaluates the model on the test set and prints accuracy, AUC, classification report, and confusion matrix.
        Also returns these metrics for further use.
        """
        if self.instance_window and self.instance_window > 1:
            X_test_windowed, y_test_windowed = self._apply_windowing(self.X_test, self.y_test)
            y_pred = self.model.predict(X_test_windowed)
            y_pred_proba = self.model.predict_proba(X_test_windowed)[:, 1]
        else:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_test_windowed = self.y_test  # In case no windowing is applied, the original labels are used

        accuracy = accuracy_score(y_test_windowed, y_pred)
        auc = roc_auc_score(y_test_windowed, y_pred_proba)
        report = classification_report(y_test_windowed, y_pred, output_dict=True)  # Return as dict for programmatic use
        confusion = confusion_matrix(y_test_windowed, y_pred)

        # Print the evaluation summary
        print("\nTest Set Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_windowed, y_pred))
        print("\nConfusion Matrix:")
        pprint(confusion)
        
        # Return the evaluation results as a dictionary
        return {
            "accuracy": accuracy,
            "auc": auc,
            "classification_report": report,
            "confusion_matrix": confusion
        }

    def evaluate_with_cross_validation(self):
        """
        Performs cross-validation, prints the accuracy per fold, and returns the cross-validation scores.
        Also prints class balance per fold and cross-validation accuracy summary.
        """
        if self.instance_window and self.instance_window > 1:
            X_train_windowed, y_train_windowed = self._apply_windowing(self.X_train, self.y_train)
            cv_scores = cross_val_score(self.model, X_train_windowed, y_train_windowed, cv=self.cv, scoring='accuracy')
        else:
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv, scoring='accuracy')

        fold_balances = []

        # Print class balance by fold and gather class balances
        fold_num = 1
        print("\nClass Balance by Fold:")
        for train_idx, test_idx in self.cv.split(self.X_train, self.y_train):
            y_train_fold = self.y_train[train_idx]
            unique, counts = np.unique(y_train_fold, return_counts=True)
            class_balance = dict(zip(unique, counts))
            fold_balances.append({
                "fold": fold_num,
                "class_balance": class_balance
            })
            print(f"Fold {fold_num} class balance: {class_balance}")
            fold_num += 1
        
        # Print cross-validation results
        print("\nCross-validation Accuracy Scores:")
        pprint(cv_scores)
        print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
        
        # Return cross-validation results and class balances
        return {
            "cv_scores": cv_scores,
            "mean_cv_accuracy": np.mean(cv_scores),
            "class_balance_by_fold": fold_balances
        }

    def get_classification_report(self):
        """
        Return the classification report in a string format, useful for displaying or saving in a file.
        """
        if self.instance_window and self.instance_window > 1:
            X_test_windowed, _ = self._apply_windowing(self.X_test, self.y_test)
            report = classification_report(self.y_test, self.model.predict(X_test_windowed))
        else:
            report = classification_report(self.y_test, self.model.predict(self.X_test))
        
        return report



# class ModelEvaluator:
#     def __init__(self, model, X_train, X_test, y_train, y_test, cv):
#         self.model = model
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.cv = cv

#     def evaluate_on_test_set(self):
#         """
#         Evaluates the model on the test set and prints accuracy, AUC, classification report, and confusion matrix.
#         Also returns these metrics for further use.
#         """
#         y_pred = self.model.predict(self.X_test)
#         y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
#         accuracy = accuracy_score(self.y_test, y_pred)
#         auc = roc_auc_score(self.y_test, y_pred_proba)
#         report = classification_report(self.y_test, y_pred, output_dict=True)  # Return as dict for programmatic use
#         confusion = confusion_matrix(self.y_test, y_pred)

#         # Print the evaluation summary
#         print("\nTest Set Evaluation:")
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"AUC: {auc:.4f}")
#         print("\nClassification Report:")
#         print(classification_report(self.y_test, y_pred))
#         print("\nConfusion Matrix:")
#         pprint(confusion)
        
#         # Return the evaluation results as a dictionary
#         return {
#             "accuracy": accuracy,
#             "auc": auc,
#             "classification_report": report,
#             "confusion_matrix": confusion
#         }

#     def evaluate_with_cross_validation(self):
#         """
#         Performs cross-validation, prints the accuracy per fold, and returns the cross-validation scores.
#         Also prints class balance per fold and cross-validation accuracy summary.
#         """
#         cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv, scoring='accuracy')
#         fold_balances = []

#         # Print class balance by fold and gather class balances
#         fold_num = 1
#         print("\nClass Balance by Fold:")
#         for train_idx, test_idx in self.cv.split(self.X_train, self.y_train):
#             y_train_fold = self.y_train[train_idx]
#             unique, counts = np.unique(y_train_fold, return_counts=True)
#             class_balance = dict(zip(unique, counts))
#             fold_balances.append({
#                 "fold": fold_num,
#                 "class_balance": class_balance
#             })
#             print(f"Fold {fold_num} class balance: {class_balance}")
#             fold_num += 1
        
#         # Print cross-validation results
#         print("\nCross-validation Accuracy Scores:")
#         pprint(cv_scores)
#         print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
        
#         # Return cross-validation results and class balances
#         return {
#             "cv_scores": cv_scores,
#             "mean_cv_accuracy": np.mean(cv_scores),
#             "class_balance_by_fold": fold_balances
#         }

#     def get_classification_report(self):
#         """
#         Return the classification report in a string format, useful for displaying or saving in a file.
#         """
#         report = classification_report(self.y_test, self.model.predict(self.X_test))
#         return report


# class ModelEvaluator:
#     def __init__(self, model, X_train, X_test, y_train, y_test, cv):
#         self.model = model
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.cv = cv

#     def evaluate_on_test_set(self):
#         y_pred = self.model.predict(self.X_test)
#         y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
#         accuracy = accuracy_score(self.y_test, y_pred)
#         auc = roc_auc_score(self.y_test, y_pred_proba)
#         report = classification_report(self.y_test, y_pred)
#         confusion = confusion_matrix(self.y_test, y_pred)
#         print("\nTest Set Evaluation:")
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"AUC: {auc:.4f}")
#         print("\nClassification Report:")
#         print(report)
#         print("\nConfusion Matrix:")
#         pprint(confusion)
#         return accuracy, auc, report, confusion

#     def evaluate_with_cross_validation(self):
#         cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv, scoring='accuracy')
#         fold_num = 1
#         print("\nClass Balance by Fold:")
#         for train_idx, test_idx in self.cv.split(self.X_train, self.y_train):
#             y_train_fold = self.y_train[train_idx]
#             unique, counts = np.unique(y_train_fold, return_counts=True)
#             class_balance = dict(zip(unique, counts))
#             print(f"Fold {fold_num} class balance: {class_balance}")
#             fold_num += 1
#         print("\nCross-validation Accuracy Scores:")
#         pprint(cv_scores)
#         print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
#         return cv_scores

#     def get_classification_report(self):
#         """
#         Return the classification report in a string format, useful for displaying or saving in a file.
#         """
#         report_str = classification_report(self.y_test, self.model.predict(self.X_test))
#         return report_str


class ModelRegressionEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test, cv):
        """
        Initialize the evaluator with the model and data.
        - model: The regression model (e.g., XGBRegressor).
        - X_train, X_test, y_train, y_test: Training and test data.
        - cv: Cross-validation splitter.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv

    def evaluate_on_test_set(self):
        """Evaluate the model on the test set, returning regression metrics."""
        y_pred = self.model.predict(self.X_test)

        # Regression metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)

        print("\nTest Set Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

        return mse, rmse, r2

    def evaluate_with_cross_validation(self):
        """Evaluate the model using cross-validation on the training set."""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self.cv, scoring='neg_mean_squared_error')
        
        # Convert negative MSE scores to positive
        mse_scores = -scores
        rmse_scores = np.sqrt(mse_scores)

        print("\nCross-validation MSE Scores:", mse_scores)
        print("Cross-validation RMSE Scores:", rmse_scores)
        print(f"\nMean CV RMSE: {np.mean(rmse_scores):.4f}")
        return rmse_scores

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


# class ShipLabeler:
#     def __init__(self, data):
#         """
#         Initialize with the data.
#         - data: A list of lists, where each sublist contains dictionaries with 'mmsi', 'distance', and other information.
#         """
#         self.data = data

#     def _get_closest_distance(self, ships):
#         """Returns the minimum distance from a list of ship dictionaries."""
#         return min(ship['distance'] for ship in ships)

#     def _get_closest_distance_with_mmsi(self, ships, valid_mmsi):
#         """Returns the minimum distance from ships whose 'mmsi' is in the valid_mmsi list."""
#         filtered_ships = [ship for ship in ships if ship['mmsi'] in valid_mmsi]
#         if not filtered_ships:
#             return float('inf')  # No valid ships, return a large number
#         return min(ship['distance'] for ship in filtered_ships)

#     def _get_n_ships_below_threshold(self, ships, threshold):
#         """Returns the number of ships with a distance less than or equal to the threshold."""
#         return sum(1 for ship in ships if ship['distance'] <= threshold)

#     def _get_n_ships_below_threshold_with_mmsi(self, ships, threshold, valid_mmsi):
#         """Returns the number of ships with a distance less than or equal to the threshold and valid MMSI."""
#         return sum(1 for ship in ships if ship['distance'] <= threshold and ship['mmsi'] in valid_mmsi)

#     def label_by_closest_distance(self, thresholds):
#         """
#         Label each list of ship dictionaries according to the closest distance and given thresholds.
#         - thresholds: List of distance thresholds.
#         """
#         labels = []
#         for ships in self.data:
#             closest_distance = self._get_closest_distance(ships)
#             label = self._assign_label_by_threshold(closest_distance, thresholds)
#             labels.append(label)
#         return labels

#     def label_by_closest_distance_with_mmsi(self, thresholds, valid_mmsi):
#         """
#         Label each list of ship dictionaries according to the closest distance for ships in valid_mmsi.
#         - thresholds: List of distance thresholds.
#         - valid_mmsi: List of valid MMSI numbers to consider.
#         """
#         labels = []
#         for ships in self.data:
#             closest_distance = self._get_closest_distance_with_mmsi(ships, valid_mmsi)
#             label = self._assign_label_by_threshold(closest_distance, thresholds)
#             labels.append(label)
#         return labels

#     def label_by_n_ships_below_threshold(self, thresholds, n):
#         """
#         Label each list of ship dictionaries according to whether at least n ships are below the given thresholds.
#         - thresholds: List of distance thresholds.
#         - n: Number of ships required to be below each threshold.
#         """
#         labels = []
#         for ships in self.data:
#             for i, threshold in enumerate(thresholds):
#                 if self._get_n_ships_below_threshold(ships, threshold) >= n:
#                     labels.append(i)
#                     break
#             else:
#                 labels.append(len(thresholds))
#         return labels

#     def label_by_n_ships_below_threshold_with_mmsi(self, thresholds, n, valid_mmsi):
#         """
#         Label each list of ship dictionaries according to whether at least n ships with valid MMSI are below the thresholds.
#         - thresholds: List of distance thresholds.
#         - n: Number of ships required to be below each threshold.
#         - valid_mmsi: List of valid MMSI numbers to consider.
#         """
#         labels = []
#         for ships in self.data:
#             for i, threshold in enumerate(thresholds):
#                 if self._get_n_ships_below_threshold_with_mmsi(ships, threshold, valid_mmsi) >= n:
#                     labels.append(i)
#                     break
#             else:
#                 labels.append(len(thresholds))
#         return labels

#     def _assign_label_by_threshold(self, distance, thresholds):
#         """Assign a label based on the distance and the threshold list."""
#         for i, threshold in enumerate(thresholds):
#             if distance <= threshold:
#                 return i
#         return len(thresholds)  # If the distance exceeds all thresholds, assign the highest label

# # Example data
# data = [
#     [{'mmsi': 310750000, 'closest_latitude': 51.406866666666666, 'closest_longitude': 3.0789966666666664, 'distance': 41.88},
#      {'mmsi': 245618000, 'closest_latitude': 51.397539, 'closest_longitude': 3.11135, 'distance': 2399.88}],
#     [{'mmsi': 257724000, 'closest_latitude': 51.399299, 'closest_longitude': 3.128926, 'distance': 3541.93}]
# ]

# # Create an instance of ShipLabeler
# labeler = ShipLabeler(data)

# # Case 1: Label by closest distance
# thresholds = [1000, 2000]
# labels = labeler.label_by_closest_distance(thresholds)
# print(labels)  # Output: [0, 2]

# # Case 2: Label by closest distance for specific MMSI
# valid_mmsi = [310750000, 245618000]
# labels = labeler.label_by_closest_distance_with_mmsi(thresholds, valid_mmsi)
# print(labels)

# # Case 3: Label by n ships below threshold
# labels = labeler.label_by_n_ships_below_threshold([1000, 2000], 2)
# print(labels)

# # Case 4: Label by n ships below threshold with MMSI
# labels = labeler.label_by_n_ships_below_threshold_with_mmsi([1000, 2000], 2, valid_mmsi)
# print(labels)


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


# class ShipDistanceTargetGenerator:
#     def __init__(self, data, target_method='average', saturation_threshold=None):
#         """
#         Initialize with the data, target method, and saturation threshold.
#         - data: A list of lists, where each sublist contains dictionaries with 'mmsi', 'distance', and other information.
#         - target_method: Method to determine the target ('minimum', 'average', or 'median').
#         - saturation_threshold: Maximum distance threshold, beyond which values are clipped.
#         """
#         self.data = data
#         self.target_method = target_method.lower()  # Ensure it's lowercase for consistency
#         self.saturation_threshold = saturation_threshold  # Optional threshold for clipping

#     def generate_targets(self):
#         """
#         Generate targets for regression based on the specified target method and saturation threshold.
#         Returns a list of distances representing the calculated target for each element in the data.
#         """
#         targets = []
#         for ships in self.data:
#             target_distance = self._get_target_distance(ships)
#             # Apply saturation threshold if specified
#             if self.saturation_threshold is not None:
#                 target_distance = min(target_distance, self.saturation_threshold)
#             targets.append(target_distance)
#         return targets

#     def _get_target_distance(self, ships):
#         """Returns the target distance based on the specified method (minimum, average, or median)."""
#         distances = np.array([ship['distance'] for ship in ships])

#         # Filter out invalid values (inf, NaN)
#         valid_distances = distances[np.isfinite(distances)]

#         # Handle the case where all values are invalid
#         if len(valid_distances) == 0:
#             return np.nan  # Return NaN if no valid distances are available

#         # Apply the specified target method
#         if self.target_method == 'minimum':
#             return np.min(valid_distances)
#         elif self.target_method == 'median':
#             return np.median(valid_distances)
#         else:  # Default is 'average'
#             return np.mean(valid_distances)


# class ShipDistanceTargetGenerator:
#     def __init__(self, data, target_method='average'):
#         """
#         Initialize with the data and target method.
#         - data: A list of lists, where each sublist contains dictionaries with 'mmsi', 'distance', and other information.
#         - target_method: Method to determine the target ('minimum', 'average', or 'median').
#         """
#         self.data = data
#         self.target_method = target_method.lower()  # Ensure it's lowercase for consistency

#     def generate_targets(self):
#         """
#         Generate targets for regression based on the specified target method.
#         Returns a list of distances representing the calculated target for each element in the data.
#         """
#         targets = []
#         for ships in self.data:
#             target_distance = self._get_target_distance(ships)
#             targets.append(target_distance)
#         return targets

#     def _get_target_distance(self, ships):
#         """Returns the target distance based on the specified method (minimum, average, or median)."""
#         distances = np.array([ship['distance'] for ship in ships])

#         # Filter out invalid values (inf, NaN)
#         valid_distances = distances[np.isfinite(distances)]

#         # Handle the case where all values are invalid
#         if len(valid_distances) == 0:
#             return np.nan  # Return NaN if no valid distances are available

#         # Apply the specified target method
#         if self.target_method == 'minimum':
#             return np.min(valid_distances)
#         elif self.target_method == 'median':
#             return np.median(valid_distances)
#         else:  # Default is 'average'
#             return np.mean(valid_distances)

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
        dt_res = np.copy(self.dt)  # Start with a copy of the original datetime array
        
        
        # Extend the datetime array with modified datetimes of the original closest samples
        for i in range(len(self.y), len(y_res)):
            # Randomly pick a datetime from the original datetimes to modify
            original_datetime = np.random.choice(self.dt)
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
        dt_res = np.copy(self.dt)  # Start with a copy of the original datetime array
        for i in range(len(self.y), len(y_res)):
            original_datetime = np.random.choice(self.dt)
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
    

# class ShipImpactAnalyzer:
#     def __init__(self, X_reduced, y_reduced, dt_reduced, ships_reduced, classifier):
#         """
#         Initialize with the reduced data and the classifier.
#         - X_reduced: Reduced feature set (list or array).
#         - y_reduced: Ground truth labels.
#         - dt_reduced: Reduced list of datetime objects.
#         - ships_reduced: Reduced list of ship dictionaries for each instance.
#         - classifier: A trained classifier (e.g., XGBClassifier).
#         """
#         self.X_reduced = X_reduced
#         self.y_reduced = y_reduced
#         self.dt_reduced = dt_reduced
#         self.ships_reduced = ships_reduced
#         self.classifier = classifier

#     def analyze(self):
#         """
#         Analyze the classifier's performance and track which ships (mmsi) caused the most TPs, TNs, FPs, and FNs.
#         Returns a dictionary containing lists of ships that contributed to each type of result, ordered by frequency.
#         """
#         # Step 1: Make predictions
#         y_pred = self.classifier.predict(self.X_reduced)

#         # Step 2: Initialize result containers
#         ship_stats = {
#             'true_positives': {},
#             'true_negatives': {},
#             'false_positives': {},
#             'false_negatives': {}
#         }

#         # Step 3: Track the ships responsible for each outcome
#         for i, (pred, actual, dt, ships) in enumerate(zip(y_pred, self.y_reduced, self.dt_reduced, self.ships_reduced)):
#             # Get the nearest ship (shortest distance) from ships_reduced
#             nearest_ship = min(ships, key=lambda ship: ship['distance'])
#             mmsi = nearest_ship['mmsi']

#             # Determine if this is a TP, TN, FP, or FN
#             if pred == 1 and actual == 1:
#                 category = 'true_positives'
#             elif pred == 0 and actual == 0:
#                 category = 'true_negatives'
#             elif pred == 1 and actual == 0:
#                 category = 'false_positives'
#             else:
#                 category = 'false_negatives'

#             # Track the mmsi and the datetime of the event
#             if mmsi not in ship_stats[category]:
#                 ship_stats[category][mmsi] = {'count': 0, 'times': []}
#             ship_stats[category][mmsi]['count'] += 1
#             ship_stats[category][mmsi]['times'].append(dt)

#         # Step 4: Sort the ships by frequency in each category
#         sorted_stats = {
#             category: sorted(mmsi_info.items(), key=lambda item: item[1]['count'], reverse=True)
#             for category, mmsi_info in ship_stats.items()
#         }

#         return sorted_stats

#     def print_results(self, sorted_stats):
#         """
#         Print the sorted results in a human-readable format.
#         """
#         for category, ships in sorted_stats.items():
#             print(f"\n{category.replace('_', ' ').capitalize()}:")
#             for mmsi, info in ships:
#                 times_str = ', '.join(str(dt) for dt in info['times'])
#                 print(f"MMSI: {mmsi}, Count: {info['count']}, Times: {times_str}")


# Example usage
# classifier = xgb.XGBClassifier()  # Assume this is already trained
# analyzer = ShipImpactAnalyzer(X_reduced, y_reduced, dt_reduced, ships_reduced, classifier)

# # Analyze the impact of ships on the classifier's performance
# sorted_stats = analyzer.analyze()

# # Print the results
# analyzer.print_results(sorted_stats)

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

        print(f"Results successfully saved to {file_name}")


# class ShipImpactAnalyzer:
#     def __init__(self, X_reduced, y_reduced, dt_reduced, ships_reduced, classifier):
#         self.X_reduced = X_reduced
#         self.y_reduced = y_reduced
#         self.dt_reduced = dt_reduced
#         self.ships_reduced = ships_reduced
#         self.classifier = classifier

#     def analyze(self):
#         y_pred = self.classifier.predict(self.X_reduced)

#         ship_stats = {
#             'true_positives': {},
#             'true_negatives': {},
#             'false_positives': {},
#             'false_negatives': {}
#         }

#         for i, (pred, actual, dt, ships) in enumerate(zip(y_pred, self.y_reduced, self.dt_reduced, self.ships_reduced)):
#             nearest_ship = min(ships, key=lambda ship: ship['distance'])
#             mmsi = nearest_ship['mmsi']

#             if pred == 1 and actual == 1:
#                 category = 'true_positives'
#             elif pred == 0 and actual == 0:
#                 category = 'true_negatives'
#             elif pred == 1 and actual == 0:
#                 category = 'false_positives'
#             else:
#                 category = 'false_negatives'

#             if mmsi not in ship_stats[category]:
#                 ship_stats[category][mmsi] = {'count': 0, 'times': []}
#             ship_stats[category][mmsi]['count'] += 1
#             ship_stats[category][mmsi]['times'].append(dt)

#         sorted_stats = {
#             category: sorted(mmsi_info.items(), key=lambda item: item[1]['count'], reverse=True)
#             for category, mmsi_info in ship_stats.items()
#         }

#         return sorted_stats

#     def save_to_csv(self, sorted_stats, file_name):
#         """
#         Save the sorted results to a CSV file.
#         Each row will correspond to an MMSI, and each column to a category (TP, TN, FP, FN).
#         """
#         # Create a dictionary to hold all MMSI ships and their respective category counts
#         all_mmsi_stats = {}

#         # Iterate over each category and populate the dictionary
#         for category, ships in sorted_stats.items():
#             for mmsi, info in ships:
#                 if mmsi not in all_mmsi_stats:
#                     all_mmsi_stats[mmsi] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
                
#                 if category == 'true_positives':
#                     all_mmsi_stats[mmsi]['TP'] = info['count']
#                 elif category == 'true_negatives':
#                     all_mmsi_stats[mmsi]['TN'] = info['count']
#                 elif category == 'false_positives':
#                     all_mmsi_stats[mmsi]['FP'] = info['count']
#                 elif category == 'false_negatives':
#                     all_mmsi_stats[mmsi]['FN'] = info['count']

#         # Write the output to a CSV file
#         with open(file_name, mode='w', newline='') as file:
#             writer = csv.writer(file)
            
#             # Write header
#             writer.writerow(["MMSI", "True Positives", "True Negatives", "False Positives", "False Negatives"])

#             # Write each ship's results
#             for mmsi, counts in all_mmsi_stats.items():
#                 writer.writerow([mmsi, counts['TP'], counts['TN'], counts['FP'], counts['FN']])

#         print(f"Results successfully saved to {file_name}")

# Example usage:
# Assuming the classifier is already trained
# classifier = xgb.XGBClassifier()

# # Create an instance of ShipImpactAnalyzer
# analyzer = ShipImpactAnalyzer(X_reduced, y_reduced, dt_reduced, ships_reduced, classifier)

# # Analyze the impact of ships
# sorted_stats = analyzer.analyze()

# # Save the results to a CSV file
# analyzer.save_to_csv(sorted_stats, 'ship_impact_analysis.csv')

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

        print(f"Metrics successfully saved to {file_name}")

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



# class VesselListReducerByDatetime:
#     def __init__(self, vessel_lists, dt):
#         """
#         Initialize with vessel lists and a smaller list of datetimes.
#         - vessel_lists: A list of lists, where each sublist contains dictionaries representing ships with a 'datetime' key.
#         - dt: A smaller list of utc-aware datetime objects used for reducing vessel_lists.
#         """
#         self.vessel_lists = vessel_lists
#         self.dt = dt

#     def reduce_by_datetimes(self):
#         """
#         Reduce vessel_lists by retaining only the ships whose 'datetime' matches any in dt.
#         The order of the reduced vessel lists will follow the order of dt.
#         """
#         reduced_vessel_lists = []

#         # Iterate over the target datetimes
#         for target_datetime in self.dt:
#             # For each target datetime, find matching ships from the vessel_lists
#             matching_vessels = []
#             for vessels in self.vessel_lists:
#                 for ship in vessels:
#                     if ship['datetime'] == target_datetime:
#                         matching_vessels.append(ship)
#             # Append the matching vessels for the current datetime (if any)
#             if matching_vessels:
#                 reduced_vessel_lists.append(matching_vessels)

#         return reduced_vessel_lists


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

# # Usage Example:

# X = np.random.rand(100, 50, 100)  # Example features
# y = np.random.randint(0, 2, 100)  # Example binary labels
# dt = [datetime(2023, 6, 22, 0, 0) + timedelta(seconds=i*10) for i in range(100)]  # Datetime every 10 seconds
# closest_distances_list = np.random.rand(100, 10)  # Example closest distances list

# # Initialize the applier
# applier = TimeOffsetApplier(X, y, dt, closest_distances_list, time_offset_seconds=7200)

# # Apply the time offset
# applier.apply_time_offset()

# # Get the reduced data
# X_reduced, y_reduced, dt_reduced, closest_distances_reduced = applier.get_data()
