#!/usr/bin/env python3
"""
read_and_split.py

This script loads HDF5 datasets (X, y, datetimes, ship_info, etc.) from a given file
and splits X and y into training and testing sets based on a specified ISO date.
All instances whose datetimes fall on the specified date are reserved for testing;
the rest are used for training.

Usage:
    python read_and_split.py --h5_path /path/to/dataset_sensor_range_1440_1690.h5 --test_date YYYY-MM-DD

Dependencies:
    - h5py
    - numpy
    - pandas

Outputs:
    - Prints shapes of loaded arrays and resulting train/test splits.
    - Saves NumPy .npz files with train/test splits:
        * X_train.npy, X_test.npy
        * y_train.npy, y_test.npy
        * datetimes_train.npy, datetimes_test.npy
        * (optional) ship_info_train.npz, ship_info_test.npz  (if ship_info is needed downstream)
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load HDF5 datasets and split into train/test by date."
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file (e.g., dataset_sensor_range_1440_1690.h5)."
    )
    parser.add_argument(
        "--test_date",
        type=str,
        required=True,
        help=(
            "ISO-format date (YYYY-MM-DD) to use as the test set. "
            "All rows whose datetimes fall on this date (UTC) will be held out."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where train/test splits (.npy/.npz) will be saved."
    )
    return parser.parse_args()

def validate_date(date_str):
    """
    Validate that date_str follows YYYY-MM-DD and is a real date.
    Returns a pandas.Timestamp (date component).
    """
    try:
        ts = pd.to_datetime(date_str, utc=True)
        # Extract date (drop time)
        return ts.normalize().date()
    except Exception as e:
        raise ValueError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.") from e

def load_hdf5_datasets(h5_path):
    """
    Load datasets from the HDF5 file into memory.
    Returns a dict with keys: 'X', 'y', 'datetimes', and optionally 'ship_info'.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at: {h5_path}")

    data = {}
    with h5py.File(h5_path, "r") as f:
        # Required datasets
        if "X" not in f or "y" not in f or "datetimes" not in f:
            raise KeyError("HDF5 file must contain 'X', 'y', and 'datetimes' datasets.")

        # Load X, y, datetimes fully into memory
        data["X"] = f["X"][()]  # shape: (N, 250, 100)
        data["y"] = f["y"][()]  # shape: (N,)
        # datetimes are stored as bytes (e.g. |S26). Convert to str
        raw_datetimes = f["datetimes"][()]
        if raw_datetimes.dtype.kind == "S":
            data["datetimes"] = raw_datetimes.astype("U")  # convert bytes to unicode
        else:
            data["datetimes"] = raw_datetimes.astype(str)

        # Optionally load ship_info group if downstream needed
        if "ship_info" in f:
            ship_info_group = f["ship_info"]
            ship_info = {}
            for key in ship_info_group.keys():
                ship_info[key] = ship_info_group[key][()]
            data["ship_info"] = ship_info

    return data

def split_by_date(datetimes_array, test_date):
    """
    Given an array of ISO-format datetime strings and a test_date (datetime.date),
    return boolean masks for train/test.
    """
    # Parse ISO strings into pandas.DatetimeIndex (UTC-aware)
    # pandas.to_datetime can parse an array of strings at once
    dt_index = pd.to_datetime(datetimes_array, utc=True)

    # Extract date component (UTC)
    dates = dt_index.date  # ndarray of datetime.date

    # Create boolean mask: True for test (date == test_date), False otherwise
    test_mask = (dates == test_date)
    train_mask = ~test_mask

    return train_mask, test_mask

def save_splits(output_dir, X_train, X_test, y_train, y_test,
                datetimes_train, datetimes_test, ship_info_train=None, ship_info_test=None):
    """
    Save train/test splits to .npy or .npz files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "datetimes_train.npy"), datetimes_train)
    np.save(os.path.join(output_dir, "datetimes_test.npy"), datetimes_test)

    if ship_info_train is not None and ship_info_test is not None:
        # Save ship_info as a .npz archive with named arrays
        np.savez(
            os.path.join(output_dir, "ship_info_train.npz"),
            **ship_info_train
        )
        np.savez(
            os.path.join(output_dir, "ship_info_test.npz"),
            **ship_info_test
        )

def main():
    args = parse_args()

    # Validate and normalize test_date
    try:
        test_date_obj = validate_date(args.test_date)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load datasets
    try:
        data = load_hdf5_datasets(args.h5_path)
    except Exception as e:
        print(f"Error loading HDF5: {e}", file=sys.stderr)
        sys.exit(1)

    X = data["X"]
    y = data["y"]
    datetimes = data["datetimes"]
    ship_info = data.get("ship_info", None)

    N = X.shape[0]
    if y.shape[0] != N or datetimes.shape[0] != N:
        print("Error: 'X', 'y', and 'datetimes' must have the same length.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded data: X.shape = {X.shape}, y.shape = {y.shape}, datetimes.shape = {datetimes.shape}")
    if ship_info is not None:
        print(f"Loaded ship_info keys: {list(ship_info.keys())}")

    # Split by date
    train_mask, test_mask = split_by_date(datetimes, test_date_obj)

    n_test = test_mask.sum()
    n_train = train_mask.sum()
    if n_test == 0:
        print(f"Warning: No samples found for test_date {test_date_obj}.", file=sys.stderr)

    print(f"Splitting data: {n_train} training samples, {n_test} testing samples.")

    # Apply masks
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    datetimes_train = datetimes[train_mask]
    datetimes_test = datetimes[test_mask]

    ship_info_train = None
    ship_info_test = None
    if ship_info is not None:
        ship_info_train = {}
        ship_info_test = {}
        for key, arr in ship_info.items():
            ship_info_train[key] = arr[train_mask]
            ship_info_test[key] = arr[test_mask]

    # Save splits
    save_splits(
        args.output_dir,
        X_train, X_test,
        y_train, y_test,
        datetimes_train, datetimes_test,
        ship_info_train, ship_info_test
    )

    print(f"Train/test splits saved in directory: {args.output_dir}")

if __name__ == "__main__":
    main()
