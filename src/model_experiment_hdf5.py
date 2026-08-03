import argparse
from datetime import datetime, timezone
import json
import logging
import numpy as np
from pathlib import Path
import re
import shlex
import sys
import uuid

from data_splitter_hdf5 import TripletReducer, TripletRegressionReducer
from hdf5_data_loader import HDF5DataLoader
from model_experiment_hdf5_base import PipelineExecutorHDF5
# from model_experiment_hdf5_base import FlexibleModelExecutor  # To be implemented or adapted

logging.addLevelName(logging.INFO, "INF")
logging.addLevelName(logging.WARNING, "WRN")
logging.addLevelName(logging.ERROR, "ERR")
logger = logging.getLogger(__name__)


def _filename_component(value):
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return component or "unknown"


def prepare_execution_paths(args):
    """Create the shared local/MLflow identity and resolve output paths."""
    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    random_suffix = uuid.uuid4().hex[:8]
    execution_id = f"{timestamp}-{random_suffix}"

    if args.run_name:
        run_component = _filename_component(args.run_name)
    elif args.joblib_save_file:
        # Existing scripts already carry a useful human-readable result stem.
        # Reuse it so their logs and MLflow run names remain easy to correlate.
        run_component = _filename_component(Path(args.joblib_save_file).stem)
    else:
        task = "regression" if args.is_regression == "true" else "classification"
        model_name = Path(args.model_file).stem
        if args.test_date_end:
            scope = f"folds-{args.test_date_start}-{args.test_date_end}"
        else:
            scope = f"single-{args.test_date_start}"
        run_component = _filename_component(f"{model_name}-{task}-{scope}")

    experiment_component = _filename_component(args.mlflow_experiment_name)
    execution_name = f"{run_component}-{execution_id}"

    if args.joblib_save_file is None:
        result_path = (
            Path(args.results_dir).expanduser()
            / experiment_component
            / execution_name
            / "metrics.joblib"
        )
        args.joblib_save_file = str(result_path.resolve())
        args.joblib_save_file_overridden = False
    else:
        result_path = Path(args.joblib_save_file).expanduser()
        args.joblib_save_file = str(result_path.resolve())
        args.joblib_save_file_overridden = True
    result_path.parent.mkdir(parents=True, exist_ok=True)

    args.run_name = run_component
    args.mlflow_run_name = execution_name
    args.execution_id = execution_id
    return execution_id, experiment_component, execution_name


def configure_logging(args):
    """Configure the execution log using the shared run identity."""
    execution_id, experiment_component, execution_name = prepare_execution_paths(args)

    if args.log_file is not None:
        log_path = Path(args.log_file).expanduser()
    else:
        log_path = (
            Path(args.log_dir).expanduser()
            / experiment_component
            / execution_name
            / "execution.log"
        )

    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing log file: {log_path}. "
            "Choose another --log_file path."
        )

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return execution_id, log_path


def format_array_for_log(arr, name="array", sample_size=5):
    """
    Generate a concise summary string for numpy arrays or sequences for logging.
    """
    import numpy as _np

    if not isinstance(arr, _np.ndarray):
        if hasattr(arr, '__len__') and len(arr) > 0:
            return f"{name} sample ({len(arr)} items): {str(list(arr)[:sample_size])}"
        return f"{name}: {str(arr)}"

    log_str = f"{name} shape: {arr.shape}, dtype: {arr.dtype}\n"
    if arr.ndim == 1:
        log_str += f"{name} sample: {_np.array2string(arr[:sample_size], precision=5, separator=', ')}\n"
    elif arr.ndim == 2:
        log_str += f"{name} sample (first row): {_np.array2string(arr[0, :sample_size], precision=5, separator=', ')}\n"
    elif arr.ndim == 3:
        log_str += f"{name} sample (first item, first feature_vector): {_np.array2string(arr[0, 0, :sample_size], precision=5, separator=', ')}\n"

    if hasattr(arr, 'size') and arr.size > 0 and (_np.issubdtype(arr.dtype, _np.number) or arr.dtype == object):
        try:
            if _np.issubdtype(arr.dtype, _np.number) and not _np.all(_np.isfinite(arr)):
                log_str += f"{name} contains non-finite values (NaN/inf).\n"
            if arr.ndim == 1 and arr.size < 100000 and len(_np.unique(arr)) < 20:
                unique_elements, counts_elements = _np.unique(arr, return_counts=True)
                log_str += f"{name} unique counts: {dict(zip(unique_elements, counts_elements))}\n"
        except Exception:
            pass
    return log_str.strip()

def parse_args():
    # parser must show default values in help text
    parser = argparse.ArgumentParser(description="Run ML experiment with HDF5 vessel dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--h5_path', type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument('--model_file', type=str, required=True, help="Path to Python script with a `load_model` function")
    parser.add_argument('--is_NN', type=str, default="false", choices=["true", "false"], help="Use Neural Network?")

    # Neural Network hyperparameters (only used when --is_NN true)
    parser.add_argument('--nn_epochs', type=int, default=100, help="[NN only] Max training epochs")
    parser.add_argument('--nn_hidden_dim', type=int, default=256, help="[NN only] Hidden dimension passed to load_model()")
    parser.add_argument('--nn_batch_size', type=int, default=32, help="[NN only] Training batch size. The loader drops incomplete batches; folds smaller than this value produce no training batch")
    parser.add_argument('--nn_patience', type=int, default=20, help="[NN only] Early stopping patience measured on training loss; the NN path has no validation-loss monitor")
    parser.add_argument('--nn_lr', type=float, default=0.001, help="[INACTIVE] Recorded for NN runs but does not change the optimizer; set learning rate in the model file")

    parser.add_argument('--is_regression', type=str, default="false", choices=["true", "false"], help="Regression task?")
    parser.add_argument(
        '--classification_thresholds',
        '--thresholds',
        dest='classification_thresholds',
        type=int,
        nargs='+',
        default=None,
        help=(
            "Classification threshold(s) in meters. "
            "Specify one or more space-separated values (e.g., --classification_thresholds 1000 or --classification_thresholds 500 1000 2000). "
            "Default: 1000 meters if not specified. Evaluation is currently binary; keep --join_higher_classes true when specifying multiple thresholds."
        )
    )
    parser.add_argument(
        '--saturation_threshold', type=int, default=None,
        help=(
            "[INACTIVE] Accepted for compatibility but does not currently clip "
            "or otherwise change regression targets."
        )
    )
    parser.add_argument(
        '--regression_threshold', type=int, default=5000,
        help=(
            "Maximum allowable target distance in regression: "
            "any example with target above this value will be removed."
        )
    )
    parser.add_argument(
        '--regression_evaluation_threshold', type=float, default=None,
        help=(
            "Additionally report regression metrics for test frames whose true "
            "distance is less than or equal to this value. This does not affect "
            "training, prediction, or the existing overall metrics."
        )
    )
    parser.add_argument(
        '--instance_window', type=int, default=None,
        help="Number of consecutive windows to group at evaluation for temporal smoothing. Classification labels are voted, but probability scores are not equivalently aggregated, so smoothed AUC is not reliable."
    )
    parser.add_argument(
        '--n_seconds', type=int, default=10,
        help="Length in seconds of each feature-aggregation window."
    )
    parser.add_argument(
        '--n_overlapping_seconds', type=int, default=-10,
        help=(
            "Number of seconds each aggregation window overlaps with its predecessor. "
            "If positive, windows overlap by that many seconds. "
            "If negative, the overlap is calculated as n_seconds + n_overlapping_seconds seconds."
        )
    )
    parser.add_argument(
        '--average_signals',
        type=str,
        default='time_channel',
        choices=['none', 'time', 'channel', 'time_channel'],
        help=(
            "How to aggregate signals: none (no averaging), "
            "time (average over time per channel), "
            "channel (average over channels per time step), "
            "time_channel (average over both time and channels). "
            "Only 'channel' and 'time_channel' have been tested so far."
        )
    )
    parser.add_argument('--apply_log', type=str, default="true", choices=["true", "false"], help="Apply log scaling to features?")
    parser.add_argument('--join_higher_classes', type=str, default="true", choices=["true", "false"], help="Collapse all classification labels above zero into class 1. The current evaluator is binary, so false is unsupported when thresholds create more than two classes")
    parser.add_argument('--time_offset_seconds', type=int, default=None, help="Shift labels by this many seconds")
    parser.add_argument('--balance_test', type=str, default="false", choices=["true", "false"], help="[INACTIVE] Accepted for compatibility but the active pipeline does not balance the test set")
    parser.add_argument('--balance_classes', type=str, default='unbalanced', choices=['unbalanced', 'smote', 'adasyn', 'naive', 'undersample'], help="Class balancing method")
    parser.add_argument('--use_mid_target', type=str, default="true", choices=["true", "false"], help="Use midpoint as target in reduction?")
    parser.add_argument(
        '--regression_target_method',
        choices=['legacy', 'central_t', 'first_t', 'last_t', 'min', 'mean', 'median'],
        default=None,
        help=(
            "Regression target and representative timestamp calculation. "
            "When omitted, --use_mid_target true selects legacy and false selects min."
        ),
    )
    parser.add_argument(
        '--test_date_start',
        type=str,
        default="2023-06-16",
        help="Start date for testing (YYYY-MM-DD). Without --test_date_end, single-day mode produces Joblib/log output but not the normal CSV or main MLflow metric/artifact set."
    )
    parser.add_argument(
        '--test_date_end',
        type=str,
        default=None,
        help="End date for testing (YYYY-MM-DD). With --test_date_start, runs one fold per date in the inclusive range; only the last fold model is persisted."
    )
    parser.add_argument(
        '--joblib_save_file',
        type=str,
        default=None,
        help=(
            "Exact path for the results joblib. If omitted, results are saved "
            "under results/<MLflow experiment>/<run name>-<execution ID>/metrics.joblib."
        ),
    )
    parser.add_argument('--output_suffix', type=str, default='', help="Suffix for output files")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for splitting, balancing, NN training, and most bootstraps. Classification confusion-matrix bootstrap still uses a fixed seed of 42")
    parser.add_argument('--perform_grid_search', type=str, default="false", choices=["true", "false"], help="[INACTIVE] Accepted but the active dynamically loaded model path does not run grid search")
    parser.add_argument('--param_grid', type=str, default=None, help="[INACTIVE] Grid-search JSON retained for compatibility; it is not used by the active model path")
    parser.add_argument('--model_output_suffix', type=str, default='', help="[INACTIVE] Accepted but does not change the active trained-model artifact path")
    parser.add_argument(
        '--mlflow_experiment_name',
        type=str,
        default='Marlinks-NS-DAS-dataset',
        help="MLflow experiment name and automatic local artifact directory stem.",
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help=(
            "Human-readable run label. The MLflow run and automatic local "
            "directories use <run name>-<execution ID>."
        ),
    )
    parser.add_argument(
        '--mlflow_tracking_uri',
        type=str,
        default='sqlite:///mlflow.db',
        help=(
            "MLflow tracking URI. Defaults to the repository-local SQLite "
            "database; may also be a remote URI such as http://127.0.0.1:5000."
        ),
    )

    parser.add_argument(
        '--reduce_to_size',
        type=int,
        default=None,
        help=(
            "Center-slice the original 2D feature map along the channel axis to this many sensors. "
            "Only channels around the central sensor are kept; default None retains all channels (250)."
        )
    )
    parser.add_argument(
        '--vessel_joblib_path',
        type=str,
        default=None,
        help="[INACTIVE] Optional speed-regression vessel joblib path retained for compatibility; unused by the active pipeline"
    )
    parser.add_argument(
        '--freq_limit_joblib',
        type=str,
        default=None,
        help=(
            "Path to a joblib file containing 'band_limits' for frequency bands. "
            "If provided, SHAP values are computed per band for train and test data."
        )
    )
    parser.add_argument(
        '--compute_daywise_bootstrap',
        type=str,
        default="false",
        choices=["true", "false"],
        help=(
            "Enable sample-level per-day bootstrap uncertainty for classification. "
            "The regression evaluator currently ignores this option; it never affects final/global confidence intervals."
        )
    )
    parser.add_argument('--skip_if_output_exists', type=str, default="true", choices=["true", "false"], help="[INACTIVE] Accepted for compatibility but the active pipeline does not check existing result files before running")

    parser.add_argument(
        '--invert_threshold_logic', type=str, default="false", choices=["true","false"],
        help="If true, class 1 = y ≤ threshold (instead of y > threshold)"
        )


    parser.add_argument(
        '--target_file', type=str, default=None,
        help="Path to an HDF5 or joblib file containing target data; supports classification, regression or multi-output. Requires --target_key."
    )
    parser.add_argument(
        '--target_key', type=str, default=None,
        help="Key name in the target_file for loading the target array (e.g. dataset name or dict key)."
    )

    parser.add_argument(
        '--center_truth', type=str, default="false", choices=["true", "false"],
        help="Use the central ground truth instead of group/window majority (classification) or mean (regression). With classification instance smoothing, probability scores are not centered or grouped equivalently."
    )

    parser.add_argument(
        '--y_min',
        type=float,
        default=None,
        help="Minimum y value for filtering (inclusive). If --y_max is not provided, filters to y >= y_min only."
    )

    parser.add_argument(
        '--y_max',
        type=float,
        default=None,
        help="Maximum y value for filtering (inclusive). If provided with --y_min, filters to y_min <= y <= y_max. If provided alone, filters to y <= y_max."
    )

    parser.add_argument(
        '--save_fold_txt', type=str, default="false", choices=["true","false"],
        help="If true, save per-day fold features, labels, and datetimes as txt files (only for channel/time_channel averaging). NOT AVAILABLE IN PUBLIC VERSION."
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help="Directory for an automatically named, unique execution log."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help="Root directory for automatically named result artifacts.",
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help="Exact execution log path. Existing files are never overwritten."
    )

    args = parser.parse_args()

    # Require both target_file and target_key if one is provided
    if (args.target_file is None) ^ (args.target_key is None):
        parser.error('Both --target_file and --target_key must be provided together')

    # Validate y_min and y_max relationship
    if args.y_min is not None and args.y_max is not None:
        if args.y_min > args.y_max:
            parser.error(f'--y_min ({args.y_min}) must be less than or equal to --y_max ({args.y_max})')
    if (
        args.regression_evaluation_threshold is not None
        and args.regression_evaluation_threshold < 0
    ):
        parser.error('--regression_evaluation_threshold must be zero or greater')
    if (
        args.regression_evaluation_threshold is not None
        and args.is_regression != "true"
    ):
        parser.error(
            '--regression_evaluation_threshold can only be used when '
            '--is_regression true'
        )

    return args

def main():
    args = parse_args()
    try:
        execution_id, log_path = configure_logging(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not configure logging: {exc}") from exc
    args.execution_id = execution_id
    args.log_file = str(log_path)

    logger.info("Execution ID: %s", execution_id)
    logger.info("MLflow experiment: %s", args.mlflow_experiment_name)
    logger.info("MLflow run name: %s", args.mlflow_run_name)
    logger.info("Log file: %s", log_path)
    logger.info("Results joblib: %s", args.joblib_save_file)
    logger.info("Command line: %s", shlex.join([sys.executable, *sys.argv]))

    # Log start and configuration
    logger.info("Starting HDF5 pipeline for %s", args.h5_path)
    logger.debug("Config: %s", json.dumps(vars(args), indent=2, default=str))

    # Convert relevant string-bools to bools
    def str2bool(v):
        return v.lower() == "true"
    config = vars(args).copy()

    # Fix types
    config['is_NN'] = str2bool(config['is_NN'])
    config['is_regression'] = str2bool(config['is_regression'])
    config['apply_log'] = str2bool(config['apply_log'])
    config['join_higher_classes'] = str2bool(config['join_higher_classes'])
    config['balance_test'] = str2bool(config['balance_test'])
    config['use_mid_target'] = str2bool(config['use_mid_target'])
    if config['is_regression'] and config['regression_target_method'] is None:
        config['regression_target_method'] = (
            'legacy' if config['use_mid_target'] else 'min'
        )
    config['perform_grid_search'] = str2bool(config['perform_grid_search'])
    config['compute_daywise_bootstrap'] = str2bool(config['compute_daywise_bootstrap'])
    config['skip_if_output_exists'] = str2bool(config['skip_if_output_exists'])
    config['invert_threshold_logic'] = str2bool(config['invert_threshold_logic'])
    config['center_truth'] = str2bool(config['center_truth'])
    config['save_fold_txt'] = str2bool(config.get('save_fold_txt', False))

    # Parse param_grid JSON if present
    if config['param_grid'] is not None:
        config['param_grid'] = json.loads(config['param_grid'])

    # Normalize classification thresholds and retain the legacy internal key.
    if config.get('classification_thresholds') is None:
        config['classification_thresholds'] = [1000]
        logger.info("Using default classification threshold: 1000 meters")
    else:
        logger.info(
            "Using classification threshold(s): %s meters",
            config['classification_thresholds'],
        )
    config['thresholds'] = config['classification_thresholds']

    # Build test_date_range from test_date_start and test_date_end for backward compatibility with base module
    if config.get('test_date_end') is not None:
        # Date range mode: [start, end]
        config['test_date_range'] = [config['test_date_start'], config['test_date_end']]
        logger.info(f"Test date range: {config['test_date_start']} to {config['test_date_end']}")
    else:
        # Single day mode: [start]
        config['test_date_range'] = [config['test_date_start']]
        logger.info(f"Single test date: {config['test_date_start']}")

    # Use correct output keys (keep legacy compatibility)
    config['output'] = config.get('joblib_save_file', config.get('output'))

    logger.info("Step 1: Load HDF5 data from %s", config['h5_path'])
    loader = HDF5DataLoader(config['h5_path'])
    try:
        X, y, datetimes = loader.load()
    except Exception as e:
        logger.error("Failed to load HDF5 data: %s", e)
        exit(1)

    # Keep original HDF5 'y' for optional filtering, regardless of override
    y_h5 = y

    # Optionally override ML target y from external file (does not affect filtering)
    if config.get('target_file') is not None:
        logger.info(f"Overriding target y from '{config['target_file']}' key '{config['target_key']}'")
        from target_loader import TargetLoader
        y = TargetLoader.load(config['target_file'], config['target_key'])
        logger.info(format_array_for_log(y, "Overridden y"))

    logger.info("Loaded: X shape %s, y shape %s, datetimes shape %s", X.shape, y.shape, datetimes.shape)
    logger.debug(format_array_for_log(X, "HDF5 Initial X"))
    logger.debug(format_array_for_log(y, "HDF5 Initial y (raw distances)"))
    logger.debug(format_array_for_log(datetimes, "HDF5 Initial datetimes"))

    reduce_to_size = config.get('reduce_to_size', None)
    if reduce_to_size is not None and X.shape[1] > reduce_to_size:
        logger.info("Optional: reducing feature width to %d sensors", reduce_to_size)
        start = (X.shape[1] - reduce_to_size) // 2
        end   = start + reduce_to_size
        X = X[:, start:end]
        logger.info("Reduced X to central %d features (dim=1).", reduce_to_size)
        logger.debug(format_array_for_log(X, "HDF5 X after width reduction"))


    logger.info("Step 2: Prepare targets")
    # if config['is_regression']:
    #     y_targets = y
    #     logger.info("Regression task. Using raw y (distances) as targets.")
    #     logger.debug(format_array_for_log(y_targets, "HDF5 y_targets (raw for regression)"))
    # else:
    #     thresholds = config.get('thresholds', [1000])
    #     if len(thresholds) == 1:
    #         threshold_val = thresholds[0]
    #         if config.get('invert_threshold_logic', False):
    #             y_targets = (y <= threshold_val).astype(int)
    #         else:
    #             y_targets = (y > threshold_val).astype(int)


    #     else:
    #         y_targets = np.zeros_like(y, dtype=int)
    #         for i, t_val in enumerate(sorted(thresholds)):
    #             y_targets[y > t_val] = i + 1
    #     logger.info("Classification task. Binarized y_targets.")
    #     logger.debug(format_array_for_log(y_targets, "HDF5 y_targets (binarized for classification)"))

    # Optional y-range filtering on raw HDF5 y before any target preparation
    y_min = config.get('y_min')
    y_max = config.get('y_max')

    if y_min is not None or y_max is not None:
        # Build filter mask based on provided bounds
        if y_min is not None and y_max is not None:
            # Both bounds provided: y_min <= y <= y_max
            mask = (y_h5 >= y_min) & (y_h5 <= y_max)
            logger.info(f"Filtering to {y_min} ≤ y ≤ {y_max} (keeping {mask.sum()} of {len(y_h5)} points)")
        elif y_min is not None:
            # Only minimum provided: y >= y_min
            mask = (y_h5 >= y_min)
            logger.info(f"Filtering to y ≥ {y_min} (keeping {mask.sum()} of {len(y_h5)} points)")
        else:
            # Only maximum provided: y <= y_max
            mask = (y_h5 <= y_max)
            logger.info(f"Filtering to y ≤ {y_max} (keeping {mask.sum()} of {len(y_h5)} points)")

        # Apply same mask to features, raw y, override y (if any), and datetimes
        X, y_h5, datetimes = X[mask], y_h5[mask], datetimes[mask]
        y = y[mask]


    # If user supplied a target_file/key, use its raw content for any task (classification or regression)
    if config.get('target_file') is not None:
        y_targets = y
        logger.info("Custom-target task. Using target-key values as targets.")
        logger.debug(format_array_for_log(y_targets, "HDF5 or joblib y_target"))
    else:
        if config['is_regression']:
            y_targets = y
            logger.info("Regression task.")
            logger.debug(format_array_for_log(y_targets, "Regression y_targets"))
        else:
            thresholds = config.get('thresholds', [1000])
            if len(thresholds) == 1:
                threshold_val = thresholds[0]
                if config.get('invert_threshold_logic', False):
                    y_targets = (y <= threshold_val).astype(int)
                else:
                    y_targets = (y > threshold_val).astype(int)
            else:
                y_targets = np.zeros_like(y, dtype=int)
                for i, t_val in enumerate(sorted(thresholds)):
                    y_targets[y > t_val] = i + 1
            logger.info("Classification task. Binarized y_targets.")
            logger.debug(format_array_for_log(y_targets, "y_targets (binarized for classification)"))

    n_seconds = config['n_seconds']
    n_overlapping_seconds = config['n_overlapping_seconds']
    average_signals = config['average_signals']
    apply_log_val = config['apply_log']
    time_offset_seconds_val = config['time_offset_seconds']
    join_higher_classes_val = config['join_higher_classes']
    is_regression_val = config['is_regression']
    use_mid_target_val = config.get('use_mid_target', True)
    center_truth = config.get('center_truth', False)

    logger.info("Instantiating Reducer...")

    if is_regression_val:
        logger.info("Using TripletRegressionReducer for regression task.")
        reducer = TripletRegressionReducer(
            X, y_targets, datetimes,
            n_seconds=n_seconds,
            n_overlapping_seconds=n_overlapping_seconds,
            average_signals=average_signals,
            apply_log=apply_log_val,
            epsilon=1e-23,
            time_offset_seconds=time_offset_seconds_val,
            threshold=config.get('regression_threshold', None),
            use_mid_target=use_mid_target_val,
            regression_target_method=config.get('regression_target_method'),
        )
    else: # Classification
        logger.info("Using TripletReducer for classification task.")
        reducer = TripletReducer(
            X, y_targets, datetimes,
            n_seconds=n_seconds,
            n_overlapping_seconds=n_overlapping_seconds,
            average_signals=average_signals,
            apply_log=apply_log_val,
            epsilon=1e-23,
            time_offset_seconds=time_offset_seconds_val,
            join_higher_classes=join_higher_classes_val,
            use_mid_target=use_mid_target_val,
            center_truth = center_truth
        )

    X_reduced, y_reduced, dt_reduced = reducer.reduce_triplets()
    logger.debug(format_array_for_log(X_reduced, "HDF5 X_reduced"))
    # Debug: show reduced targets and timestamps for grouping logic
    logger.info(format_array_for_log(y_reduced, "TripletReducer output y_reduced"))
    logger.info(format_array_for_log(dt_reduced, "TripletReducer output dt_reduced"))
    # Summarize reduced samples per date and positives per date (parse ISO strings)
    from datetime import datetime as _dt
    dates = [_dt.fromisoformat(str(dt_val)).date() for dt_val in dt_reduced]
    uniq_dates, counts_dates = np.unique(dates, return_counts=True)
    logger.info("Reduced samples per day: %s", dict(zip(uniq_dates, counts_dates)))
    pos_map = {}
    for d, yv in zip(dates, y_reduced):
        pos_map.setdefault(d, []).append(int(yv))
    pos_counts = {d: sum(vals) for d, vals in pos_map.items()}
    logger.info("Reduced positives per day: %s", pos_counts)

    config['X'] = X_reduced
    config['y'] = y_reduced
    config['datetimes'] = dt_reduced

    logger.info("Ready to call executor with prepared data and config.")
    executor = PipelineExecutorHDF5(is_regression=config['is_regression'], config=config)
    executor.run()
    logger.info("HDF5 pipeline execution complete for %s", args.h5_path)

if __name__ == "__main__":
    main()
