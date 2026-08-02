import os
import random
import numpy as np
from typing import List, Optional, Union, Literal
from datetime import datetime, time, timedelta
from joblib import dump
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import logging
import mlflow
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile

# module-level logger
logger = logging.getLogger(__name__)
from data_splitter_hdf5 import (
    TripletReducer, DataSplitter, CrossValidator, ModelEvaluator, ShipLabeler,
    DataDateTimeBalancer, ShipDistanceTargetGenerator, TripletRegressionReducer, DataRegressionSplitter,
    RegressionCrossValidator, ModelRegressionEvaluator
)

from optional_import import optional_import

class FoldTextSaver:
    """
    Save per-day fold features, labels and datetimes to text files.
    """
    @staticmethod
    def save_fold(day_str: str,
                  fold_idx: int,
                  X: np.ndarray,
                  y: np.ndarray,
                  dt: list,
                  dataset_name: str,
                  n_seconds: int,
                  average_signals: str,
                  is_regression: bool):
        # Only for channel or time_channel modes
        if average_signals not in ("channel", "time_channel"):
            return
        # Prepare output directory: <dataset_name>_txt
        out_dir = f"{dataset_name}_txt"
        os.makedirs(out_dir, exist_ok=True)
        # Build filename components
        fold_no = fold_idx + 1
        fold_label = f"fold{fold_no:02d}"
        wint = n_seconds // 10
        prefix = f"{day_str}-{dataset_name}"
        # Save labels
        label_type = "dists" if is_regression else "label"
        label_file = os.path.join(out_dir, f"{prefix}-{label_type}-{fold_label}.txt")
        np.savetxt(label_file, y, fmt="%s")
        # Save datetimes
        dates_file = os.path.join(out_dir, f"{prefix}-dates-{fold_label}.txt")
        with open(dates_file, "w", encoding="utf-8") as f:
            for dt_item in dt:
                f.write(str(dt_item) + "\n")
        # Save features
        feats_file = os.path.join(out_dir, f"{prefix}-feats-{fold_label}-avg_wint{wint}.txt")
        np.savetxt(feats_file, X, fmt="%s")

class Splitter:
    """
    Handles data splitting for training and testing, supporting different split methods.
    """

    def __init__(self, X, y, dt):
        """
        Initializes the Splitter with the data to be split.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels or targets.
            dt (np.ndarray): Datetimes.
        """
        self.X = X
        self.y = y
        self.dt = np.array(dt, dtype=object)

    def split_by_day(self, test_day: Union[str, List[str]], balance_test: bool = False):
        """
        Splits data by leaving out a specific day or a range of days for testing.
        """
        if isinstance(test_day, str):
            test_day = [test_day]
        if len(test_day) == 1:
            # Single test day
            test_days = [datetime.strptime(test_day[0], '%Y-%m-%d').date()]
        elif len(test_day) == 2:
            start_date = datetime.strptime(test_day[0], '%Y-%m-%d').date()
            end_date = datetime.strptime(test_day[1], '%Y-%m-%d').date()
            test_days = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
        else:
            raise ValueError("test_day must be a single date or a range of two dates in 'YYYY-MM-DD' format.")

        # Now test_days is always a list of date objects

        # Make sure self.dt are also date objects for comparison!
        dt_dates = [datetime.fromisoformat(dt).date() if isinstance(dt, str) else dt.date() for dt in self.dt]
        test_indices = [i for i, date in enumerate(dt_dates) if date in test_days]
        train_indices = [i for i in range(len(self.dt)) if i not in test_indices]

        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        dt_train, dt_test = self.dt[train_indices], self.dt[test_indices]

        if balance_test:
            # Add balancing logic here if required
            pass

        return X_train, X_test, y_train, y_test, dt_train, dt_test


    def split_by_time_interval(self, interval_start: time, interval_end: time, balance_test: bool = False):
        """
        Splits data by leaving out a specific time interval for testing.

        Args:
            interval_start (time): Start of the interval.
            interval_end (time): End of the interval.
            balance_test (bool): Whether to balance the test set.

        Returns:
            tuple: Split data (X_train, X_test, y_train, y_test, dt_train, dt_test).
        """
        test_indices = [
            i for i, date in enumerate(self.dt)
            if interval_start <= date.time() <= interval_end
        ]
        train_indices = [i for i in range(len(self.dt)) if i not in test_indices]

        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        dt_train, dt_test = self.dt[train_indices], self.dt[test_indices]

        if balance_test:
            # Add balancing logic here if required
            pass

        return X_train, X_test, y_train, y_test, dt_train, dt_test

    def split_random(self, test_size: float = 0.3, random_state: Optional[int] = None):
        """
        Splits data randomly into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (Optional[int]): Random seed for reproducibility.

        Returns:
            tuple: Split data (X_train, X_test, y_train, y_test, dt_train, dt_test).
        """
        indices = list(range(len(self.X)))
        np.random.seed(random_state)
        np.random.shuffle(indices)

        split_idx = int(len(self.X) * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        dt_train, dt_test = self.dt[train_indices], self.dt[test_indices]

        return X_train, X_test, y_train, y_test, dt_train, dt_test

class Balancer:
    """
    Handles balancing of the training dataset for classification tasks.
    Supports multiple balancing methods such as SMOTE, ADASYN, naive oversampling, and undersampling.
    """

    def __init__(self, X_train, y_train, dt_train):
        """
        Initializes the Balancer with the training data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            dt_train (np.ndarray): Training datetimes.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.dt_train = dt_train

    def balance(self, method: Literal['unbalanced', 'smote', 'adasyn', 'naive', 'undersample'], random_state: int = 42):
        """
        Balances the training dataset using the specified method.

        Args:
            method (Literal['unbalanced', 'smote', 'adasyn', 'naive', 'undersample']): Balancing method.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Balanced training data (X_train, y_train, dt_train).
        """
        if method == 'unbalanced':
            return self.X_train, self.y_train, self.dt_train

        balancer = DataDateTimeBalancer(self.X_train, self.y_train, self.dt_train)

        if method == 'smote':
            return balancer.oversample_smote(random_state=random_state, k_neighbors=2)
        elif method == 'adasyn':
            return balancer.oversample_adasyn(random_state=random_state, n_neighbors=2)
        elif method == 'naive':
            return balancer.naive_oversample(random_state=random_state)
        elif method == 'undersample':
            return balancer.undersample(random_state=random_state)
        else:
            raise ValueError(f"Invalid balancing method: {method}. Must be one of ['unbalanced', 'smote', 'adasyn', 'naive', 'undersample'].")


class ModelTrainer:
    """
    Handles training of an XGBoost model with early stopping and optional grid search for hyperparameter tuning.
    Supports both regression and classification tasks.
    """

    def __init__(self, is_regression: bool, perform_grid_search: bool = False, param_grid: dict = None):
        """
        Initializes the ModelTrainer.

        Args:
            is_regression (bool): Specifies whether the task is regression or classification.
            perform_grid_search (bool): Whether to perform grid search for hyperparameter tuning.
            param_grid (dict): Dictionary of hyperparameter grids for grid search.
        """
        self.is_regression = is_regression
        self.perform_grid_search = perform_grid_search
        self.param_grid = param_grid if param_grid else {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }


    def train(self, X_train, y_train, X_val, y_val, random_state: int = 42):
        """
        Trains the XGBoost model with early stopping and optional grid search.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels or targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels or targets.
            random_state (int): Random seed for reproducibility.

        Returns:
            object: Trained XGBoost model.
        """
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        if self.is_regression:
            model = xgb.XGBRegressor(
                random_state=random_state,
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
        else:
            model = xgb.XGBClassifier(
                random_state=random_state,
                early_stopping_rounds=50,
                eval_metric='logloss'
            )

        if self.perform_grid_search:
            logger.info("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                scoring='neg_mean_squared_error' if self.is_regression else 'accuracy',
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                        verbose=False)
            best_params = grid_search.best_params_
            logger.info("Best hyperparameters: %s", best_params)

            if self.is_regression:
                model = xgb.XGBRegressor(
                    **best_params,
                    random_state=random_state,
                    early_stopping_rounds=50,
                    eval_metric='rmse'
                )
            else:
                model = xgb.XGBClassifier(
                    **best_params,
                    random_state=random_state,
                    early_stopping_rounds=50,
                    eval_metric='logloss'
                )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        return model

class NeuralNetworkHandler:
    def __init__(
        self,
        load_model_fn,
        hidden_dim=128,
        batch_size=32,
        patience=20,
        lr=0.001,
        device=None,
        is_regression=False,
        fold_label="1/1",
    ):
        """
        Handles neural network model training and evaluation, wrapping the model to be compatible with sklearn-like interfaces.

        Args:
            load_model_fn (function): Function to load the model, optimizer, and criterion.
            hidden_dim (int): Size of the hidden layer.
            batch_size (int): Batch size for training.
            patience (int): Early stopping patience.
            lr (float): Learning rate.
            device (str): Device to use ('cuda' or 'cpu').
            is_regression (bool): Flag indicating whether the task is regression.
        """
        self.load_model_fn = load_model_fn
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.is_regression = is_regression
        task_label = "regress" if is_regression else "classif"
        self.training_context = f"{task_label} fold {fold_label}"
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model_wrapper = None
        self.is_3d_input = False  # Flag to track if input is 3D (image-like, for CNN models)

    class SklearnCompatibleNNWrapper:
        def __init__(self, model, scaler, device, is_regression, is_3d_input=False):
            self.model = model
            self.scaler = scaler
            self.device = device
            self.is_regression = is_regression
            self.is_3d_input = is_3d_input  # Skip scaling for 3D input (CNN models use BatchNorm)

        def _preprocess(self, X):
            """Preprocess input: scale for 2D, pass through for 3D (CNN)."""
            if self.is_3d_input:
                # For 3D input (CNN), skip scaling - model uses internal BatchNorm
                return torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                # For 2D input (MLP), apply StandardScaler
                X_scaled = self.scaler.transform(X)
                return torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        def predict(self, X, batch_size=64):
            """
            Predict using batched inference to avoid GPU OOM on large test sets.

            For CNNs, intermediate activations can be huge (e.g., 24k samples × 32 channels × 125 × 50 = 19GB).
            Batched prediction processes data in smaller chunks to fit in GPU memory.
            """
            self.model.eval()
            predictions_list = []

            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_tensor = self._preprocess(batch_X)
                    outputs = self.model(batch_tensor)

                    if self.is_regression:
                        # If the output is 2D with a singleton dimension, squeeze it
                        if outputs.ndim > 1 and outputs.size(1) == 1:
                            preds = outputs.squeeze(1)
                        else:
                            preds = outputs
                    else:
                        # For classification, use torch.max to get class predictions
                        _, preds = torch.max(outputs, 1)

                    predictions_list.append(preds.cpu())
                    # Free GPU memory immediately
                    del batch_tensor, outputs

            return torch.cat(predictions_list).numpy()

        def predict_proba(self, X, batch_size=64):
            """
            Predict class probabilities using batched inference.
            """
            self.model.eval()
            proba_list = []

            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_tensor = self._preprocess(batch_X)
                    outputs = self.model(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    proba_list.append(probabilities.cpu())
                    del batch_tensor, outputs

            return torch.cat(proba_list).numpy()

        def __call__(self, X, batch_size=64):
            """
            Makes the wrapper callable, compatible with SHAP explainer.
            Uses batched inference for memory efficiency.
            """
            self.model.eval()
            outputs_list = []

            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_tensor = self._preprocess(batch_X)
                    outputs = self.model(batch_tensor)
                    outputs_list.append(outputs.cpu())
                    del batch_tensor

            return torch.cat(outputs_list).numpy()

    def train(self, X_train, y_train, epochs=100, seed=42):
        """
        Trains the neural network using early stopping and wraps it in a sklearn-compatible interface.

        Args:
            X_train (np.array): Training features (2D for MLP, 3D for CNN models).
            y_train (np.array): Training targets.
            epochs (int): Maximum number of epochs.
            seed (int): Random seed for reproducibility.
        """
        # --- Set all random seeds ---
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN (might affect performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Detect 3D input (CNN models) - skip scaling, use BatchNorm instead
        self.is_3d_input = X_train.ndim == 3
        if self.is_3d_input:
            # For 3D input (CNN), skip StandardScaler - model should use BatchNorm
            X_train_processed = X_train
        else:
            # For 2D input (MLP), apply StandardScaler
            X_train_processed = self.scaler.fit_transform(X_train)

        # Load model, optimizer, criterion, and optionally scheduler
        model_tuple = self.load_model_fn(X_train_processed, y_train, hidden_dim=self.hidden_dim)
        if len(model_tuple) == 4:
            model, optimizer, criterion, scheduler = model_tuple
        else:
            model, optimizer, criterion = model_tuple
            scheduler = None

        model = model.to(self.device)
        # Some losses (e.g., CrossEntropyLoss with class weights) hold internal tensors
        # that must live on the same device as model outputs/targets.
        if hasattr(criterion, "to"):
            criterion = criterion.to(self.device)

        # Convert data to tensors - keep on CPU to save GPU memory
        # Batches will be moved to GPU during training
        X_tensor = torch.tensor(X_train_processed, dtype=torch.float32)  # CPU
        if self.is_regression:
            y_tensor = torch.tensor(y_train, dtype=torch.float32)  # CPU
        else:
            y_tensor = torch.tensor(y_train, dtype=torch.long)  # CPU

        # Create dataset and data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        # Use a generator for deterministic shuffling in the DataLoader
        g = torch.Generator()
        g.manual_seed(seed)
        # drop_last=True prevents BatchNorm issues when last batch has size 1
        # pin_memory=True for faster CPU->GPU transfer, num_workers for parallel loading
        use_pin_memory = self.device.type == 'cuda'
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=g,
            drop_last=True, pin_memory=use_pin_memory, num_workers=0
        )

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            # for batch_X, batch_y in loader:
            #     optimizer.zero_grad()
            #     outputs = model(batch_X)
            #     loss = criterion(outputs, batch_y)
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.item()
            
            for batch_X, batch_y in loader:
                # Move batch to GPU
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)           # [batch, 1]
                if self.is_regression and outputs.ndim > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)    # → [batch]
                loss    = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


            avg_loss = total_loss / len(loader)
            logger.info(
                "[%s] Epoch %d/%d, Loss: %.4f",
                self.training_context,
                epoch + 1,
                epochs,
                avg_loss,
            )

            # If a scheduler is provided, update it with the current loss
            if scheduler is not None:
                scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logger.info("[%s] Early stopping triggered.", self.training_context)
                break

        # Load the best model state if available
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Wrap the model for sklearn-like interface, passing regression and 3D input flags
        self.model_wrapper = self.SklearnCompatibleNNWrapper(model, self.scaler, self.device, self.is_regression, self.is_3d_input)

    @property
    def model(self):
        return self.model_wrapper

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained neural network on the test set using the sklearn-compatible wrapper.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): Test targets.

        Returns:
            dict: Evaluation metrics.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return {"accuracy": accuracy}

class MetricsAggregator:
    """
    Aggregates and computes metrics across multiple days or splits.
    Used primarily for classification and regression evaluation.
    """

    def __init__(self, is_regression: bool):
        """
        Initializes the MetricsAggregator.

        Args:
            is_regression (bool): Specifies whether the task is regression or classification.
        """
        self.is_regression = is_regression
        self.metrics = []

    def add_metrics(self, day_metrics: dict):
        """
        Adds metrics for a single day or split to the aggregator.

        Args:
            day_metrics (dict): Metrics for a single day or split.
        """
        self.metrics.append(day_metrics)


    def compute_averages(self) -> dict:
        """
        For classification tasks, computes:
          - average_confusion_matrix: element-wise average of daily confusion_matrices
          - aggregated_confusion_matrix: element-wise sum of daily confusion_matrices
          - average_classification_report: simple average of classification_report entries
          - fold_weighted_classification_report: support-weighted avg for each metric,
            sums for support entries.
        For regression tasks, falls back to simple recursive mean over all metrics.
        """
        if not self.metrics:
            raise ValueError("No metrics to aggregate.")

        # Helper: recursive mean for regression or unexpected nested dicts
        def _recursive_mean(metrics_list):
            agg = {}
            for key in metrics_list[0]:
                values = [m[key] for m in metrics_list]
                if isinstance(values[0], dict):
                    agg[key] = _recursive_mean(values)
                elif isinstance(values[0], np.ndarray):
                    agg[key] = np.mean(values, axis=0)
                else:
                    agg[key] = np.mean(values)
            return agg

        # Regression: aggregate per-day regression metrics via dedicated aggregator
        if self.is_regression:
            return RegressionMetricsAggregator(self.metrics).compute()

        # Classification: focus on confusion_matrix and classification_report
        cms = []
        reports = []
        for m in self.metrics:
            cm = m.get("confusion_matrix")
            if cm is not None:
                arr = np.array(cm)
                if arr.ndim == 1 and arr.size == 4:
                    arr = arr.reshape(2, 2)
                cms.append(arr)
            cr = m.get("classification_report")
            if cr is not None:
                reports.append(cr)

        if not cms or not reports:
            raise ValueError("MetricsAggregator: missing confusion_matrix or classification_report entry.")

        # Aggregated and average confusion matrices
        summed_cm = np.sum(cms, axis=0)
        avg_cm = summed_cm / len(cms)

        # Simple average of classification_report entries
        def _avg_report(rep_list):
            out = {}
            for key in rep_list[0]:
                vals = [r[key] for r in rep_list]
                if isinstance(vals[0], dict):
                    out[key] = _avg_report(vals)
                else:
                    out[key] = np.mean(vals)
            return out

        avg_report = _avg_report(reports)

        # Support-weighted classification report
        def _weighted_report(rep_list):
            out = {}
            for key in rep_list[0]:
                vals = [r[key] for r in rep_list]
                if isinstance(vals[0], dict):
                    sub = {}
                    supports = [v.get("support", 0) for v in vals]
                    total_sup = sum(supports)
                    for subkey in vals[0]:
                        if "support" in subkey:
                            sub[subkey] = sum(v.get(subkey, 0) for v in vals)
                        else:
                            num = sum(v.get(subkey, 0) * v.get("support", 0) for v in vals)
                            sub[subkey] = num / total_sup if total_sup > 0 else 0.0
                    out[key] = sub
                else:
                    # Top-level numeric (e.g. accuracy) weighted by total support
                    if key == "accuracy":
                        sup = [r.get("weighted avg", {}).get("support", 0) for r in rep_list]
                        total_sup = sum(sup)
                        num = sum(r.get("accuracy", 0) * s for r, s in zip(rep_list, sup))
                        out[key] = num / total_sup if total_sup > 0 else 0.0
                    else:
                        out[key] = np.mean(vals)
            return out

        weighted_report = _weighted_report(reports)

        return {
            "aggregated_confusion_matrix": summed_cm,
            "average_confusion_matrix":     avg_cm,
            "average_classification_report":    avg_report,
            "fold_weighted_classification_report": weighted_report,
        }

   
class ConfidenceIntervalCalculator:
    """
    Calculates confidence intervals for metrics assuming a normal distribution.
    Compatible with global and class-weighted metrics aggregated over k-folds or multiple days.
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_value = norm.ppf(1 - (1 - confidence_level) / 2)  # Z-score for the confidence level

    def calculate(self, metric_values: np.ndarray, total_instances: int = None, weights: np.ndarray = None) -> dict:
        """
        Calculate the confidence interval for a given set of metric values.

        Args:
            metric_values (np.ndarray): Array of metric values (e.g., accuracies across k-days).
            total_instances (int, optional): Total number of instances used for scaling the standard error.
                                            Defaults to the length of metric_values if not provided.
            weights (np.ndarray, optional): Array of weights corresponding to metric values for weighted calculations.

        Returns:
            dict: Dictionary containing the mean, lower bound, upper bound, and margin of error.
        """
        if len(metric_values) == 0:
            raise ValueError("At least one metric value is required.")

        if len(metric_values) == 1:
            mean_metric = float(metric_values[0])
            logger.warning(
                "Only one fold is available; a between-fold confidence interval "
                "cannot be estimated. Returning the fold value with zero margin."
            )
            return {
                "mean": mean_metric,
                "lower_bound": mean_metric,
                "upper_bound": mean_metric,
                "margin_of_error": 0.0,
            }

        if weights is not None:
            if len(metric_values) != len(weights):
                raise ValueError("metric_values and weights must have the same length.")
            mean_metric = np.average(metric_values, weights=weights)
            variance = np.average((metric_values - mean_metric) ** 2, weights=weights)
            std_metric = np.sqrt(variance)
        else:
            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values, ddof=1)

        if total_instances is None:
            total_instances = len(metric_values)

        margin_of_error = self.z_value * (std_metric / np.sqrt(total_instances))
        lower_bound = mean_metric - margin_of_error
        upper_bound = mean_metric + margin_of_error

        return {
            "mean": mean_metric,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "margin_of_error": margin_of_error
        }
    
class JoblibSaver:
    """
    Handles saving experiment results and metadata to a joblib file.
    """

    @staticmethod
    def save(file_path: str, data: dict, metadata: dict, output_suffix: str = ""):
        """
        Saves experiment data and metadata into a joblib file.

        Args:
            file_path (str): Base path to save the joblib file.
            data (dict): Dictionary containing experiment results.
            metadata (dict): Dictionary containing experiment parameters and metadata.
            output_suffix (str): Optional suffix to append before the file extension.

        Returns:
            str: The full path of the saved joblib file.
        """
        # Add suffix to the file path
        if output_suffix:
            base, ext = os.path.splitext(file_path)
            file_path = f"{base}{output_suffix}{ext}"

        # Combine data and metadata into a single dictionary
        metadata.pop('X', None)
        metadata.pop('y', None)
        metadata.pop('datetimes', None)
        
        
        to_save = {
            "metrics": data,
            "metadata": metadata,
        }
        
        

        # Save the dictionary to the joblib file
        dump(to_save, file_path)
        logger.info("Results saved to %s", file_path)
        return file_path


class RegressionMetricsAggregator:
    """
    Aggregates per-day regression metrics by simple mean and support-weighted mean.
    """
    def __init__(self, daily_metrics: list):
        self.daily_metrics = daily_metrics

    def compute(self) -> dict:
        import numpy as np
        # Determine support per day (use 'SUPPORT' or fallback to fold_summary n)
        supports = [
            m.get('SUPPORT', m.get('fold_summary', {}).get('n', 1))
            for m in self.daily_metrics
        ]
        total_support = sum(supports)
        avg = {}
        weighted = {}
        # Collect keys to aggregate, excluding nested summaries
        for key in self.daily_metrics[0]:
            if key in ('fold_summary', 'day'):
                continue
            values = [m.get(key) for m in self.daily_metrics]
            # Per-sample arrays such as residuals, y_true, and y_pred must remain
            # available in metrics_by_day, but they are not fold-level scalar
            # metrics and cannot be support-weighted with one weight per fold.
            if not all(np.isscalar(value) for value in values):
                continue
            try:
                arr = np.array(values, dtype=float)
            except Exception:
                continue
            avg[key] = arr.mean()
            # SUPPORT should be summed for fold-weighted case, others are support-weighted means
            if key == 'SUPPORT':
                weighted[key] = total_support
            else:
                weighted[key] = float(arr.dot(supports) / total_support) if total_support > 0 else np.nan
        return {
            'average_metrics': avg,
            'fold_weighted_metrics': weighted
        }


def aggregate_regression_threshold_metrics(daily_metrics, threshold, random_state=42):
    """Aggregate an optional evaluation-only slice across regression folds."""
    fold_metrics = [
        metric.get("regression_threshold_evaluation")
        for metric in daily_metrics
        if isinstance(metric.get("regression_threshold_evaluation"), dict)
    ]
    y_true_parts = [m["y_true"] for m in fold_metrics if len(m.get("y_true", []))]
    y_pred_parts = [m["y_pred"] for m in fold_metrics if len(m.get("y_pred", []))]
    support = sum(int(m.get("SUPPORT", 0)) for m in fold_metrics)
    result = {
        "threshold": float(threshold),
        "SUPPORT": support,
        "metrics_by_day": fold_metrics,
    }

    scalar_keys = ("MAE", "RMSE", "MSE", "R2", "MAE_STD", "RMSE_STD", "MEAN_TARGET")
    nonempty = [m for m in fold_metrics if int(m.get("SUPPORT", 0)) > 0]
    result["average_metrics"] = {
        key: float(np.nanmean([m.get(key, np.nan) for m in nonempty]))
        for key in scalar_keys
        if nonempty
    }
    result["average_metrics"]["SUPPORT"] = support
    result["fold_weighted_metrics"] = {
        key: (
            float(np.average(
                [m.get(key, np.nan) for m in nonempty],
                weights=[m["SUPPORT"] for m in nonempty],
            ))
            if nonempty else np.nan
        )
        for key in scalar_keys
    }
    result["fold_weighted_metrics"]["SUPPORT"] = support

    if not y_true_parts:
        result["final_results"] = {
            key: (np.nan, (np.nan, np.nan))
            for key in ("MAE", "RMSE", "MSE", "R2")
        }
        return result

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    rng = np.random.RandomState(random_state)
    boot = {key: [] for key in ("MAE", "RMSE", "MSE", "R2")}
    for _ in range(1000):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        true_sample = y_true[indices]
        pred_sample = y_pred[indices]
        residuals = true_sample - pred_sample
        mse = np.mean(residuals ** 2)
        boot["MAE"].append(np.mean(np.abs(residuals)))
        boot["RMSE"].append(np.sqrt(mse))
        boot["MSE"].append(mse)
        denominator = np.sum((true_sample - np.mean(true_sample)) ** 2)
        boot["R2"].append(
            1 - np.sum(residuals ** 2) / denominator if denominator > 0 else np.nan
        )

    result["final_results"] = {}
    for key, values in boot.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            result["final_results"][key] = (
                float(np.mean(finite)),
                tuple(float(value) for value in np.percentile(finite, [2.5, 97.5])),
            )
        else:
            result["final_results"][key] = (np.nan, (np.nan, np.nan))
    return result


def mlflow_threshold_metrics(threshold_result):
    """Flatten threshold results into MLflow-safe scalar metric names."""
    values = {"threshold_support": float(threshold_result.get("SUPPORT", 0))}
    for key, result in threshold_result.get("final_results", {}).items():
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            continue
        point, interval = result
        if np.isfinite(point):
            values[f"threshold_{key}"] = float(point)
        if isinstance(interval, (tuple, list)) and len(interval) == 2:
            if np.isfinite(interval[0]):
                values[f"threshold_{key}_low_CI"] = float(interval[0])
            if np.isfinite(interval[1]):
                values[f"threshold_{key}_high_CI"] = float(interval[1])
    return values


def log_performance_results(metrics, is_regression, scope, day=None, fold_label=None):
    """Log performance from the same dictionaries persisted to Joblib."""
    task = "regression" if is_regression else "classification"
    context = [scope]
    if fold_label:
        context.append(f"fold {fold_label}")
    if day:
        context.append(f"test day {day}")
    logger.info("Performance results: %s (%s)", task, ", ".join(context))

    final_results = metrics.get("final_results")
    if isinstance(final_results, dict):
        names = ("MAE", "RMSE", "MSE", "R2") if is_regression else (
            "f1", "accuracy", "auc"
        )
        for name in names:
            result = final_results.get(name)
            if isinstance(result, (tuple, list)) and len(result) == 2:
                value, interval = result
                logger.info(
                    "  %s: %.4f (95%% CI: %.4f, %.4f)",
                    name, value, interval[0], interval[1],
                )
        if not is_regression:
            for class_name, (value, interval) in final_results.get("per_class", {}).items():
                logger.info(
                    "  Class %s F1: %.4f (95%% CI: %.4f, %.4f)",
                    class_name, value, interval[0], interval[1],
                )
            confusion = metrics.get("aggregated_confusion_matrix")
            if confusion is not None:
                confusion = np.asarray(confusion)
                logger.info("  Support: %d", int(confusion.sum()))
                logger.info("  Confusion matrix: %s", confusion.tolist())
    elif is_regression:
        logger.info("  Support: %d", int(metrics.get("SUPPORT", 0)))
        for name in ("MAE", "RMSE", "MSE", "R2", "MAE_STD", "RMSE_STD"):
            if name in metrics:
                logger.info("  %s: %.4f", name, metrics[name])
        if "MEAN_TARGET" in metrics:
            logger.info("  Mean target: %.4f", metrics["MEAN_TARGET"])
    else:
        report = metrics.get("classification_report", {})
        support = report.get("weighted avg", {}).get("support")
        if support is not None:
            logger.info("  Support: %d", int(support))
        logger.info("  Accuracy: %.4f", metrics["accuracy"])
        logger.info("  ROC AUC: %.4f", metrics["auc"])
        for average_name in ("macro avg", "weighted avg"):
            average = report.get(average_name, {})
            if "f1-score" in average:
                logger.info(
                    "  %s F1: %.4f",
                    average_name.replace(" avg", "").capitalize(),
                    average["f1-score"],
                )
        logger.info(
            "  Confusion matrix: %s",
            np.asarray(metrics["confusion_matrix"]).tolist(),
        )

    for name, result in metrics.get("uncertainty", {}).items():
        if all(key in result for key in ("mean", "lower_bound", "upper_bound")):
            logger.info(
                "  Daily bootstrap %s: %.4f (95%% CI: %.4f, %.4f)",
                name, result["mean"], result["lower_bound"], result["upper_bound"],
            )

    threshold_result = metrics.get("regression_threshold_evaluation")
    if is_regression and isinstance(threshold_result, dict):
        logger.info(
            "  Evaluation subset: true target <= %s (support=%d)",
            threshold_result.get("threshold"), int(threshold_result.get("SUPPORT", 0)),
        )
        threshold_metrics = threshold_result.get("final_results", threshold_result)
        for name in ("MAE", "RMSE", "MSE", "R2"):
            result = threshold_metrics.get(name)
            if isinstance(result, (tuple, list)) and len(result) == 2:
                value, interval = result
                logger.info(
                    "    %s: %.4f (95%% CI: %.4f, %.4f)",
                    name, value, interval[0], interval[1],
                )
            elif result is not None:
                logger.info("    %s: %.4f", name, result)


class CsvSaver:
    """
    Handles saving per-day and global metrics to a CSV file alongside the joblib output.
    """
    @staticmethod
    def save(joblib_path: str, metrics: dict):
        """
        Writes metrics_by_day, average and weighted reports, and final_results into a CSV.

        Args:
            joblib_path: Path to the saved .joblib file.
            metrics: The aggregated metrics dict (same as saved to joblib under 'metrics').
        """
        from pathlib import Path
        import csv

        path = Path(joblib_path)
        csv_path = path.with_suffix('.csv')

        day_metrics = metrics.get('metrics_by_day')
        if not day_metrics:
            return

        rows = []
        # Classification CSV if classification metrics present
        if 'average_classification_report' in metrics:
            avg_report = metrics['average_classification_report']
            # build header
            def _flatten_keys(report):
                keys = []
                for k, v in report.items():
                    if isinstance(v, dict):
                        for subk in v:
                            keys.append(f"{k}_{subk}")
                    else:
                        keys.append(k)
                return keys

            header = ['day'] + _flatten_keys(avg_report)

            # daily rows
            for entry in day_metrics:
                day = entry.get('day', '')
                report = entry.get('classification_report', {})
                flat = {}
                for k, v in report.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            flat[f"{k}_{subk}"] = subv
                    else:
                        flat[k] = v
                rows.append([day] + [flat.get(col, '-') for col in header[1:]])

            # macro avg and fold-weighted avg
            for label, report in (
                ('macro avg', metrics.get('average_classification_report', {})),
                ('fold-weighted avg', metrics.get('fold_weighted_classification_report', {})),
            ):
                row = [label]
                for col in header[1:]:
                    key, sub = (col.split('_', 1) + [None])[:2]
                    if key in report:
                        val = report[key].get(sub) if sub else report.get(key)
                    else:
                        val = '-'
                    row.append(val)
                rows.append(row)

            # final_results row
            final = metrics.get('final_results', {})
            flat_final = {}
            if 'f1' in final:
                v, ci = final['f1']
                flat_final['weighted avg_f1-score'] = f"{v:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
            if 'accuracy' in final:
                v, ci = final['accuracy']
                flat_final['accuracy'] = f"{v:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
            for cls, (v, ci) in final.get('per_class', {}).items():
                flat_final[f"{cls}_f1-score"] = f"{v:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
            rows.append(['bootstrap final results'] + [flat_final.get(col, '-') for col in header[1:]])

        # Regression CSV if regression metrics present
        elif 'average_metrics' in metrics:
            metric_names = ['MAE', 'SUPPORT', 'RMSE', 'R2', 'MAE_STD']
            threshold_result = metrics.get('regression_threshold_evaluation')
            threshold_names = (
                [f'threshold_{name}' for name in metric_names]
                if isinstance(threshold_result, dict) else []
            )
            header = ['day'] + metric_names + threshold_names

            # daily rows
            for entry in day_metrics:
                day = entry.get('day', '')
                threshold_fold = entry.get('regression_threshold_evaluation', {})
                rows.append(
                    [day]
                    + [entry.get(m, '-') for m in metric_names]
                    + [threshold_fold.get(m, '-') for m in metric_names]
                )

            # macro avg (average_metrics)
            avgm = metrics.get('average_metrics', {})
            threshold_avg = (threshold_result or {}).get('average_metrics', {})
            rows.append(
                ['macro avg']
                + [avgm.get(m, '-') for m in metric_names]
                + [threshold_avg.get(m, '-') for m in metric_names]
            )

            # fold-weighted avg (fold_weighted_metrics)
            wtm = metrics.get('fold_weighted_metrics', {})
            threshold_weighted = (threshold_result or {}).get(
                'fold_weighted_metrics', {}
            )
            rows.append(
                ['fold-weighted avg']
                + [wtm.get(m, '-') for m in metric_names]
                + [threshold_weighted.get(m, '-') for m in metric_names]
            )

            # final_results (bootstrap)
            fin = metrics.get('final_results', {})
            fr = []
            for m in metric_names:
                if m in fin:
                    v, ci = fin[m]
                    fr.append(f"{v:.4f} ({ci[0]:.4f}, {ci[1]:.4f})")
                else:
                    fr.append('-')
            threshold_final = (threshold_result or {}).get('final_results', {})
            threshold_fr = []
            for m in metric_names:
                if m in threshold_final:
                    v, ci = threshold_final[m]
                    threshold_fr.append(f"{v:.4f} ({ci[0]:.4f}, {ci[1]:.4f})")
                else:
                    threshold_fr.append('-')
            rows.append(['bootstrap final results'] + fr + threshold_fr)

        else:
            return

        # write CSV
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        logger.info("CSV report saved to %s", csv_path)

class MLflowArtifactCreator:
    """
    Creates visualization artifacts for MLflow logging.
    """

    @staticmethod
    def create_confusion_matrix_plot(confusion_matrix, class_names=None, title="Confusion Matrix"):
        """
        Create a confusion matrix heatmap and save to a temporary file.

        Args:
            confusion_matrix: 2D array
            class_names: Optional list of class names
            title: Plot title

        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(confusion_matrix))]

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        plt.tight_layout()

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return temp_file.name

    @staticmethod
    def create_residual_plots(y_true, y_pred, title_prefix="Regression"):
        """
        Create comprehensive residual analysis plots for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
            title_prefix: Prefix for plot titles

        Returns:
            Dictionary with paths to saved figures
        """
        residuals = y_true - y_pred

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Predicted vs Actual
        axes[0, 0].scatter(y_pred, y_true, alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Predicted Distance (m)')
        axes[0, 0].set_ylabel('True Distance (m)')
        axes[0, 0].set_title(f'{title_prefix}: Predicted vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Distance (m)')
        axes[0, 1].set_ylabel('Residuals (m)')
        axes[0, 1].set_title(f'{title_prefix}: Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residual Histogram with Normal Overlay
        axes[1, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 0].plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        axes[1, 0].set_xlabel('Residuals (m)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'{title_prefix}: Residual Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{title_prefix}: Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return {'residual_analysis': temp_file.name}

    @staticmethod
    def create_per_day_metrics_plot(metrics_by_day, metric_name, ylabel, title, global_ci=None):
        """
        Create a line plot showing metrics across days with confidence intervals.

        Args:
            metrics_by_day: List of daily metrics dictionaries
            metric_name: Name of the metric to plot
            ylabel: Y-axis label
            title: Plot title
            global_ci: Optional dict with 'mean', 'lower_bound', 'upper_bound' for global CI band

        Returns:
            Path to saved figure

        Notes:
            - If per-day CIs are available (as 'low_CI' and 'high_CI' suffixes), they are shown as error bars
            - If global_ci is provided, a shaded band shows the overall confidence interval
            - Gracefully degrades to simple line plot if CIs are not available
        """
        days = [m.get('day', f'Day {i+1}') for i, m in enumerate(metrics_by_day)]
        values = [m.get(metric_name, np.nan) for m in metrics_by_day]

        # Try to extract per-day confidence intervals
        # Look for keys like 'MAE_low_CI', 'MAE_high_CI' or nested bootstrap results
        low_ci_key = f'{metric_name}_low_CI'
        high_ci_key = f'{metric_name}_high_CI'

        lower_bounds = []
        upper_bounds = []
        has_per_day_ci = False

        for m in metrics_by_day:
            # Check if CI keys exist directly
            if low_ci_key in m and high_ci_key in m:
                lower_bounds.append(m[low_ci_key])
                upper_bounds.append(m[high_ci_key])
                has_per_day_ci = True
            # Check if there's a nested bootstrap_ci structure
            elif 'bootstrap_ci' in m and metric_name in m.get('bootstrap_ci', {}):
                ci_data = m['bootstrap_ci'][metric_name]
                if isinstance(ci_data, dict):
                    lower_bounds.append(ci_data.get('lower_bound', np.nan))
                    upper_bounds.append(ci_data.get('upper_bound', np.nan))
                    has_per_day_ci = True
                else:
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)
            else:
                lower_bounds.append(np.nan)
                upper_bounds.append(np.nan)

        fig, ax = plt.subplots(figsize=(12, 6))
        x_positions = range(len(days))

        # Plot with or without error bars based on availability
        if has_per_day_ci and not all(np.isnan(lower_bounds)) and not all(np.isnan(upper_bounds)):
            # Calculate error bar sizes (distances from the point values)
            values_array = np.array(values)
            lower_array = np.array(lower_bounds)
            upper_array = np.array(upper_bounds)

            # Error bars expect the distance from the point, not absolute values
            yerr_lower = values_array - lower_array
            yerr_upper = upper_array - values_array

            ax.errorbar(x_positions, values, yerr=[yerr_lower, yerr_upper],
                       marker='o', linewidth=2, markersize=8, capsize=5,
                       capthick=2, elinewidth=1.5, label='Per-Day Value with 95% CI')
        else:
            # Simple line plot without error bars
            ax.plot(x_positions, values, marker='o', linewidth=2, markersize=8,
                   label='Per-Day Value')

        # Add global mean and CI band if provided
        mean_val = np.nanmean(values)

        if global_ci and 'lower_bound' in global_ci and 'upper_bound' in global_ci:
            # Use provided global CI
            global_mean = global_ci.get('mean', mean_val)
            global_lower = global_ci['lower_bound']
            global_upper = global_ci['upper_bound']

            # Add shaded confidence band
            ax.axhspan(global_lower, global_upper, alpha=0.15, color='red',
                      label=f'Global 95% CI: [{global_lower:.4f}, {global_upper:.4f}]')

            # Add mean line
            ax.axhline(y=global_mean, color='red', linestyle='--', linewidth=2,
                      label=f'Global Mean: {global_mean:.4f}')
        else:
            # Just add mean line without CI band
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.4f}')

        ax.set_xlabel('Test Day', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(days, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)

        # Add subtle background
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return temp_file.name

class PipelineExecutorHDF5:
    """
    Executes the end-to-end pipeline for the CLI, using X, y, and datetimes arrays (from HDF5).
    Integrates preprocessing, training, evaluation, and result saving.
    """
    def __init__(self, is_regression: bool, config: dict):
        self.is_regression = is_regression
        self.config = config

    def log_completion_instructions(self, joblib_path: str, csv_path: str):
        """Log actionable instructions for inspecting a successful MLflow run."""
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run is not None else "unknown"
        execution_id = self.config.get('execution_id', 'unknown')
        experiment_name = self.config.get(
            'mlflow_experiment_name', 'Marlinks-NS-DAS-dataset'
        )
        run_name = self.config.get('mlflow_run_name', 'unknown')
        tracking_uri = mlflow.get_tracking_uri()

        logger.info("Experiment completed successfully.")
        logger.info("MLflow experiment: %s", experiment_name)
        logger.info("MLflow run name: %s", run_name)
        logger.info("MLflow run ID: %s", run_id)
        logger.info("Execution ID: %s", execution_id)
        logger.info("Local joblib results: %s", os.path.abspath(joblib_path))
        if os.path.isfile(csv_path):
            logger.info("Local CSV results: %s", os.path.abspath(csv_path))
        logger.info("Local execution log: %s", self.config.get('log_file', 'unknown'))

        if tracking_uri.startswith(('http://', 'https://')):
            logger.info("MLflow is already configured at: %s", tracking_uri)
            logger.info("Open that URL in a web browser to inspect this run.")
        else:
            logger.info("Start the MLflow web UI from the repository root with:")
            logger.info("  python -m mlflow ui --backend-store-uri %s", tracking_uri)
            logger.info("Open in a browser: http://127.0.0.1:5000")
            logger.info("For a remote server, run this on your local computer:")
            logger.info("  ssh -L 5000:127.0.0.1:5000 <user>@<server>")
            logger.info("Then open http://127.0.0.1:5000 on your local computer.")

        logger.info(
            "In MLflow, select experiment '%s' and find run '%s' "
            "(MLflow run ID '%s', execution_id tag '%s').",
            experiment_name,
            run_name,
            run_id,
            execution_id,
        )
        logger.info(
            "Open Metrics for performance, Parameters for configuration, and "
            "Artifacts for results, plots, model files, and the execution log."
        )

    @staticmethod
    def extract_feature_suffix(h5_path: str) -> str:
        """
        Extract the substring between the last underscore and the '.hdf5' or '.h5' extension.

        Args:
            h5_path: Path to HDF5 file (e.g., 'dataset_sensor_range_1440_1690_-68.h5')

        Returns:
            str: Feature suffix (e.g., '-68') or empty string if not found

        Examples:
            >>> extract_feature_suffix('dataset_sensor_range_1440_1690_-68.h5')
            '-68'
            >>> extract_feature_suffix('dataset_sensor_range_1440_1690.h5')
            '1440_1690'
        """
        if not h5_path:
            return ''

        basename = os.path.basename(h5_path)

        # Remove .h5 or .hdf5 extension
        if basename.endswith('.hdf5'):
            name_without_ext = basename[:-5]
        elif basename.endswith('.h5'):
            name_without_ext = basename[:-3]
        else:
            name_without_ext = basename

        # Find last underscore and extract suffix
        if '_' in name_without_ext:
            suffix = name_without_ext.rsplit('_', 1)[-1]
            return suffix

        return ''

    def run(self):
        logger.info("Running HDF5 pipeline...")

        # Get prepared arrays from config
        X = self.config['X']
        y = self.config['y']
        datetimes = self.config['datetimes']
        instance_window = self.config.get('instance_window', None)
        n_seconds = self.config.get('n_seconds', None)
        freq_limit_joblib = self.config.get('freq_limit_joblib', None)
        center_ground_truth = self.config.get('center_truth', False)
        

        run_name = self.config.get('mlflow_run_name')
        if not run_name:
            # Compatibility fallback for callers that instantiate the executor
            # without going through the CLI.
            run_name = self.config.get('run_name', 'run')

        # Set MLflow tracking URI if provided
        mlflow_tracking_uri = self.config.get('mlflow_tracking_uri')
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        else:
            logger.info("MLflow tracking URI not set, using default (local mlruns folder)")

        mlflow.set_experiment(
            self.config.get('mlflow_experiment_name', 'Marlinks-NS-DAS-dataset')
        )

        # Extract feature_suffix from h5_path
        h5_path = self.config.get('h5_path', '')
        feature_suffix = self.extract_feature_suffix(h5_path)
        if feature_suffix:
            logger.info(f"Extracted feature_suffix: '{feature_suffix}' from h5_path: {h5_path}")
        else:
            logger.info(f"No feature_suffix found in h5_path: {h5_path}")

        # --- Optional: Apply reduction/averaging if needed ---
        # (For now, we assume X, y are ready.
        #  If you want to support time/channel averaging, add logic here.)



        # --- Split data by day (or other modes) ---
        splitter = Splitter(X, y, datetimes)
        test_date_range = self.config['test_date_range']

        with mlflow.start_run(run_name=run_name):
            execution_id = self.config.get('execution_id')
            if execution_id:
                mlflow.set_tag('execution_id', execution_id)
            params_to_log = {k: v for k, v in self.config.items() if k not in ['X', 'y', 'datetimes']}
            if params_to_log.get("regression_evaluation_threshold") is None:
                params_to_log.pop("regression_evaluation_threshold", None)
            # Add feature_suffix to MLflow parameters
            if feature_suffix:
                params_to_log['feature_suffix'] = feature_suffix
            mlflow.log_params(params_to_log)

            if isinstance(test_date_range, list) and len(test_date_range) == 2:
                start_day, end_day = test_date_range
                metrics_aggregator = MetricsAggregator(is_regression=self.is_regression)
                daily_metrics_list = []

                start_date = datetime.strptime(start_day, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_day, '%Y-%m-%d').date()
                total_folds = (end_date - start_date).days + 1
                for n in range(total_folds):
                    single_day = start_date + timedelta(days=n)
                    single_day_str = single_day.strftime('%Y-%m-%d')
                    logger.info("Processing day %s...", single_day_str)
                    X_train, X_test, y_train, y_test, dt_train, dt_test = splitter.split_by_day([single_day_str])
                    # Optionally save fold data to txt files
                    if self.config.get('save_fold_txt', False):
                        # This will either import `txt_saver` or raise:
                        txt_saver = optional_import(
                            "txt_saver",
                            message="The txt-saving option is not available in the public version."
                        )                    
                        ds_name = os.path.splitext(os.path.basename(self.config['h5_path']))[0]
                        txt_saver.FoldTextSaver.save_fold(
                            day_str=single_day_str,
                            fold_idx=n,
                            X=X_test,
                            y=y_test,
                            dt=dt_test,
                            dataset_name=ds_name,
                            n_seconds=self.config.get('n_seconds', 10),
                            average_signals=self.config.get('average_signals'),
                            is_regression=self.is_regression
                        )

                    if not self.is_regression:
                        balancer = Balancer(X_train, y_train, dt_train)
                        X_train, y_train, dt_train = balancer.balance(
                            self.config['balance_classes'],
                            random_state=self.config['random_state']
                        )

                    # TRAINING (XGBoost or NN based on CLI arg)
                    model = self.train_model(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        fold_label=f"{n + 1}/{total_folds}",
                    )

                    # EVALUATION
                    if self.is_regression:
                        evaluator = ModelRegressionEvaluator(
                            model,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            None,
                            dt_train=dt_train, 
                            dt_test=dt_test,
                            instance_window=instance_window,
                            freq_limit_joblib=freq_limit_joblib,
                            compute_daywise_bootstrap=self.config.get('compute_daywise_bootstrap', False),
                            center_ground_truth=center_ground_truth,
                            include_predictions=self.config.get('include_predictions', True),
                            regression_evaluation_threshold=self.config.get(
                                'regression_evaluation_threshold'
                            ),
                        )
                        day_metrics = evaluator.evaluate_on_test_set()
                        daily_metrics_list.append(day_metrics['MAE'])
                   
                    else: # For classification
                        evaluator = ModelEvaluator(
                            model,
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            None,
                            dt_train=dt_train,  
                            dt_test=dt_test,    
                            instance_window=instance_window,
                            freq_limit_joblib=freq_limit_joblib,
                            compute_daywise_bootstrap=self.config.get('compute_daywise_bootstrap', False),
                            center_ground_truth=center_ground_truth,
                            include_predictions=self.config.get('include_predictions', True) # Also good to add for consistency
                        )
                        day_metrics = evaluator.evaluate_on_test_set()
                        daily_metrics_list.append(day_metrics['accuracy'])

                    # tag day for CSV output
                    day_metrics['day'] = single_day_str
                    metrics_aggregator.add_metrics(day_metrics)
                    log_performance_results(
                        day_metrics,
                        is_regression=self.is_regression,
                        scope="daily evaluation",
                        day=single_day_str,
                        fold_label=f"{n + 1}/{total_folds}",
                    )

                    # Clean up GPU memory between days to prevent fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # AVERAGING + CONFIDENCE INTERVAL
                aggregated_metrics = metrics_aggregator.compute_averages()
                metric_key = 'MAE' if self.is_regression else 'accuracy'
                ci_calculator = ConfidenceIntervalCalculator()
                aggregated_metrics["confidence_intervals"] = {
                    metric_key: ci_calculator.calculate(
                        metric_values=np.array(daily_metrics_list),
                        total_instances=len(y)
                    )
                }
                aggregated_metrics['metrics_by_day'] = metrics_aggregator.metrics

                # --- now compute “final_results” ---
                from model_csv_report import (
                    compute_global_accuracy, compute_weighted_f1,
                    compute_f1_for_class, bootstrap_metric,
                    bootstrap_metric_numbers
                )

                
                # === FINAL_RESULTS: branch by regression vs classification ===
                if not self.is_regression:
                    # — classification path —
                    # build confusion matrices list:
                    cms = [np.array(d['confusion_matrix']).reshape(2,2)
                        for d in metrics_aggregator.metrics]
                    # global confusion matrix
                    global_cm = sum(cms)

                    def class_precision(cm, cls):
                        true_positive = cm[cls, cls]
                        predicted_positive = cm[:, cls].sum()
                        return true_positive / predicted_positive if predicted_positive else 0.0

                    def class_recall(cm, cls):
                        true_positive = cm[cls, cls]
                        actual_positive = cm[cls, :].sum()
                        return true_positive / actual_positive if actual_positive else 0.0

                    def support_weighted(cm, metric_func):
                        supports = cm.sum(axis=1)
                        total_support = supports.sum()
                        if not total_support:
                            return 0.0
                        return sum(
                            metric_func(cm, cls) * supports[cls]
                            for cls in range(cm.shape[0])
                        ) / total_support

                    def weighted_precision(cm):
                        return support_weighted(cm, class_precision)
                    def weighted_recall(cm):
                        return support_weighted(cm, class_recall)
                    # metrics + CI
                    f1_val = compute_weighted_f1(global_cm)
                    f1_ci  = bootstrap_metric(cms, compute_weighted_f1)
                    precision_val = weighted_precision(global_cm)
                    precision_ci = bootstrap_metric(cms, weighted_precision)
                    recall_val = weighted_recall(global_cm)
                    recall_ci = bootstrap_metric(cms, weighted_recall)
                    acc_val = compute_global_accuracy(global_cm)
                    acc_ci  = bootstrap_metric(cms, compute_global_accuracy)

                    # Compute AUC if available
                    auc_values = [d.get('auc', None) for d in metrics_aggregator.metrics if 'auc' in d]
                    if auc_values:
                        auc_mean = float(np.mean(auc_values))
                        auc_std = float(np.std(auc_values))
                        # Bootstrap CI for AUC
                        rng = np.random.RandomState(self.config.get('random_state', 42))
                        boot_aucs = []
                        for _ in range(1000):
                            idx = rng.choice(len(auc_values), size=len(auc_values), replace=True)
                            boot_aucs.append(np.mean([auc_values[i] for i in idx]))
                        auc_ci_lo, auc_ci_hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (0.0, 0.0)
                        auc_with_ci = (auc_mean, (auc_ci_lo, auc_ci_hi))
                    else:
                        auc_with_ci = None

                    # Compute precision and recall per class from confusion matrix
                    # Class 0: TN, FP / FN, TP in 2x2 matrix [[TN, FP], [FN, TP]]
                    tn, fp, fn, tp = global_cm[0,0], global_cm[0,1], global_cm[1,0], global_cm[1,1]

                    # Class 1 precision and recall
                    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                    # Class 0 precision and recall
                    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                    # Weighted average
                    total = tn + fp + fn + tp
                    support_0 = tn + fp
                    support_1 = fn + tp
                    precision_weighted = (precision_0 * support_0 + precision_1 * support_1) / total if total > 0 else 0.0
                    recall_weighted = (recall_0 * support_0 + recall_1 * support_1) / total if total > 0 else 0.0

                    # per‐class F1
                    per_class = {}
                    per_class_precision = {}
                    per_class_recall = {}
                    for cls in (0,1):
                        c_val = compute_f1_for_class(global_cm, cls)
                        c_ci  = bootstrap_metric(
                            cms,
                            lambda cm, cls=cls: compute_f1_for_class(cm, cls)
                        )
                        per_class[str(cls)] = (c_val, c_ci)
                        def precision_func(cm, cls=cls):
                            return class_precision(cm, cls)
                        def recall_func(cm, cls=cls):
                            return class_recall(cm, cls)
                        per_class_precision[str(cls)] = (
                            precision_func(global_cm), bootstrap_metric(cms, precision_func)
                        )
                        per_class_recall[str(cls)] = (
                            recall_func(global_cm), bootstrap_metric(cms, recall_func)
                        )

                    aggregated_metrics['final_results'] = {
                        'f1':        (f1_val,  f1_ci),
                        'precision': (precision_val, precision_ci),
                        'recall':    (recall_val, recall_ci),
                        'accuracy':  (acc_val, acc_ci),
                        'per_class': per_class,
                        'per_class_precision': per_class_precision,
                        'per_class_recall': per_class_recall,
                    }
                    if auc_with_ci:
                        aggregated_metrics['final_results']['auc'] = auc_with_ci

                    log_performance_results(
                        aggregated_metrics,
                        is_regression=False,
                        scope="global date-range evaluation",
                    )
                    logger.info("  Precision (weighted): %.4f", precision_weighted)
                    logger.info("  Recall (weighted):    %.4f", recall_weighted)

                    classification_metrics_to_log = {
                        'global_f1': f1_val,
                        'global_f1_low_CI': f1_ci[0],
                        'global_f1_high_CI': f1_ci[1],
                        'global_accuracy': acc_val,
                        'global_accuracy_low_CI': acc_ci[0],
                        'global_accuracy_high_CI': acc_ci[1],
                        'precision_weighted': precision_weighted,
                        'recall_weighted': recall_weighted,
                        'precision_class_0': precision_0,
                        'recall_class_0': recall_0,
                        'precision_class_1': precision_1,
                        'recall_class_1': recall_1,
                        'n_test_samples': int(total),
                        'support_class_0': int(support_0),
                        'support_class_1': int(support_1),
                    }

                    if auc_with_ci:
                        classification_metrics_to_log['global_auc'] = auc_with_ci[0]
                        classification_metrics_to_log['global_auc_low_CI'] = auc_with_ci[1][0]
                        classification_metrics_to_log['global_auc_high_CI'] = auc_with_ci[1][1]
                        classification_metrics_to_log['auc_std'] = auc_std

                    for cls, (v, (lo, hi)) in per_class.items():
                        classification_metrics_to_log[f"class_{cls}_f1"] = v
                        classification_metrics_to_log[f"class_{cls}_f1_low_CI"] = lo
                        classification_metrics_to_log[f"class_{cls}_f1_high_CI"] = hi

                    # Add per-day average metrics
                    if len(metrics_aggregator.metrics) > 0:
                        avg_acc = np.mean([d.get('accuracy', 0) for d in metrics_aggregator.metrics])
                        classification_metrics_to_log['avg_accuracy'] = float(avg_acc)
                        if auc_values:
                            classification_metrics_to_log['avg_auc'] = float(np.mean(auc_values))

                    mlflow.log_metrics(classification_metrics_to_log)

                    # Create and log confusion matrix visualization
                    try:
                        artifact_creator = MLflowArtifactCreator()
                        cm_plot = artifact_creator.create_confusion_matrix_plot(
                            global_cm,
                            class_names=['No Vessel (>threshold)', 'Vessel (<threshold)'],
                            title='Global Confusion Matrix'
                        )
                        mlflow.log_artifact(cm_plot, artifact_path="confusion_matrix")
                        os.unlink(cm_plot)
                        logger.info("Confusion matrix plot logged to MLflow")
                    except Exception as e:
                        logger.warning(f"Failed to create confusion matrix plot: {e}")

                    # Create and log per-day accuracy/AUC plots
                    if len(metrics_aggregator.metrics) > 1:
                        try:
                            artifact_creator = MLflowArtifactCreator()

                            # Extract global CI for accuracy from final_results
                            accuracy_global_ci = None
                            if 'final_results' in aggregated_metrics and 'accuracy' in aggregated_metrics['final_results']:
                                acc_val, (acc_ci_lo, acc_ci_hi) = aggregated_metrics['final_results']['accuracy']
                                accuracy_global_ci = {
                                    'mean': acc_val,
                                    'lower_bound': acc_ci_lo,
                                    'upper_bound': acc_ci_hi
                                }

                            acc_plot = artifact_creator.create_per_day_metrics_plot(
                                metrics_aggregator.metrics, 'accuracy',
                                'Accuracy', 'Per-Day Accuracy Performance',
                                global_ci=accuracy_global_ci
                            )
                            mlflow.log_artifact(acc_plot, artifact_path="per_day_metrics")
                            os.unlink(acc_plot)

                            if auc_values:
                                # Extract global CI for AUC if available
                                auc_global_ci = None
                                if 'final_results' in aggregated_metrics and 'auc' in aggregated_metrics['final_results']:
                                    auc_val, (auc_ci_lo, auc_ci_hi) = aggregated_metrics['final_results']['auc']
                                    auc_global_ci = {
                                        'mean': auc_val,
                                        'lower_bound': auc_ci_lo,
                                        'upper_bound': auc_ci_hi
                                    }

                                auc_plot = artifact_creator.create_per_day_metrics_plot(
                                    metrics_aggregator.metrics, 'auc',
                                    'AUC (ROC)', 'Per-Day AUC Performance',
                                    global_ci=auc_global_ci
                                )
                                mlflow.log_artifact(auc_plot, artifact_path="per_day_metrics")
                                os.unlink(auc_plot)

                            logger.info("Per-day classification metrics plots logged to MLflow")
                        except Exception as e:
                            logger.warning(f"Failed to create per-day metrics plots: {e}")                  

                else:
                    # --- regression via residual bootstrap across days/folds ---
                    daily_res = [d.get('residuals', np.array([])) for d in metrics_aggregator.metrics]
                    all_res = np.concatenate(daily_res) if daily_res else np.array([])
                    n_res = len(all_res)
                    rng = np.random.RandomState(self.config.get('random_state', 42))

                    # Bootstrap for multiple metrics
                    boot_maes, boot_rmses, boot_r2s, boot_mses = [], [], [], []

                    # Collect all true and predicted values for R2 bootstrap
                    all_y_true = []
                    all_y_pred = []
                    for d in metrics_aggregator.metrics:
                        if 'y_true' in d and 'y_pred' in d:
                            all_y_true.append(d['y_true'])
                            all_y_pred.append(d['y_pred'])

                    if all_y_true:
                        all_y_true = np.concatenate(all_y_true)
                        all_y_pred = np.concatenate(all_y_pred)
                        all_res = all_y_true - all_y_pred

                    for _ in range(1000):
                        idx = rng.choice(n_res, size=n_res, replace=True)
                        boot_res = all_res[idx]
                        boot_maes.append(np.mean(np.abs(boot_res)))
                        boot_rmses.append(np.sqrt(np.mean(boot_res ** 2)))
                        boot_mses.append(np.mean(boot_res ** 2))

                        # R2 bootstrap (if we have true and predicted values)
                        if all_y_true is not None and len(all_y_true) > 0:
                            y_true_boot = all_y_true[idx]
                            y_pred_boot = all_y_pred[idx]
                            ss_res = np.sum((y_true_boot - y_pred_boot) ** 2)
                            ss_tot = np.sum((y_true_boot - np.mean(y_true_boot)) ** 2)
                            r2_boot = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            boot_r2s.append(r2_boot)

                    # Compute CIs
                    mae_ci_lo, mae_ci_hi = np.percentile(boot_maes, [2.5, 97.5]) if boot_maes else (0.0, 0.0)
                    rmse_ci_lo, rmse_ci_hi = np.percentile(boot_rmses, [2.5, 97.5]) if boot_rmses else (0.0, 0.0)
                    mse_ci_lo, mse_ci_hi = np.percentile(boot_mses, [2.5, 97.5]) if boot_mses else (0.0, 0.0)
                    r2_ci_lo, r2_ci_hi = np.percentile(boot_r2s, [2.5, 97.5]) if boot_r2s else (0.0, 0.0)

                    mae_mean = float(np.mean(boot_maes)) if boot_maes else 0.0
                    rmse_mean = float(np.mean(boot_rmses)) if boot_rmses else 0.0
                    mse_mean = float(np.mean(boot_mses)) if boot_mses else 0.0
                    r2_mean = float(np.mean(boot_r2s)) if boot_r2s else 0.0

                    aggregated_metrics['final_results'] = {
                        "MAE": (mae_mean, (mae_ci_lo, mae_ci_hi)),
                        "RMSE": (rmse_mean, (rmse_ci_lo, rmse_ci_hi)),
                        "MSE": (mse_mean, (mse_ci_lo, mse_ci_hi)),
                        "R2": (r2_mean, (r2_ci_lo, r2_ci_hi))
                    }

                    evaluation_threshold = self.config.get(
                        'regression_evaluation_threshold'
                    )
                    if evaluation_threshold is not None:
                        threshold_result = aggregate_regression_threshold_metrics(
                            metrics_aggregator.metrics,
                            evaluation_threshold,
                            self.config.get('random_state', 42),
                        )
                        aggregated_metrics[
                            'regression_threshold_evaluation'
                        ] = threshold_result

                    log_performance_results(
                        aggregated_metrics,
                        is_regression=True,
                        scope="global date-range evaluation",
                    )

                    # Prepare metrics for MLflow
                    regression_metrics_to_log = {
                        'global_MAE': mae_mean,
                        'global_MAE_low_CI': mae_ci_lo,
                        'global_MAE_high_CI': mae_ci_hi,
                        'global_RMSE': rmse_mean,
                        'global_RMSE_low_CI': rmse_ci_lo,
                        'global_RMSE_high_CI': rmse_ci_hi,
                        'global_MSE': mse_mean,
                        'global_MSE_low_CI': mse_ci_lo,
                        'global_MSE_high_CI': mse_ci_hi,
                        'global_R2': r2_mean,
                        'global_R2_low_CI': r2_ci_lo,
                        'global_R2_high_CI': r2_ci_hi,
                    }

                    # Add per-day average metrics
                    avg_metrics = aggregated_metrics.get('average_metrics', {})
                    for key, val in avg_metrics.items():
                        if key != 'SUPPORT' and isinstance(val, (int, float)):
                            regression_metrics_to_log[f'avg_{key}'] = float(val)

                    # Add fold-weighted metrics
                    fold_weighted = aggregated_metrics.get('fold_weighted_metrics', {})
                    for key, val in fold_weighted.items():
                        if key != 'SUPPORT' and isinstance(val, (int, float)):
                            regression_metrics_to_log[f'fold_weighted_{key}'] = float(val)

                    # Add dataset statistics
                    if all_y_true is not None and len(all_y_true) > 0:
                        regression_metrics_to_log['n_test_samples'] = len(all_y_true)
                        regression_metrics_to_log['mean_true_distance'] = float(np.mean(all_y_true))
                        regression_metrics_to_log['std_true_distance'] = float(np.std(all_y_true))
                        regression_metrics_to_log['median_true_distance'] = float(np.median(all_y_true))

                        # Residual statistics
                        regression_metrics_to_log['residual_mean'] = float(np.mean(all_res))
                        regression_metrics_to_log['residual_std'] = float(np.std(all_res))
                        regression_metrics_to_log['residual_skewness'] = float(np.mean([d.get('RESIDUAL_SKEWNESS', 0)
                                                                                         for d in metrics_aggregator.metrics]))
                        regression_metrics_to_log['residual_kurtosis'] = float(np.mean([d.get('RESIDUAL_KURTOSIS', 0)
                                                                                         for d in metrics_aggregator.metrics]))

                    mlflow.log_metrics(regression_metrics_to_log)
                    if self.config.get('regression_evaluation_threshold') is not None:
                        mlflow.log_metrics(mlflow_threshold_metrics(
                            aggregated_metrics['regression_threshold_evaluation']
                        ))

                    # Create and log residual analysis plots
                    if all_y_true is not None and len(all_y_true) > 0:
                        try:
                            artifact_creator = MLflowArtifactCreator()
                            residual_plots = artifact_creator.create_residual_plots(
                                all_y_true, all_y_pred, title_prefix="Vessel Distance Regression"
                            )
                            for plot_name, plot_path in residual_plots.items():
                                mlflow.log_artifact(plot_path, artifact_path="residual_analysis")
                                os.unlink(plot_path)  # Clean up temp file
                            logger.info("Residual analysis plots logged to MLflow")
                        except Exception as e:
                            logger.warning(f"Failed to create residual plots: {e}")

                    # Create and log per-day metrics plot
                    if len(metrics_aggregator.metrics) > 1:
                        try:
                            artifact_creator = MLflowArtifactCreator()

                            # Extract global CIs from final_results
                            mae_global_ci = None
                            rmse_global_ci = None
                            r2_global_ci = None
                            mse_global_ci = None

                            if 'final_results' in aggregated_metrics:
                                if 'MAE' in aggregated_metrics['final_results']:
                                    mae_val, (mae_ci_lo, mae_ci_hi) = aggregated_metrics['final_results']['MAE']
                                    mae_global_ci = {
                                        'mean': mae_val,
                                        'lower_bound': mae_ci_lo,
                                        'upper_bound': mae_ci_hi
                                    }

                                if 'RMSE' in aggregated_metrics['final_results']:
                                    rmse_val, (rmse_ci_lo, rmse_ci_hi) = aggregated_metrics['final_results']['RMSE']
                                    rmse_global_ci = {
                                        'mean': rmse_val,
                                        'lower_bound': rmse_ci_lo,
                                        'upper_bound': rmse_ci_hi
                                    }

                                if 'R2' in aggregated_metrics['final_results']:
                                    r2_val, (r2_ci_lo, r2_ci_hi) = aggregated_metrics['final_results']['R2']
                                    r2_global_ci = {
                                        'mean': r2_val,
                                        'lower_bound': r2_ci_lo,
                                        'upper_bound': r2_ci_hi
                                    }

                                if 'MSE' in aggregated_metrics['final_results']:
                                    mse_val, (mse_ci_lo, mse_ci_hi) = aggregated_metrics['final_results']['MSE']
                                    mse_global_ci = {
                                        'mean': mse_val,
                                        'lower_bound': mse_ci_lo,
                                        'upper_bound': mse_ci_hi
                                    }

                            # Create MAE plot with CI
                            mae_plot = artifact_creator.create_per_day_metrics_plot(
                                metrics_aggregator.metrics, 'MAE',
                                'MAE (meters)', 'Per-Day MAE Performance',
                                global_ci=mae_global_ci
                            )
                            mlflow.log_artifact(mae_plot, artifact_path="per_day_metrics")
                            os.unlink(mae_plot)

                            # Create RMSE plot with CI
                            rmse_plot = artifact_creator.create_per_day_metrics_plot(
                                metrics_aggregator.metrics, 'RMSE',
                                'RMSE (meters)', 'Per-Day RMSE Performance',
                                global_ci=rmse_global_ci
                            )
                            mlflow.log_artifact(rmse_plot, artifact_path="per_day_metrics")
                            os.unlink(rmse_plot)

                            # Create R² plot with CI
                            r2_plot = artifact_creator.create_per_day_metrics_plot(
                                metrics_aggregator.metrics, 'R2',
                                'R² Score', 'Per-Day R² Performance',
                                global_ci=r2_global_ci
                            )
                            mlflow.log_artifact(r2_plot, artifact_path="per_day_metrics")
                            os.unlink(r2_plot)

                            logger.info("Per-day metrics plots logged to MLflow")
                        except Exception as e:
                            logger.warning(f"Failed to create per-day metrics plots: {e}")   

                # save joblib and CSV of metrics
                joblib_path = JoblibSaver.save(
                    self.config['joblib_save_file'],
                    aggregated_metrics,
                    self.config,
                    self.config.get('output_suffix', '')
                )
                csv_path = joblib_path.rsplit('.',1)[0] + '.csv'
                CsvSaver.save(joblib_path, aggregated_metrics)

                # Log joblib and CSV to MLflow
                try:
                    mlflow.log_artifact(joblib_path, artifact_path="results")
                    if os.path.exists(csv_path):
                        mlflow.log_artifact(csv_path, artifact_path="results")
                    logger.info("Results artifacts logged to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log results artifacts to MLflow: {e}")

                # Log the trained model to MLflow (last trained model from the loop)
                if 'model' in locals() and model is not None:
                    try:
                        model_path = joblib_path.rsplit('.',1)[0] + '_model.joblib'
                        dump(model, model_path)
                        mlflow.log_artifact(model_path, artifact_path="model")
                        logger.info(f"Trained model logged to MLflow: {model_path}")
                        # Clean up local model file optionally
                        # os.unlink(model_path)
                    except Exception as e:
                        logger.warning(f"Failed to log model to MLflow: {e}")

                # console summary of saved artifacts
                logger.info("Results saved to %s", joblib_path)
                logger.info("CSV report saved to %s", csv_path)
                logger.info(
                    "Artifacts logged to MLflow run: %s",
                    mlflow.active_run().info.run_id,
                )


            else:
                logger.info("Processing single test day...")
                single_day = test_date_range[0] if isinstance(test_date_range, list) else test_date_range
                X_train, X_test, y_train, y_test, dt_train, dt_test = splitter.split_by_day(single_day)

                if not self.is_regression:
                    balancer = Balancer(X_train, y_train, dt_train)
                    X_train, y_train, dt_train = balancer.balance(
                        self.config['balance_classes'],
                        random_state=self.config['random_state']
                    )

                # TRAINING (XGBoost or NN)
                model = self.train_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    fold_label="1/1",
                )

                # EVALUATION
                if self.is_regression:
                    evaluator = ModelRegressionEvaluator(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        None,
                        dt_train=dt_train, 
                        dt_test=dt_test,
                        instance_window=instance_window,
                        freq_limit_joblib=freq_limit_joblib,
                        compute_daywise_bootstrap=self.config.get('compute_daywise_bootstrap', False),
                        center_ground_truth=center_ground_truth,
                        include_predictions=self.config.get('include_predictions', True),
                        regression_evaluation_threshold=self.config.get(
                            'regression_evaluation_threshold'
                        ),
                    )
                else:
                    evaluator = ModelEvaluator(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        None,
                        instance_window=instance_window,
                        freq_limit_joblib=freq_limit_joblib,
                        compute_daywise_bootstrap=self.config.get('compute_daywise_bootstrap', False),
                        center_ground_truth=center_ground_truth
                    )
                aggregated_metrics = evaluator.evaluate_on_test_set()
                if (
                    self.is_regression
                    and self.config.get('regression_evaluation_threshold') is not None
                ):
                    aggregated_metrics['regression_threshold_evaluation'] = (
                        aggregate_regression_threshold_metrics(
                            [aggregated_metrics],
                            self.config['regression_evaluation_threshold'],
                            self.config.get('random_state', 42),
                        )
                    )
                    mlflow.log_metrics(mlflow_threshold_metrics(
                        aggregated_metrics['regression_threshold_evaluation']
                    ))
                log_performance_results(
                    aggregated_metrics,
                    is_regression=self.is_regression,
                    scope="single-day evaluation",
                    day=str(single_day),
                    fold_label="1/1",
                )
                
                


            joblib_path = JoblibSaver.save(
                file_path=self.config['joblib_save_file'],
                data=aggregated_metrics,
                metadata=self.config,
                output_suffix=self.config.get('output_suffix', '')
            )
            CsvSaver.save(joblib_path, aggregated_metrics)
            csv_path = os.path.splitext(joblib_path)[0] + '.csv'
            if self.config.get('regression_evaluation_threshold') is not None:
                try:
                    mlflow.log_artifact(joblib_path, artifact_path="results")
                    if os.path.exists(csv_path):
                        mlflow.log_artifact(csv_path, artifact_path="results")
                except Exception as e:
                    logger.warning(
                        "Failed to log threshold-evaluation result artifacts to MLflow: %s",
                        e,
                    )
            self.log_completion_instructions(joblib_path, csv_path)
            for handler in logging.getLogger().handlers:
                handler.flush()

            log_file = self.config.get('log_file')
            if log_file and os.path.isfile(log_file):
                try:
                    mlflow.log_artifact(log_file, artifact_path="logs")
                except Exception as e:
                    logger.warning("Failed to upload execution log to MLflow: %s", e)

    def train_model(self, X_train, y_train, X_test, y_test, fold_label="1/1"):
        """
        Selects and trains either an XGBoost/LightGBM model or a neural network,
        based on CLI/config args and using the user's model_file.
        """
        if self.config.get('is_NN', False):
            # Dynamically load user's load_model function
            load_model = self.dynamic_load_model(self.config['model_file'])
            nn_handler = NeuralNetworkHandler(
                load_model,
                hidden_dim=self.config.get('nn_hidden_dim', 256),
                batch_size=self.config.get('nn_batch_size', 32),
                patience=self.config.get('nn_patience', 20),
                lr=self.config.get('nn_lr', 0.001),
                is_regression=self.is_regression,
                fold_label=fold_label,
            )
            nn_handler.train(
                X_train, y_train,
                epochs=self.config.get('nn_epochs', 100),
                seed=self.config['random_state']
            )
            return nn_handler.model
        else:
            # **LOAD YOUR CUSTOM MODEL** instead of default
            load_model_fn = self.dynamic_load_model(self.config['model_file'])
            model = load_model_fn()
            
            # split train/val for early stopping
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.config['random_state']
            )
            
            import inspect
            fit_params = inspect.signature(model.fit).parameters
            fit_kwargs = {}
            
            if 'eval_set' in fit_params:
                fit_kwargs['eval_set'] = [(X_val, y_val)]
            
            fit_args = [X_train_split, y_train_split]
            
            if 'verbose' in fit_params:
                try:
                    # Try with verbose=False (common for bool-accepting models like XGBoost)
                    model.fit(*fit_args, **fit_kwargs, verbose=False)
                    return model
                except TypeError:
                    # If that fails, it might expect an int (like LightGBM or StackingClassifier)
                    pass
                
                try:
                    # Try with verbose=-1 (common for int-accepting models for silence)
                    model.fit(*fit_args, **fit_kwargs, verbose=-1)
                    return model
                except TypeError:
                    # If both fail, fall back to fitting without the verbose argument
                    pass

            # Fit without verbose, or if both verbose attempts failed
            model.fit(*fit_args, **fit_kwargs)
            return model

    @staticmethod
    def dynamic_load_model(model_file_path):
        """
        Loads the user-supplied model_file.py containing a load_model function.
        Returns the function object.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model", model_file_path)
        user_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_model)
        return user_model.load_model
