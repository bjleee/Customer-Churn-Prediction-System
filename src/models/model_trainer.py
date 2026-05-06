"""Reusable model training utilities for the churn prediction project."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.exception_handler import CustomError  # noqa: E402
from src.utils.logging_handler import LOGGER  # noqa: E402

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

_DATA_DIR = _PROJECT_ROOT / "data"
_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts"
_TARGET_MAPPING = {"No": 0, "Yes": 1}


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for dataset paths, artifacts, and training behavior."""

    train_data_path: Path = field(default_factory=lambda: _DATA_DIR / "train" / "train.csv")
    test_data_path: Path = field(default_factory=lambda: _DATA_DIR / "test" / "test.csv")
    target_column: str = "Churn"
    drop_columns: tuple[str, ...] = ("customerID",)
    random_state: int = 42
    artifact_dir: Path = field(default_factory=lambda: _ARTIFACT_DIR)
    best_model_path: Path = field(
        default_factory=lambda: _ARTIFACT_DIR / "best_model.joblib"
    )
    metrics_path: Path = field(default_factory=lambda: _ARTIFACT_DIR / "metrics.csv")


@dataclass(frozen=True)
class TrainingArtifact:
    """Result bundle produced by the trainer."""

    best_model_name: str
    best_model_path: Path
    metrics_path: Path
    metrics_frame: pd.DataFrame


class ModelTrainer:
    """Builds preprocessing pipelines, trains candidate models, and persists the best."""

    def __init__(self, config: ModelTrainerConfig | None = None) -> None:
        """Initializes the trainer with default or provided configuration."""
        self.config = config or ModelTrainerConfig()

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        """Loads a CSV dataset from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return pd.read_csv(path)

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Applies notebook-aligned cleaning rules before model training."""
        prepared = dataframe.copy()

        if "TotalCharges" in prepared.columns:
            prepared["TotalCharges"] = pd.to_numeric(
                prepared["TotalCharges"], errors="coerce"
            )

        string_columns = prepared.select_dtypes(include=["object", "string"]).columns
        for column in string_columns:
            prepared[column] = prepared[column].astype(str).str.strip()

        if self.config.target_column not in prepared.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' is missing from the dataset."
            )

        if prepared[self.config.target_column].dtype == "object":
            prepared[self.config.target_column] = prepared[self.config.target_column].map(
                _TARGET_MAPPING
            )

        prepared = prepared.dropna(subset=[self.config.target_column]).copy()
        return prepared

    def _split_features_and_target(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separates the target column from training features."""
        feature_frame = dataframe.drop(
            columns=[self.config.target_column, *self.config.drop_columns],
            errors="ignore",
        )
        target_series = dataframe[self.config.target_column].astype(int)
        return feature_frame, target_series

    def _build_preprocessor(self, features: pd.DataFrame) -> ColumnTransformer:
        """Constructs a preprocessing block for numeric and categorical features."""
        categorical_columns = (
            features.select_dtypes(include=["object", "category", "string"])
            .columns.tolist()
        )
        numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("cat", categorical_pipeline, categorical_columns),
                ("num", numeric_pipeline, numeric_columns),
            ],
            remainder="drop",
        )

    def _build_candidate_models(self, y_train: pd.Series) -> dict[str, Any]:
        """Creates candidate estimators for model selection."""
        candidates: dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.config.random_state,
                solver="liblinear",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.config.random_state
            ),
        }

        if XGBClassifier is not None:
            negative_count, positive_count = y_train.value_counts().reindex(
                [0, 1], fill_value=0
            )
            scale_pos_weight = (
                float(negative_count) / float(positive_count)
                if positive_count
                else 1.0
            )
            candidates["xgboost"] = XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                n_jobs=-1,
            )

        return candidates

    def _evaluate_pipeline(
        self,
        model_name: str,
        pipeline: Pipeline,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float | str]:
        """Fits a pipeline and returns evaluation metrics."""
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probabilities)
        else:  # pragma: no cover
            roc_auc = float("nan")

        metrics: dict[str, float | str] = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc,
        }
        LOGGER.info("Finished training %s with roc_auc=%.4f", model_name, roc_auc)
        return metrics

    def initiate_model_training(self) -> TrainingArtifact:
        """Runs end-to-end training and saves the best model artifact."""
        try:
            LOGGER.info("Loading training data from %s", self.config.train_data_path)
            LOGGER.info("Loading test data from %s", self.config.test_data_path)

            train_dataframe = self._prepare_dataframe(
                self._load_dataframe(self.config.train_data_path)
            )
            test_dataframe = self._prepare_dataframe(
                self._load_dataframe(self.config.test_data_path)
            )

            x_train, y_train = self._split_features_and_target(train_dataframe)
            x_test, y_test = self._split_features_and_target(test_dataframe)

            preprocessor = self._build_preprocessor(x_train)
            candidate_models = self._build_candidate_models(y_train)

            trained_pipelines: dict[str, Pipeline] = {}
            metric_rows: list[dict[str, float | str]] = []

            for model_name, estimator in candidate_models.items():
                pipeline = Pipeline(
                    steps=[("preprocess", preprocessor), ("model", estimator)]
                )
                trained_pipelines[model_name] = pipeline
                metric_rows.append(
                    self._evaluate_pipeline(
                        model_name=model_name,
                        pipeline=pipeline,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                    )
                )

            metrics_frame = pd.DataFrame(metric_rows).sort_values(
                by="roc_auc", ascending=False
            )
            best_model_name = str(metrics_frame.iloc[0]["model"])
            best_pipeline = trained_pipelines[best_model_name]

            self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_pipeline, self.config.best_model_path)
            metrics_frame.to_csv(self.config.metrics_path, index=False)

            LOGGER.info("Saved best model '%s' to %s", best_model_name, self.config.best_model_path)
            LOGGER.info("Saved metrics to %s", self.config.metrics_path)

            return TrainingArtifact(
                best_model_name=best_model_name,
                best_model_path=self.config.best_model_path,
                metrics_path=self.config.metrics_path,
                metrics_frame=metrics_frame,
            )
        except Exception as error:
            LOGGER.exception("Model training failed")
            raise CustomError(error) from error


if __name__ == "__main__":
    artifact = ModelTrainer().initiate_model_training()
    print(f"Best model: {artifact.best_model_name}")
    print(f"Saved model: {artifact.best_model_path}")
    print(f"Saved metrics: {artifact.metrics_path}")
