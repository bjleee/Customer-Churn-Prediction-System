"""Data ingestion utilities for reading raw data and creating train/test splits."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exception_handler import CustomError
from src.utils.logging_handler import LOGGER

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration values for data ingestion paths and split behavior."""

    raw_data_path: Path = field(default_factory=lambda: _DATA_DIR / "raw" / "data.csv")
    train_data_path: Path = field(
        default_factory=lambda: _DATA_DIR / "train" / "train.csv"
    )
    test_data_path: Path = field(
        default_factory=lambda: _DATA_DIR / "test" / "test.csv"
    )
    test_size: float = 0.2
    random_state: int = 42


class DataIngestor:
    """Handles raw data loading and train/test data export."""

    def __init__(self, config: DataIngestionConfig | None = None) -> None:
        """Initializes the ingestor with default or provided configuration.

        Args:
            config: Optional ingestion configuration.
        """
        self.data_config = config or DataIngestionConfig()

    def init_data_file(self) -> tuple[Path, Path]:
        """Reads raw CSV data, splits it, and writes train/test files.

        Returns:
            Tuple of `(train_data_path, test_data_path)`.

        Raises:
            CustomError: If reading, splitting, or writing data fails.
        """
        try:
            LOGGER.info("Reading raw data from: %s", self.data_config.raw_data_path)
            dataframe = pd.read_csv(self.data_config.raw_data_path)

            train_data, test_data = train_test_split(
                dataframe,
                test_size=self.data_config.test_size,
                random_state=self.data_config.random_state,
            )

            self.data_config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_config.test_data_path.parent.mkdir(parents=True, exist_ok=True)

            train_data.to_csv(self.data_config.train_data_path, index=False)
            test_data.to_csv(self.data_config.test_data_path, index=False)

            LOGGER.info("Saved train data to: %s", self.data_config.train_data_path)
            LOGGER.info("Saved test data to: %s", self.data_config.test_data_path)

            return self.data_config.train_data_path, self.data_config.test_data_path

        except Exception as error:
            LOGGER.exception("Data ingestion failed")
            raise CustomError(error) from error
