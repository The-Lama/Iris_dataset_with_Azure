import pandas as pd
from pathlib import Path


def read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and ensures it is not empty."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError(
            f"The file {file_path} is empty. Please provide a valid dataset."
        )

    return data


def load_train_and_test_data(file_path: str) -> pd.DataFrame:
    """
    Read the data from the files train.csv and test.csv.

    Return the data in form of a Dataframe.
    """
    file_path = Path(file_path)
    train_file = file_path / "train.csv"
    test_file = file_path / "test.csv"
    return read_csv(train_file), read_csv(test_file)


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return the feature data of a Dataframe."""
    feature_columns = data.columns[:-1]
    return data[feature_columns]


def get_targets(data: pd.DataFrame) -> pd.Series:
    """Return the target data of a Dataframe."""
    target_column = data.columns[-1]
    return data[target_column]
