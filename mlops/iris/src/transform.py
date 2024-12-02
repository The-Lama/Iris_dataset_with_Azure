import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


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


def transform_data(prepared_data_path: str, transformed_data_path: str):
    """Transform the data with a normalization."""
    logging.info("Starting data transformation")
    logging.debug(f"Prepared data path: {prepared_data_path}")
    logging.debug(f"transformed data path: {transformed_data_path}")

    prepared_data_path = Path(prepared_data_path)
    transformed_data_path = Path(transformed_data_path)
    transformed_data_path.mkdir(parents=True, exist_ok=True)

    train_file = prepared_data_path / "train.csv"
    test_file = prepared_data_path / "test.csv"

    train_data = read_csv(train_file)
    test_data = read_csv(test_file)

    feature_columns = train_data.columns[:-1]
    target_column = train_data.columns[-1]

    train_features = train_data[feature_columns]
    train_target = train_data[target_column]

    test_features = test_data[feature_columns]
    test_target = test_data[target_column]

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features_normalized = scaler.transform(train_features)
    test_features_normalized = scaler.transform(test_features)

    transformed_train_data = pd.DataFrame(
        train_features_normalized, columns=train_features.columns
    )
    transformed_train_data[target_column] = train_target

    transformed_test_data = pd.DataFrame(
        test_features_normalized, columns=test_features.columns
    )
    transformed_test_data[target_column] = test_target

    transformed_train_data_path = transformed_data_path / "train.csv"
    transformed_test_data_path = transformed_data_path / "test.csv"
    transformed_train_data.to_csv(transformed_train_data_path, index=False)
    transformed_test_data.to_csv(transformed_test_data_path, index=False)

    logging.info(f"Transformed train data saved to {transformed_train_data_path}")
    logging.info(f"transformed test data saved to {transformed_test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared_data_path",
        type=str,
        required=True,
        help="Path to the directory that contains both train.csv and test.csv",
    )
    parser.add_argument(
        "--transformed_data_path",
        type=str,
        required=True,
        help="Path to save the transformed data.",
    )

    args = parser.parse_args()

    transform_data(args.prepared_data_path, args.transformed_data_path)
