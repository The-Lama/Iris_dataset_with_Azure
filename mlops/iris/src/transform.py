import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import utils

logging.basicConfig(level=logging.INFO)


def combine_features_with_target(
    feature_data: pd.DataFrame, target_series: pd.Series, all_columns: list
):
    """Combine the feature Dataframe with the target Series to a new Dataframe."""
    feature_columns = all_columns[:-1]
    target_column = all_columns[-1]
    combined_df = pd.DataFrame(feature_data, columns=feature_columns)
    combined_df[target_column] = target_series
    return combined_df


def transform_data(prepared_data_dir: str, transformed_data_dir: str):
    """Transform the data with a normalization."""
    logging.info("Starting data transformation")
    logging.debug(f"Prepared data path: {prepared_data_dir}")
    logging.debug(f"transformed data path: {transformed_data_dir}")

    transformed_data_dir = Path(transformed_data_dir)
    transformed_data_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = utils.load_train_and_test_data(prepared_data_dir)
    all_columns = train_data.columns

    train_features = utils.get_features(train_data)
    test_features = utils.get_features(test_data)

    train_labels = utils.get_targets(train_data)
    test_labels = utils.get_targets(test_data)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features_normalized = scaler.transform(train_features)
    test_features_normalized = scaler.transform(test_features)

    transformed_train_data = combine_features_with_target(
        train_features_normalized, train_labels, all_columns
    )
    transformed_test_data = combine_features_with_target(
        test_features_normalized, test_labels, all_columns
    )

    transformed_train_data_path = transformed_data_dir / "train.csv"
    transformed_test_data_path = transformed_data_dir / "test.csv"
    transformed_train_data.to_csv(transformed_train_data_path, index=False)
    transformed_test_data.to_csv(transformed_test_data_path, index=False)

    logging.info(f"Transformed train data saved to {transformed_train_data_path}")
    logging.info(f"transformed test data saved to {transformed_test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared_data_dir",
        type=str,
        required=True,
        help="Path to the directory that contains both train.csv and test.csv",
    )
    parser.add_argument(
        "--transformed_data_dir",
        type=str,
        required=True,
        help="Path to save the transformed data.",
    )

    args = parser.parse_args()

    transform_data(args.prepared_data_dir, args.transformed_data_dir)
