import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import utils

logging.basicConfig(level=logging.INFO)


def combine_features_with_target(
    features: pd.DataFrame, target: pd.Series, columns: list
):
    """Combine the feature Dataframe with the target Series to a new Dataframe."""
    feature_columns = columns[:-1]
    target_column = columns[-1]
    combined_df = pd.DataFrame(features, columns=feature_columns)
    combined_df[target_column] = target
    return combined_df


def transform_data(prepared_data_path: str, transformed_data_path: str):
    """Transform the data with a normalization."""
    logging.info("Starting data transformation")
    logging.debug(f"Prepared data path: {prepared_data_path}")
    logging.debug(f"transformed data path: {transformed_data_path}")

    transformed_data_path = Path(transformed_data_path)
    transformed_data_path.mkdir(parents=True, exist_ok=True)

    train_data, test_data = utils.load_train_and_test_data(prepared_data_path)
    columns = train_data.columns

    train_features = utils.get_features(train_data)
    test_features = utils.get_features(test_data)

    train_target = utils.get_targets(train_data)
    test_target = utils.get_targets(test_data)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features_normalized = scaler.transform(train_features)
    test_features_normalized = scaler.transform(test_features)

    transformed_train_data = combine_features_with_target(
        train_features_normalized, train_target, columns
    )
    transformed_test_data = combine_features_with_target(
        test_features_normalized, test_target, columns
    )

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
