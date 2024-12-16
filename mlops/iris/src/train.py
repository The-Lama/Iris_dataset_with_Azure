import argparse
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

import utils

logging.basicConfig(level=logging.INFO)


def load_training_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training data from the directory containing train.csv and test.csv."""
    train_data, _ = utils.load_train_and_test_data(data_path)

    X_train = utils.get_features(train_data)
    y_train = utils.get_targets(train_data)

    return X_train, y_train


def train_model(transformed_data_path: str, model_output_path: str):
    """Train the model with the provided data."""
    logging.info("Starting model training.")
    logging.debug(f"Transformed data path: {transformed_data_path}")
    logging.debug(f"Model output path: {model_output_path}")

    mlflow.autolog()

    X_train, y_train = load_training_data(transformed_data_path)

    model_output_path = Path(model_output_path)
    model_output_path.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        training_score = model.score(X_train, y_train)
        logging.info(f"Model score: {training_score}")

        model_file_path = model_output_path / "model.joblib"
        joblib.dump(model, model_file_path)
        logging.info(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformed_data_path",
        type=str,
        required=True,
        help="Folder that contains both the train.csv and "
        "the test.csv for model training.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Folder that saves the model output.",
    )

    args = parser.parse_args()

    train_model(args.transformed_data_path, args.model_output_path)
