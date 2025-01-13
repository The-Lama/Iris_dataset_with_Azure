import argparse
import json
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

import utils

logging.basicConfig(level=logging.INFO)


def load_training_data(data_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training data from the directory containing train.csv and test.csv."""
    train_data, _ = utils.load_train_and_test_data(data_dir)

    X_train = utils.get_features(train_data)
    y_train = utils.get_targets(train_data)

    return X_train, y_train


def train_model(transformed_data_dir: str, model_dir: str, model_metadata_path: str):
    """Train the model with the provided data."""
    logging.info("Starting model training.")
    logging.debug(f"Transformed data dir: {transformed_data_dir}")
    logging.debug(f"Model output dir: {model_dir}")

    mlflow.autolog()

    X_train, y_train = load_training_data(transformed_data_dir)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        model = LogisticRegression()
        model.fit(X_train, y_train)

        training_score = model.score(X_train, y_train)
        logging.info(f"Model score: {training_score}")

        model_file_path = model_dir / "model.joblib"
        joblib.dump(model, model_file_path)
        logging.info(f"Model saved to {model_file_path}")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        model_metadata = {
            "run_id": run_id,
            "model_uri": model_uri,
        }
        with open(model_metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformed_data_dir",
        type=str,
        required=True,
        help="Directory that contains both the train.csv and "
        "the test.csv for model training.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory that saves the model output.",
    )
    parser.add_argument(
        "--model_metadata_path",
        type=str,
        required=True,
        help="Path of the model metadata.",
    )

    args = parser.parse_args()

    train_model(
        args.transformed_data_dir,
        args.model_dir,
        args.model_metadata_path,
    )
