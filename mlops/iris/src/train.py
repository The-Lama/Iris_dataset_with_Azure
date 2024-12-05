import argparse
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression

import utils

logging.basicConfig(level=logging.INFO)


def train_model(transformed_data_path: str, model_output_path: str):
    """Train the model with the provided data."""
    logging.info("Starting model training.")
    logging.debug(f"Transformed data path: {transformed_data_path}")
    logging.debug(f"Model output path: {model_output_path}")

    model_output_path = Path(model_output_path)
    model_output_path.mkdir(parents=True, exist_ok=True)

    train_data, test_data = utils.load_train_and_test_data(transformed_data_path)

    X_train = utils.get_features(train_data)
    y_train = utils.get_targets(train_data)

    X_test = utils.get_features(test_data)
    y_test = utils.get_targets(test_data)

    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model trained sucessfully.")
    logging.info(f"Model score: {model.score(X_test, y_test)}")


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
