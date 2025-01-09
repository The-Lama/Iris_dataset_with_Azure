import argparse
import joblib
import logging
import pandas as pd
from pathlib import Path

import utils

logging.basicConfig(level=logging.DEBUG)


def predict(model_dir: str, test_data_dir: str, predictions_dir: str):
    """Predict the data with the given model."""
    logging.info("Predicting data..")
    logging.debug(f"Model directory: {model_dir}")
    logging.debug(f"Test data directory: {test_data_dir}")

    model_path = Path(model_dir) / "model.joblib"
    model = joblib.load(model_path)

    test_data_path = Path(test_data_dir) / "test.csv"
    test_data_df = pd.read_csv(test_data_path)

    test_data_features = utils.get_features(test_data_df)

    predictions = model.predict(test_data_features)

    logging.debug(f"predictions type: {type(predictions)}")
    logging.debug(f"predictions: {predictions}")

    output_dir = Path(predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.csv"
    logging.info(f"Saving predictions to {predictions_path}")
    predictions_df = pd.DataFrame(predictions, columns=["prediction"])
    predictions_df.to_csv(predictions_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory which contains the trained model."
        "The model has to be called model.joblib",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Path to the directory which contains the CSV data to be predicted.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path where the predictions will be saved.",
    )

    args = parser.parse_args()

    predict(args.model_dir, args.test_data_dir, args.predictions_dir)
