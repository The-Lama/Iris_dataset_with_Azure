import argparse
import joblib
import logging
import pandas as pd
from pathlib import Path

import utils

logging.basicConfig(level=logging.DEBUG)


def predict(model_path: str, test_data_path: str, predictions_path: str):
    """Predict the data with the given model."""
    logging.info("Predicting data..")
    logging.debug(f"Model path: {model_path}")
    logging.debug(f"Test data path: {test_data_path}")

    model_path = Path(model_path) / "model.joblib"
    model = joblib.load(model_path)

    test_data_path = Path(test_data_path) / "test.csv"
    test_data_df = pd.read_csv(test_data_path)

    test_data_features = utils.get_features(test_data_df)

    predictions = model.predict(test_data_features)

    logging.debug(f"predictions type: {type(predictions)}")
    logging.debug(f"predictions: {predictions}")

    output_dir = Path(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.csv"
    logging.info(f"Saving predictions to {predictions_file}")
    predictions_df = pd.DataFrame(predictions, columns=["prediction"])
    predictions_df.to_csv(predictions_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory which contains the trained model."
        "The model has to be called model.joblib",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the directory which contains the CSV data to be predicted.",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path where the predictions will be saved.",
    )

    args = parser.parse_args()

    predict(args.model_path, args.test_data_path, args.predictions_path)
