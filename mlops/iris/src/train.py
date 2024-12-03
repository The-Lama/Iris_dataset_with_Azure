import argparse
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from mlops.iris.src.utils import read_csv

logging.basicConfig(level=logging.INFO)


def train_model(transformed_data_path: str, model_output_path: str):
    """Train the model with the provided data."""
    logging.info("Starting model training.")
    logging.debug(f"Transformed data path: {transformed_data_path}")
    logging.debug(f"Model output path: {model_output_path}")

    transformed_data_path = Path(transformed_data_path)
    model_output_path = Path(model_output_path)
    model_output_path.mkdir(parents=True, exist_ok=True)

    train_file = transformed_data_path / "train.csv"
    test_file = transformed_data_path / "test.csv"

    train_data = read_csv(train_file)
    test_data = read_csv(test_file)

    feature_columns = train_data.columns[:-1]
    target_column = train_data.columns[-1]

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

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
