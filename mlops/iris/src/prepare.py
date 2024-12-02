import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)


def prepare_data(
    raw_data_path: str, prepared_data_path: str, test_size: float, random_state: int
):
    """Prepare the iris train data with train and test split."""
    logging.info("Starting data preparation...")
    logging.debug(f"Raw data path: {raw_data_path}")
    logging.debug(f"Prepared data path: {prepared_data_path}")

    prepared_data_path = Path(prepared_data_path)
    prepared_data_path.mkdir(parents=True, exist_ok=True)

    raw_data_path = Path(raw_data_path)

    if not raw_data_path.is_file():
        raise FileNotFoundError(f"file not found at {raw_data_path}")

    raw_data = pd.read_csv(raw_data_path)

    if raw_data.empty:
        raise ValueError("The dataset is empty. Please provide a valid dataset.")

    train_data, test_data = train_test_split(
        raw_data, test_size=test_size, random_state=random_state
    )
    logging.info(
        f"Split data into {len(train_data)} train samples and "
        f"{len(test_data)} test samples"
    )

    train_data_path = prepared_data_path / "train.csv"
    test_data_path = prepared_data_path / "test.csv"
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    logging.info(f"Train data saved to {train_data_path}")
    logging.info(f"Test data saved to {test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="path to the raw data file.",
    )
    parser.add_argument(
        "--prepared_data_path",
        type=str,
        required=True,
        help="path to save the prepared data.",
    )
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--random_state", type=int)

    args = parser.parse_args()

    prepare_data(
        raw_data_path=args.raw_data_path,
        prepared_data_path=args.prepared_data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
