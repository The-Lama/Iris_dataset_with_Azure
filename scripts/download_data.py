import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
import logging

CSV_FILENAME = "iris.csv"
DESCRIPTION_FILENAME = "data_information.md"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_relative_path(path: Path):
    try:
        relative_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = path
    return relative_path

def ensure_directory_exists(directory: Path):
    """Ensure that the directory exists, creating it if nescessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        relative_dir = get_relative_path(directory)
        logging.info(f"Directory created or already exists: {relative_dir}")
    except OSError as e:
        logging.error(f"Failed to save content to {relative_dir}: {e}")

def save_file(path: Path, save_func, content):
    """
    Generic function to save content to a file if it doesn't already exist.
    
    Parameters:
    - path: The path where the file should be saved.
    - save_func: A function that handles the actual saving logic for the content.
    - content: The content to be saved, passed to save_func.
    """
    relative_path = get_relative_path(path)
    if path.exists():
        logging.warning(f"{relative_path} already exists, skipping writing file")
        return

    try:
        save_func(path, content)
        logging.info(f"Saved file content to {relative_path}")
    except OSError as e:
        logging.error(f"failed to create directory {relative_path}: {e}")

def save_csv(path: Path, data: pd.DataFrame):
    """Save a dataframe to a csv file"""
    save_func = lambda p, df: df.to_csv(p, index=False)
    save_file(path, save_func, data)

def save_dataset_description(path: Path, description: str):
    """Save the dataset description to a file"""
    save_func = lambda p, desc: p.write_text(desc)
    save_file(path, save_func, description)

def parse_arguments():
    """parse CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "mlops" / "iris" / "data")
    parser.add_argument("--docs-dir", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--csv-filename", type=str, default=CSV_FILENAME)
    parser.add_argument("--description-filename", type=str, default=DESCRIPTION_FILENAME)

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_arguments()

    data_dir = args.data_dir
    docs_dir = args.docs_dir

    ensure_directory_exists(data_dir)
    ensure_directory_exists(docs_dir)

    iris = load_iris(as_frame=True)
    iris_df = iris.frame

    csv_path = data_dir / args.csv_filename
    description_path = docs_dir / args.description_filename

    save_csv(csv_path, iris_df)
    save_dataset_description(description_path, iris.DESCR)