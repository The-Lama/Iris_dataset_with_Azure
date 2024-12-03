import pandas as pd
from pathlib import Path


def read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and ensures it is not empty."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError(
            f"The file {file_path} is empty. Please provide a valid dataset."
        )

    return data
