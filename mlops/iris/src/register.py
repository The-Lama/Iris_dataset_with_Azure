import argparse
import json
import mlflow
import logging
from pathlib import Path
from json import JSONDecodeError

logging.basicConfig(level=logging.DEBUG)


def read_json_file(filepath: str) -> dict:
    """Read a JSON file and return its contents."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found at {filepath}")

    with open(filepath, "r") as f:
        try:
            return json.load(f)
        except JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file {filepath}")


def register_model(
    model_name: str,
    model_metadata_path: str,
    evaluation_report_path: str,
):
    """Register a model to Azure ML."""
    logging.debug(f"Registering model: {model_name}")
    logging.debug(f"Model metadata path: {model_metadata_path}")
    logging.debug(f"Evaluation report path: {evaluation_report_path}")

    model_metadata = read_json_file(model_metadata_path)
    evaluation_report = read_json_file(evaluation_report_path)

    model_uri = model_metadata.get("model_uri")
    if not model_uri:
        raise KeyError(f"'model_uri' was not found in model metadata: {model_metadata}")

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    logging.info(f"Model registered successfully: {model_version}")

    client = mlflow.MlflowClient()

    for metric, score in evaluation_report.items():
        client.set_model_version_tag(
            name=model_version.name,
            version=model_version.version,
            key=metric,
            value=score,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the registered model."
    )
    parser.add_argument(
        "--model_metadata_path",
        type=str,
        required=True,
        help="Path to the model metadata JSON file.",
    )
    parser.add_argument(
        "--evaluation_report_path",
        type=str,
        required=True,
        help="Path to the evaluation report. (JSON)",
    )
    args = parser.parse_args()

    register_model(
        args.model_name,
        args.model_metadata_path,
        args.evaluation_report_path,
    )
