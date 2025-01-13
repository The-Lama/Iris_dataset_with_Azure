import argparse
import json
import logging
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.DEBUG)


def evaluate(ground_truth_dir: str, predictions_dir: str, evaluation_report_path: str):
    """Evaluate the model predictions with the test data."""
    logging.debug("evaluating..")
    ground_truth_file = Path(ground_truth_dir) / "test.csv"
    predictions_file = Path(predictions_dir) / "predictions.csv"
    y_true = pd.read_csv(ground_truth_file)["target"].values
    y_pred = pd.read_csv(predictions_file)["prediction"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    confusion = confusion_matrix(y_true, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    evaluation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion.tolist(),
    }

    with open(evaluation_report_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Directory that holds the file test.csv with the ground truth.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory that holds the predictions.csv file with the predictions.",
    )
    parser.add_argument(
        "--evaluation_report_path",
        type=str,
        required=True,
        help="File that will be used to save the evaluations.",
    )

    args = parser.parse_args()

    evaluate(args.ground_truth_dir, args.predictions_dir, args.evaluation_report_path)
