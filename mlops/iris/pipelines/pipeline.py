from pathlib import Path
import os
import logging
import argparse
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.ai.ml import load_component
from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline

logging.basicConfig(level=logging.INFO)

PIPELINE_COMPONENTS = {}


@pipeline
def iris_pipeline():
    """Define the iris pipeline."""
    train_component = PIPELINE_COMPONENTS["train"]
    train_component()


def construct_pipeline():
    """Construct the iris pipeline by loading the components."""
    logging.debug("loading pipeline components...")
    components_dir = Path("mlops/iris/components")
    train_model = load_component(source=components_dir / "train.yml")

    logging.debug("Constructing pipeline...")
    PIPELINE_COMPONENTS["train"] = train_model

    pipeline_job = iris_pipeline()

    return pipeline_job


def execute_pipeline(
    subscription_id, resource_group_name, workspace_name, pipeline_job
):
    """Execute the Iris pipeline."""
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        logging.debug("Submitting pipeline job...")
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="ML with Iris Dataset."
        )

        logging.info(f"{pipeline_job.name} has been submitted")

    except ClientAuthenticationError as e:
        logging.error("Invalid credentials.. try again")
        logging.error(f"Authentication failed: {e.message}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser("pipeline")
    parser.add_argument(
        "--subscription_id",
        type=str,
        default=os.getenv("SUBSCRIPTION_ID"),
        help="Azure subscription ID",
    )
    parser.add_argument(
        "--resource_group_name",
        type=str,
        default=os.getenv("RESOURCE_GROUP_NAME"),
        help="Azure Machine Learning resource group name",
    )
    parser.add_argument(
        "--workspace_name",
        type=str,
        default=os.getenv("WORKSPACE_NAME"),
        help="Azure Machine Learning workspace name",
    )

    args = parser.parse_args()

    pipeline_job = construct_pipeline()
    execute_pipeline(
        args.subscription_id,
        args.resource_group_name,
        args.workspace_name,
        pipeline_job,
    )
