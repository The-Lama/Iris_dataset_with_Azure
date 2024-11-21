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
from mlops.common.environment_management import get_environment


logging.basicConfig(level=logging.DEBUG)

PIPELINE_COMPONENTS = {}


@pipeline
def iris_pipeline():
    """Define the iris pipeline."""
    train_component = PIPELINE_COMPONENTS["train"]
    train_component()


def construct_pipeline(cluster_name, environment):
    """Construct the iris pipeline by loading the components."""
    logging.debug("loading pipeline components...")
    components_dir = Path("mlops/iris/components")
    train_component = load_component(source=components_dir / "train.yml")
    train_component.environment = environment

    logging.debug("Constructing pipeline...")
    PIPELINE_COMPONENTS["train"] = train_component

    pipeline_job = iris_pipeline()
    pipeline_job.compute = cluster_name

    return pipeline_job


def execute_pipeline(client, pipeline_job):
    """Execute the Iris pipeline."""
    logging.debug("Submitting pipeline job...")
    pipeline_job = client.jobs.create_or_update(
        pipeline_job, experiment_name="ML_with_Iris_Dataset"
    )

    logging.info(f"{pipeline_job.name} has been submitted")


def load_configuration():
    """Load configuration from environment variables or command-line arguments."""
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
    parser.add_argument(
        "--cluster_name",
        type=str,
        default=os.getenv("CLUSTER_NAME"),
        help="Name of azure compute cluster",
    )
    parser.add_argument(
        "--env_base_image_name",
        type=str,
        default=os.getenv("ENV_BASE_IMAGE_NAME"),
        help="Name of base image for system managed environment",
    )

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = load_configuration()

    try:
        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group_name,
            workspace_name=args.workspace_name,
        )
    except ClientAuthenticationError as e:
        logging.error("Invalid credentials.. try again")
        logging.error(f"Authentication failed: {e.message}")

    environment = get_environment(
        client=client,
        base_image=args.env_base_image_name,
        conda_file_path="mlops/iris/environments/ml-environment.yml",
        name="iris-ml",
        description="environment to run the ml code",
    )

    pipeline_job = construct_pipeline(args.cluster_name, environment)
    execute_pipeline(client, pipeline_job)
