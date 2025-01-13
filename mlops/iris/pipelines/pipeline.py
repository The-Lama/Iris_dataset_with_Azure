from pathlib import Path
import os
import logging
import argparse
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.ai.ml import load_component
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from mlops.common.environment_manager import get_environment
from mlops.common.environment_helpers import EnvironmentConfig


logging.basicConfig(level=logging.DEBUG)


def load_pipeline_components(components_dir: Path) -> dict:
    """Load pipeline components from YAML definitions."""
    logging.debug("Loading pipeline components...")
    components = {}

    component_names = [
        "prepare",
        "transform",
        "train",
        "predict",
        "evaluate",
        "register",
    ]

    for name in component_names:
        component_path = components_dir / f"{name}.yml"
        if component_path.exists():
            components[name] = load_component(source=component_path)
            logging.debug(f"Component '{name}' loaded sucessfully")
        else:
            logging.error(
                f"Component definition '{name}.yml' was not found in {components_dir}"
            )
    return components


def define_iris_pipeline(components: dict) -> pipeline:
    """Define the Iris pipeline by chaining components."""
    data_dir = Path("mlops/iris/data")
    raw_data_input = Input(type="uri_file", path=data_dir / "iris.csv")

    @pipeline
    def iris_pipeline(raw_data_path):
        """Full pipeline on the iris dataset."""
        prepare = components["prepare"](raw_data_path=raw_data_path)
        transform = components["transform"](
            prepared_data_dir=prepare.outputs.prepared_data_dir
        )
        train = components["train"](
            transformed_data_dir=transform.outputs.transformed_data_dir
        )
        predict = components["predict"](
            transformed_data_dir=transform.outputs.transformed_data_dir,
            model_dir=train.outputs.model_dir,
        )
        evaluate = components["evaluate"](
            ground_truth_dir=transform.outputs.transformed_data_dir,
            predictions_dir=predict.outputs.predictions_dir,
        )
        components["register"](
            model_name="Logistic_Regression_on_Iris_Dataset",
            model_metadata_path=train.outputs.model_metadata_path,
            evaluation_report_path=evaluate.outputs.evaluation_report_path,
        )

    return iris_pipeline(raw_data_input)


def construct_pipeline_job(cluster_name, environment):
    """Construct the iris pipeline job."""
    components_dir = Path("mlops/iris/components")

    components = load_pipeline_components(components_dir)
    for name, component in components.items():
        component.environment = environment
        logging.info(f"Assigned environment to component '{name}'")

    logging.debug("Constructing pipeline job...")
    pipeline_job = define_iris_pipeline(components)
    pipeline_job.compute = cluster_name

    return pipeline_job


def execute_pipeline(client, pipeline_job):
    """Execute the Iris pipeline."""
    logging.debug("Submitting pipeline job...")
    try:
        pipeline_job = client.jobs.create_or_update(
            pipeline_job, experiment_name="ML_with_Iris_Dataset"
        )
        logging.info(f"{pipeline_job.name} has been submitted successfully.")
    except Exception as e:
        logging.error(f"Failed to submit pipeline job: {e}")


def load_configuration() -> argparse.Namespace:
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


def create_ml_client(args: argparse.Namespace) -> MLClient:
    """Create an Azure MLClient using the provided configuration."""
    try:
        return MLClient(
            DefaultAzureCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group_name,
            workspace_name=args.workspace_name,
        )
    except ClientAuthenticationError as e:
        logging.error("Authentication failed. Please verify credentials.")
        raise e


if __name__ == "__main__":
    load_dotenv()
    args = load_configuration()

    client = create_ml_client(args)

    environment_config = EnvironmentConfig(
        base_image=args.env_base_image_name,
        conda_file_path="mlops/iris/environments/ml-environment.yml",
        name="iris-ml",
        description="environment to run the ml code",
    )
    environment = get_environment(client, environment_config)

    pipeline_job = construct_pipeline_job(args.cluster_name, environment)
    execute_pipeline(client, pipeline_job)
