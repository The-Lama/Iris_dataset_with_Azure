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
from azure.ai.ml.entities import Environment, PipelineJob
from mlops.common.environment_manager import get_environment
from mlops.common.environment_helpers import EnvironmentConfig
import sys
import yaml
from typing import Dict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""

    pass


def load_pipeline_components(components_dir: Path) -> Dict[str, callable]:
    """Load pipeline components from YAML definitions."""
    logging.info("Loading pipeline components...")
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
            try:
                components[name] = load_component(source=component_path)
                logging.info(f"Component '{name}' loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load component '{name}': {e}")
                raise PipelineError(f"Error to load component '{name}': {e}")
        else:
            raise PipelineError(
                f"Component definition '{name}.yml' was not found in {components_dir}"
            )

    return components


def define_iris_pipeline(components: Dict[str, callable], config: dict) -> pipeline:
    """Define the Iris pipeline by chaining components."""
    raw_data_path = config.get("raw_data_path")
    if not raw_data_path:
        raise PipelineError("Raw data path must be provided in the configuration.")

    raw_data_input = Input(type="uri_file", path=Path(raw_data_path))

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

        if config.get("register_model"):
            components["register"](
                model_name=config.get("model_name"),
                model_metadata_path=train.outputs.model_metadata_path,
                evaluation_report_path=evaluate.outputs.evaluation_report_path,
            )

    return iris_pipeline(raw_data_input)


def assign_environment_to_components(
    components: Dict[str, callable], environment: Environment
):
    """Assign the environment to components."""
    for name, component in components.items():
        component.environment = environment
        logging.info(f"Assigned environment to component '{name}'")


def construct_pipeline_job(environment: Environment, config: dict) -> PipelineJob:
    """Construct the iris pipeline job."""
    components_dir = Path(config["components_dir"])

    components = load_pipeline_components(components_dir)
    assign_environment_to_components(components, environment)

    logging.info("Constructing pipeline job...")
    pipeline_job = define_iris_pipeline(components, config)
    pipeline_job.compute = config["cluster_name"]

    return pipeline_job


def execute_pipeline(client, pipeline_job, config):
    """Execute the Iris pipeline."""
    logging.debug("Submitting pipeline job...")
    try:
        submitted_job = client.jobs.create_or_update(
            pipeline_job, experiment_name=config["experiment_name"]
        )
        logging.info(f"{submitted_job.name} has been submitted successfully.")
    except Exception as e:
        logging.error(f"Failed to submit pipeline job: {e}")
        raise PipelineError("Pipeline submission failed.")


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
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.getenv("PIPELINE_CONFIG_PATH"),
        help="Path to the pipeline configuration YAML",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return args


def load_pipeline_config(config_file: str) -> dict:
    """Load the pipeline configuration from a YAML file."""
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}'")
        raise PipelineError(f"Configuration file '{config_file}' is missing.")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        raise PipelineError("YAML parsing failed.")


def merge_config(args: argparse.Namespace, yaml_config: dict) -> dict:
    """Merge CLI arguments and YAML configuration, with CLI taking precedence."""
    config = yaml_config.copy()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def create_ml_client(config: dict) -> MLClient:
    """Create an Azure MLClient using the provided configuration."""
    try:
        return MLClient(
            DefaultAzureCredential(),
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group_name"],
            workspace_name=config["workspace_name"],
        )
    except ClientAuthenticationError as e:
        logging.error(f"Authentication failed. Please verify credentials. {e}")
        raise PipelineError("Authentication with Azure failed.")


if __name__ == "__main__":
    load_dotenv()
    args = load_configuration()
    yaml_config = load_pipeline_config(args.config_file)

    config = merge_config(args, yaml_config)

    try:
        client = create_ml_client(config)

        environment_config = EnvironmentConfig(
            base_image=config["env_base_image_name"],
            conda_file_path=config["conda_env_path"],
            name=config["environment_name"],
            description=config["environment_description"],
        )
        environment = get_environment(client, environment_config)

        pipeline_job = construct_pipeline_job(environment, config)
        execute_pipeline(client, pipeline_job, config)
    except PipelineError as pe:
        logging.error(f"Pipeline execution error: {pe}")
        sys.exit(1)
    except Exception as ex:
        logging.error(f"Unexpected error: {ex}")
        sys.exit(1)
