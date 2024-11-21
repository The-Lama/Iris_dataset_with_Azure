import hashlib
import json
from dataclasses import dataclass
from azure.ai.ml.entities import Environment


@dataclass
class EnvironmentConfig:
    """
    Configuration for an Azure ML environment.

    Attributes:
        base_image (str): The Docker base image to use for the environment.
        conda_file_path (str): Path to the Conda YAML file defining dependencies.
        name (str): The name of the Azure ML environment.
        description (str): A brief description of the environment's purpose.
    """

    base_image: str
    conda_file_path: str
    name: str
    description: str


def calculate_hash(environment: Environment) -> str:
    """Calculate a hash value of an environment."""
    conda_contents = environment.conda_file
    conda_contents_serialized = json.dumps(conda_contents, sort_keys=True)

    base_image = environment.image.strip()

    data = f"{conda_contents_serialized}:{base_image}"
    return hashlib.sha256(data.encode()).hexdigest()
