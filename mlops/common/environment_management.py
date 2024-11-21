import hashlib
import json
import logging
from typing import Optional
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError


def calculate_hash(environment: Environment) -> str:
    """Calculate a hash value of an environment."""
    conda_contents = environment.conda_file
    conda_contents_serialized = json.dumps(conda_contents, sort_keys=True)

    base_image = environment.image.strip()

    data = f"{conda_contents_serialized}:{base_image}"
    return hashlib.sha256(data.encode()).hexdigest()


def get_existing_environment(client: MLClient, name: str) -> Optional[Environment]:
    """Retrieve the latest version of an existing environment by name."""
    try:
        return client.environments.get(name=name, label="latest")
    except ResourceNotFoundError:
        logging.warning(f"Environment with name '{name}' was not found")
        return None


def get_environment(
    client: MLClient, base_image, conda_file_path, name, description
) -> Environment:
    """
    Create and return an Azure ML environment.

    If the environment already exists and has a different hash,
    a new version is created.
    """
    new_environment = Environment(
        image=base_image, conda_file=conda_file_path, name=name, description=description
    )
    new_env_hash = calculate_hash(new_environment)
    logging.debug(f"New environment hash: {new_env_hash}")

    existing_environment = get_existing_environment(client, name)
    if existing_environment:
        existing_env_hash = calculate_hash(existing_environment)
        logging.debug(f"existing environment hash: {existing_env_hash}")

        if new_env_hash == existing_env_hash:
            logging.info(f"Environment '{name}' is up to date. No update needed.")
            return existing_environment

    logging.info(f"Creating or updating environment '{name}'.")
    return client.environments.create_or_update(new_environment)
