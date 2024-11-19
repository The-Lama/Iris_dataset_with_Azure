import hashlib
import logging
from azure.ai.ml.entities import Environment


def calculate_hash(environment):
    """Calculate a hash value of an environment."""
    with open(environment.conda_file, "rb") as f:
        conda_data = f.read()

    logging.debug(f"conda_data: {conda_data}")
    logging.debug(f"base_image_data: {environment.image.encode()}")
    return hashlib.md5(conda_data + environment.image.encode())


def get_environment(client, base_image, conda_file, name, description):
    """Create and return an azure environment."""
    environment = Environment(
        image=base_image, conda_file=conda_file, name=name, description=description
    )

    return environment
