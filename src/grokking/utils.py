import yaml
import argparse
from pydantic import TypeAdapter


def parse_config(config_path="config.yaml", args=None):
    # Read the configuration from the YAML file.
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Override the configuration with command-line arguments.
    parser = argparse.ArgumentParser()
    for key in config.keys():
        type_adapter_validator = TypeAdapter(type(config[key])).validate_strings
        parser.add_argument("--" + key, default=config[key], type=type_adapter_validator)
    config = vars(parser.parse_args(args))
    return config
