import yaml
import argparse
from pydantic import TypeAdapter
from typing import List


def parse_config(config_path: str = "config.yaml", args: List[str] = None) -> dict:
    """Parses the given YAML configuration file, and then overwrites entries in the
    configuration with command-line arguments.

    If args is None (the default), then the command-line arguments are taken from sys.argv.

    Command-line arguments can only be given for keys that are present in the YAML configuration.

    Note that no validation of the YAML configuration file is performed beyond what yaml.safe_load
    accomplishes. Furthermore, validation of the command-line arguments is done by Pydantic assuming
    that the types from the parsed YAML configuration are correct.

    Args:
        config_path: The path to the config file to be parsed. Defaults to "config.yaml".
        args: List of command-line arguments that is directly passed to argparse's parse_args.

    Returns:
        A dictionary containing the resulting parsed configuration.
    """
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
