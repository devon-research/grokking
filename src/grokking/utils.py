import yaml
import argparse
from typing import List


def parse_config(config_path: str = "config.yaml", args: List[str] = None) -> dict:
    """Parses the given YAML configuration file, and then overwrites entries in the
    configuration with command-line arguments.

    If args is None (the default), then the command-line arguments are taken from sys.argv.

    Command-line arguments can only be given for keys that are present in the YAML configuration.

    Note that no validation of the YAML configuration file is performed beyond what yaml.safe_load
    accomplishes. Furthermore, conversion of the command-line strings is done using yaml.safe_load
    for consistency.

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
        # The argument type=yaml.safe_load converts each argument using yaml.safe_load,
        # thus converting the string arguments in a way consistent with the config file.
        parser.add_argument("--" + key, default=config[key], type=yaml.safe_load)
    config = vars(parser.parse_args(args))
    return config
