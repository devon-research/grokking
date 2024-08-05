import yaml
import argparse


def _str2bool(s):
    s_l = s.lower()
    if s_l in ["true", "1"]:
        return True
    elif s_l in ["false", "0"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")


def parse_config(config_path="config.yaml"):
    # Read the configuration from the YAML file.
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Override the configuration with command-line arguments.
    parser = argparse.ArgumentParser()
    for key in config.keys():
        value_converter = type(config[key])
        if value_converter == bool:
            value_converter = _str2bool
        parser.add_argument("--" + key, default=config[key], type=value_converter)
    config = vars(parser.parse_args())
    return config
