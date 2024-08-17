from grokking.utils import parse_config


def test_parse_config_boolean_parsing():
    # This aims to test the bug whereby the default conversion of strings to booleans (via argparse)
    # yields False for any non-empty string. This causes a silent failure.
    assert parse_config(args=["--save_checkpoints", "true"])["save_checkpoints"]
    assert parse_config(args=["--save_checkpoints", "True"])["save_checkpoints"]
    assert parse_config(args=["--save_checkpoints", "1"])["save_checkpoints"]
    assert not parse_config(args=["--save_checkpoints", "false"])["save_checkpoints"]
    assert not parse_config(args=["--save_checkpoints", "False"])["save_checkpoints"]
    assert not parse_config(args=["--save_checkpoints", "0"])["save_checkpoints"]
