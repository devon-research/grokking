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


def test_parse_config_with_lists():
    # This aims to test whether parse_config successfully parses configs with
    # list-valued entries.
    assert parse_config("tests/config-with-lists.yaml") == {
        "train_fraction": [0.1, 0.2, 0.3, 0.4],
        "random_seed": [23093, 9082, 1093],
    }
