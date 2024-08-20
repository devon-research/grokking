import itertools
import subprocess
import shlex
import os
from collections.abc import Iterable, Iterator
from grokking.utils import parse_config


def keyed_product(**kwargs: Iterable) -> Iterator[dict]:
    """
    Given an unpacked dictionary of lists (or other iterables), returns an iterator
    yielding dictionaries representing a single element or "combination" from the
    Cartesian product of the lists.

    The name is inspired by itertools.product, but is a "keyed" version of it in that
    the arguments are keyword args and the outputs are dictionaries instead of tuples.

    Note that this has the same functionality as (the Julia package) DrWatson's
    [dict_list](https://juliadynamics.github.io/DrWatson.jl/v1.15/run&list/#DrWatson.dict_list).
    The only difference is that this function accepts an unpacked dictionary instead of
    a dictionary and returns an iterator instead of a list.

    Args:
        **kwargs: An unpacked dictionary of lists (or other iterables).

    Returns:
        An iterator yielding dictionaries each with the same keys as the input and with
        values corresponding to one element of the Cartesian product of the input lists.
    """
    keys = kwargs.keys()
    for combination in itertools.product(*kwargs.values()):
        yield dict(zip(keys, combination))


def run_command(command_string: str) -> None:
    subprocess.run(shlex.split(command_string), check=True)


def option_string(options: dict, sep=" ") -> str:
    return " ".join(f"--{key}{sep}{value}" for key, value in options.items())


option_lists: dict[str, list] = {
    "train_fraction": [0.1, 0.2, 0.3, 0.4],
    "random_seed": [23093, 9082, 1093],
}

# Read in the configuration from the YAML file and override it with command-line arguments.
config = parse_config()

# Populate the dictionary of option lists with options from the config that are not yet
# present. Note that the option_lists dictionary overrides the config dictionary.
for key in config:
    if key not in option_lists:
        option_lists[key] = [config[key]]

sbatch_options = {
    "job-name": "grokking",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 1,
    "mem": "3G",
    "gres": "gpu:1",
    "time": "00:60:01",
    # If time <= 00:60:00, the job will be put in the gputtest QOS,
    # which is fine but it has a relatively small upper limit on the
    # number of jobs that can be submitted simultaneously.
}

this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")

print("Ensuring presence of data...")
data_script_params = ["modular_base", "use_equals_symbol"]
data_option_lists = {param: option_lists[param] for param in data_script_params}
for python_options in keyed_product(**data_option_lists):
    file_name = f'{"-".join(str(x) for x in python_options.values())}-dataset.pt'
    if not os.path.isfile(os.path.join(data_directory, file_name)):
        print(f"Generating {file_name}...")
        python_option_string = option_string(python_options, sep=" ")
        run_command(f"python scripts/data.py {python_option_string}")
print("Done ensuring presence of data.")

print("Submitting training jobs...")
for python_options in keyed_product(**option_lists):
    sbatch_option_string = option_string(sbatch_options, sep="=")
    python_option_string = option_string(python_options, sep=" ")
    run_command(
        f"sbatch {sbatch_option_string} "
        f'--wrap="python scripts/train.py {python_option_string}"'
    )
print("Done submitting training jobs.")
