import itertools
import subprocess
import shlex
import os
import yaml

# This has the same functionality as DrWatson's dict_list in Julia.
def product_dict_list(**kwargs):
    keys = kwargs.keys()
    for combination in itertools.product(*kwargs.values()):
        yield dict(zip(keys, combination))

def run_command(command_string):
    subprocess.run(shlex.split(command_string), check=True)

def option_string(options, sep=" "):
    return " ".join(f"--{key}{sep}{value}" for key, value in options.items())

train_option_lists = {
    "train_fraction": [0.1, 0.2, 0.3, 0.4],
    "random_seed": [23093, 9082, 1093],
}

# Read the configuration from the YAML file.
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Fill in the missing options from the YAML file.
for key in config:
    if key not in train_option_lists:
        train_option_lists[key] = [config[key]]

sbatch_options = {
    "job-name": "grokking",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 1,
    "mem": "3G",
    "gres": "gpu:1", 
    "time": "00:60:01"
    # If time <= 00:60:00, the job will be put in the gputtest QOS,
    # which is fine but it has a relatively small upper limit on the
    # number of jobs that can be submitted simultaneously.
}

this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")

data_script_params = ["modular_base", "use_equals_symbol"]
data_option_lists = {param: train_option_lists[param] for param in data_script_params}
print("Ensuring presence of data...")
for python_options in product_dict_list(**data_option_lists):
    file_name = f'{"-".join(str(x) for x in python_options.values())}-dataset.pt'
    if not os.path.isfile(os.path.join(data_directory, file_name)):
        print(f"Generating {file_name}...")
        python_option_string = option_string(python_options, sep=" ")
        run_command(f'python scripts/data.py {python_option_string}')
print("Done ensuring presence of data.")

print("Submitting training jobs...")
for python_options in product_dict_list(**train_option_lists):
    sbatch_option_string = option_string(sbatch_options, sep="=")
    python_option_string = option_string(python_options, sep=" ")
    run_command(f'sbatch {sbatch_option_string} '
                f'--wrap="python scripts/train.py {python_option_string}"')
print("Done submitting training jobs.")