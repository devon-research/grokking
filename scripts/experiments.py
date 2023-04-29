import itertools
import subprocess
import shlex

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
    # Relevant for the data and model...
    "modular_base": [113],
    # Relevant for the data...
    "train_fraction": [0.1, 0.2, 0.3, 0.4],
    # Relevant for the model...
    "model": ["GromovMLP"],
    "hidden_dim": [32, 128, 512],
    # Relevant for the optimization...
    "n_epochs": [30000],
    "random_seed": [23093, 9082, 1093],
    "optimizer": ["Adam", "SGD"],
    "loss_function": ["MSE", "CrossEntropy"],
    "learning_rate": [0.001, 0.0001],
    "batch_size": [128, 1024, -1],
}

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

print("Submitting training jobs...")
for python_options in product_dict_list(**train_option_lists):
    sbatch_option_string = option_string(sbatch_options, sep="=")
    python_option_string = option_string(python_options, sep=" ")
    run_command(f'sbatch {sbatch_option_string} '
                f'--wrap="python scripts/train.py {python_option_string}"')
print("Done submitting training jobs.")