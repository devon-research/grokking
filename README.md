# Predicting emergent and anomolous behavior in grokking

## Using the code

To set things up on a laptop using an Anaconda distribution, the following should work on a Windows machine. Navigate to the desired directory and run the following in the Anaconda Prompt:
```
conda create --name grokking pytorch cpuonly -c pytorch
conda activate grokking
conda install pyyaml tqdm
pip install accelerate wandb
git clone git@github.com:devonwp/grokking.git
cd grokking
pip install -e .
```

To set things up on a GPU cluster, use the following instead:
```
module load anaconda3/2023.3
conda create --name grokking pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate grokking
conda install pyyaml tqdm
pip install accelerate wandb
git clone git@github.com:devonwp/grokking.git
cd grokking
pip install -e .
wandb offline
```

To generate the data and train a model, do:
```
python ./scripts/data.py
python ./scripts/train.py
```

Or, if you are on a GPU cluster, do:
```
python ./scripts/data.py
sbatch ./scripts/job.slurm
```

To run many experiments at once on a GPU cluster, do:
```
python ./scripts/experiments.py
```

One can then use `wandb sync` to sync results to W&B.