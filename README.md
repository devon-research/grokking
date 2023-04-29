# Predicting emergent and anomolous behavior in grokking

## Using the code
To set things up on a laptop using an Anaconda distribution, the following should work on a Windows machine. Navigate to the desired directory and run the following in the Anaconda Prompt:

```
conda create --name grokking pytorch cpuonly -c pytorch
conda activate grokking
conda install pyyaml tqdm
pip install accelerate wandb
git clone git@github.com:devonwp/grokking.git
pip install -e grokking
```

To set things up on a GPU cluster, use the following instead:
```
conda create --name grokking pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate grokking
conda install pyyaml tqdm
pip install accelerate wandb
git clone git@github.com:devonwp/grokking.git
pip install -e grokking
wandb offline
```

To generate the data and train a model, do:

```
cd grokking
python ./scripts/data.py
python ./scripts/train.py
```

If you are going to want to use the Julia scripts to run many experiments in one go, then also install the relevant Julia packages:
```
julia -e 'import Pkg; Pkg.add(["DrWatson", "ClusterManagers"])'
```

To run many experiments at once, do:
```
sbatch ./scripts/multitask.slurm
```