This repository can be used to reproduce some of the modular addition grokking results from papers like [Power et al. 2022](https://arxiv.org/abs/2201.02177), [Nanda et al. 2023](https://arxiv.org/abs/2301.05217), and [Gromov 2023](https://arxiv.org/abs/2301.02679).

The code furthermore facilitates the systematic manipulation of the training data fed into the modular addition networks. This manipulation allows for investigating the effects (on the behavior and associated internals) of adding imperfections or inserting anomalous rules alongside the modular addition task. The insertion of such anomalous rules then allows one to assess the ability to mechanistically detect anomalous / backdoored behavior in a relatively well-understood setting.

## Motivation
This repository was originally motivated by the mechanistic interpretability work of [Nanda et al. 2023](https://arxiv.org/abs/2301.05217) and the potential challenge of extending a similar mechanistic analysis to detecting and understanding important but non-dominant circuits that exist alongside the more typical behavior of a network.

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
python scripts/data.py
python scripts/train.py
```

Or, if you are on a GPU cluster, do:
```
python scripts/data.py
sbatch scripts/job.slurm
```

To run many experiments at once on a GPU cluster, do:
```
python scripts/experiments.py
```

One can then use `wandb sync` to sync results to W&B.