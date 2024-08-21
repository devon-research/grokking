This repository can be used to reproduce some of the modular addition grokking results from papers like [Power et al. 2022](https://arxiv.org/abs/2201.02177), [Nanda et al. 2023](https://arxiv.org/abs/2301.05217), and [Gromov 2023](https://arxiv.org/abs/2301.02679).

The code furthermore facilitates the systematic manipulation of the training data fed into the modular addition networks. This manipulation allows for investigating the effects (on the behavior and associated internals) of adding imperfections or inserting anomalous rules alongside the modular addition task. The insertion of such anomalous rules then allows one to assess the ability to mechanistically detect anomalous / backdoored behavior in a relatively well-understood setting.

## Motivation
This repository was originally motivated by the mechanistic interpretability work of [Nanda et al. 2023](https://arxiv.org/abs/2301.05217) and the potential challenge of extending a similar mechanistic analysis to detecting and understanding important but non-dominant circuits that exist alongside the more typical behavior of a network.

## Using the code

To set things up using [Rye](https://github.com/astral-sh/rye), simply run the following:
```
git clone https://github.com/devon-research/grokking.git
cd grokking
rye sync
```

Furthermore, specify a configuration. By default, the given `config.yaml` will be used.

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

To run many experiments at once on a GPU cluster, first specify a configuration with one or more list-valued entries. Then run:
```
python scripts/experiments.py
```
(This will run one experiment for every entry in the Cartesian product of all lists in the configuration.)

One can then use `wandb sync` to sync results to W&B.