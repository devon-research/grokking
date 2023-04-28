import yaml
import os
import argparse
import wandb
import torch
from accelerate import Accelerator # easy GPUs
from tqdm import tqdm
from src.models import GromovMLP

# Train a model on the given dataset.
def train(model,
          ds_train,
          ds_valid,
          optimizer,
          loss_fn,
          batch_size,
          n_epochs,
          validate_every,
          logit_dtype):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size)
    acc = Accelerator()
    model, optimizer, dl_train, dl_valid = acc.prepare(model, optimizer, dl_train, dl_valid)
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for xb, yb in dl_train:
            # Cast the logits to float64 to avoid underflow.
            loss = loss_fn(model(xb).to(logit_dtype), yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss_train": loss.item()})
        if epoch % validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss_valid = sum(loss_fn(model(xb), yb) for xb, yb in dl_valid) / len(dl_valid)
            wandb.log({"loss_valid": loss_valid.item()})

# Read the configuration from the YAML file.
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Override the configuration with command-line arguments.
parser = argparse.ArgumentParser()
for key in config.keys():
    parser.add_argument("--" + key, default=config[key], type=type(config[key]))
config = vars(parser.parse_args())

# Load the datasets.
this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")
dataset_prefix = f"{config['modular_base']}-{config['train_fraction']}"
dataset_name = dataset_prefix + "-datasets.pt"
ds_train, ds_valid = torch.load(os.path.join(data_directory,
                                             dataset_name))

# Process the strings in the configuration.
if config["model"] == "GromovMLP":
    model = GromovMLP(config["modular_base"], config["hidden_dim"])
else:
    raise ValueError(f"Unknown model: {config['model']}")

if config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
else:
    raise ValueError(f"Unknown optimizer: {config['optimizer']}")

if config["loss_function"] == "MSE":
    loss_fn = torch.nn.MSELoss()
    raise NotImplementedError("Need to change the dataloader for this.")
elif config["loss_function"] == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    raise ValueError(f"Unknown loss function: {config['loss_function']}")

if config["logit_dtype"] == "float64":
    logit_dtype = torch.float64
elif config["logit_dtype"] == "float32":
    logit_dtype = torch.float32
else:
    raise ValueError(f"Unknown logit data type: {config['logit_dtype']}")

if config["full_batch"]:
    config["batch_size"] = len(ds_train)

# Set the random seed and initialize wandb.
torch.manual_seed(config["random_seed"])
wandb.init(project="grokking", config=config, tags=["initial"])

# Train the model.
train(model = model,
      ds_train = ds_train,
      ds_valid = ds_valid,
      optimizer = optimizer,
      loss_fn = loss_fn,
      batch_size = config["batch_size"],
      n_epochs = config["n_epochs"],
      validate_every = config["validate_every"],
      logit_dtype = logit_dtype)

wandb.finish()