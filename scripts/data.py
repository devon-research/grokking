import yaml
import os
import torch
import torch.nn.functional as F
# import einops

# Read the configuration from the YAML file.
with open("config.yaml") as f:
    config = yaml.safe_load(f)

P = config["modular_base"]
train_frac = config["train_fraction"]

# Construct the inputs and outputs.
input_x = torch.arange(P).repeat_interleave(P)
input_y = input_x.view(P, P).transpose(0, 1).flatten()
# An alternative way of writing the above is...
# input_x = einops.repeat(torch.arange(P), "a -> (a b)", b=P)
# input_y = einops.repeat(torch.arange(P), "b -> (a b)", a=P)
inputs = torch.stack([input_x, input_y], dim=-1)
outputs = inputs.sum(dim=1) % P

# Construct the torch dataset.
dataset = torch.utils.data.TensorDataset(inputs, outputs)

# Split the dataset into training and validation sets.
torch.manual_seed(config["random_seed"])
dataset_train, dataset_valid = torch.utils.data.random_split(
    dataset, [train_frac, 1 - train_frac])

# Write the data to a file.
file_location = os.path.dirname(os.path.abspath(__file__))
data_location = os.path.join(file_location, "..", "data")
os.makedirs(data_location, exist_ok=True)
torch.save([dataset_train, dataset_valid],
           os.path.join(data_location, f"{P}-{train_frac}-datasets.pt"))