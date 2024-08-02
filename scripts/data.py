import os
import torch
from src.utils import parse_config
# import einops

# Reads in the configuration from the YAML file and overrides it with command-line arguments.
config = parse_config()

P = config["modular_base"]
use_equals = config["use_equals_symbol"]

# Construct the inputs and outputs.
input_x = torch.arange(P).repeat_interleave(P)
input_y = input_x.view(P, P).transpose(0, 1).flatten()
# An alternative way of writing the above is...
# input_x = einops.repeat(torch.arange(P), "a -> (a b)", b=P)
# input_y = einops.repeat(torch.arange(P), "b -> (a b)", a=P)
inputs = torch.stack([input_x, input_y], dim=-1)
outputs = inputs.sum(dim=1) % P
# Add a third token to represent the equals symbol to every input.
# The equals symbol uses the input ID P.
if use_equals:
    inputs = torch.cat([inputs, torch.full((inputs.size(0), 1), P)], dim=1)

# Construct the torch dataset.
dataset = torch.utils.data.TensorDataset(inputs, outputs)

# Write the data to a file.
this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")
os.makedirs(data_directory, exist_ok=True)
torch.save(dataset, os.path.join(data_directory, f"{P}-{use_equals}-dataset.pt"))
