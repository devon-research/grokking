import os
import torch
from grokking.utils import parse_config

# Read in the configuration from the YAML file and override it with command-line arguments.
config = parse_config()

modular_base = config["modular_base"]
use_equals_symbol = config["use_equals_symbol"]

# Construct the inputs and outputs.
input_x = torch.arange(modular_base).repeat_interleave(modular_base)
input_y = input_x.view(modular_base, modular_base).transpose(0, 1).flatten()
inputs = torch.stack([input_x, input_y], dim=-1)
outputs = inputs.sum(dim=1) % modular_base
if use_equals_symbol:
    # Add a third token to represent the equals symbol to every input.
    # The equals symbol uses an input ID equal to the modular base.
    equals_symbols = torch.full((inputs.size(0), 1), modular_base)
    inputs = torch.cat([inputs, equals_symbols], dim=1)
dataset = torch.utils.data.TensorDataset(inputs, outputs)

# Write the data to a file.
this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")
os.makedirs(data_directory, exist_ok=True)
data_filename = f"{modular_base}-{use_equals_symbol}-dataset.pt"
torch.save(dataset, os.path.join(data_directory, data_filename))
