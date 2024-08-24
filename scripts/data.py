import os
import torch
from grokking.utils import parse_config
from grokking.data import generate_modular_addition_dataset


def main():
    config = parse_config()
    modular_base = config["modular_base"]
    use_equals_symbol = config["use_equals_symbol"]

    dataset = generate_modular_addition_dataset(modular_base, use_equals_symbol)

    this_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(this_directory, "..", "data")
    os.makedirs(data_directory, exist_ok=True)
    data_filename = f"{modular_base}-{use_equals_symbol}-dataset.pt"
    torch.save(dataset, os.path.join(data_directory, data_filename))


if __name__ == "__main__":
    main()
