import torch
import einops


def generate_modular_addition_dataset(
    modular_base: int, use_equals_symbol: bool
) -> torch.utils.data.TensorDataset:
    """Generates a dataset for the binary modular addition task.

    The task is to add two numbers modulo P, where P is the modular base. The dataset
    consists of all possible pairs of numbers from 0 to P-1 as the inputs, and the
    corresponding sum modulo P as the outputs.

    Note that training and test splits are not generated here.

    Args:
        modular_base: The base of the modulo operation.
        use_equals: Whether to include an equals symbol among the input tokens.

    Returns:
        A TensorDataset containing inputs of dimension (P^2, 3) [if use_equals_symbol]
        or (P^2, 2) [if not use_equals_symbol] and outputs of dimension (P^2).
    """
    input_x = einops.repeat(torch.arange(modular_base), "a -> (a b)", b=modular_base)
    input_y = einops.repeat(torch.arange(modular_base), "b -> (a b)", a=modular_base)
    # Let L be the number of examples in the dataset, and let K=2 be the number of operands.
    inputs = einops.rearrange([input_x, input_y], "K L -> L K")  # type: ignore
    outputs = einops.reduce(inputs, "L K -> L", reduction="sum") % modular_base
    if use_equals_symbol:
        # Add a third token to represent the equals symbol to every input.
        # The equals symbol uses an input ID equal to the modular base.
        equals_symbols = torch.full((inputs.size(0), 1), modular_base)
        inputs = torch.cat([inputs, equals_symbols], dim=1)
    return torch.utils.data.TensorDataset(inputs, outputs)
