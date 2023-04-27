import torch
from torch import nn
import torch.nn.functional as F

class GromovMLP(nn.Module):
    """A simple quadratic MLP based on the Gromov (2023) paper."""
    def __init__(self,
                 modular_base: int,
                 hidden_dim: int):
        super().__init__()
        self.modular_base = modular_base
        self.input_dim = 2 * modular_base # This is D in the paper.
        self.hidden_dim = hidden_dim # This is N in the paper.
        self.W1 = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, modular_base, bias=False)
    
    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Computes the forward pass of the network.

        Args:
            input_ids: LongTensor of shape (batch_size, 2) containing the
            two inputs token IDs for the binary modular task.
        
        Returns:
            FloatTensor of shape (batch_size, modular_base) containing the
            logits for the output token.
        """
        normalization = self.input_dim * self.hidden_dim
        one_hot_encodings = F.one_hot(input_ids, num_classes=self.modular_base)
        # The above will have shape (batch_size, 2, modular_base).
        one_hot_inputs = one_hot_encodings.view(-1, 2 * self.modular_base).float()
        return self.W2(torch.pow(self.W1(one_hot_inputs), 2)) / normalization