import torch
from torch import nn
import torch.nn.functional as F


class GromovMLP(nn.Module):
    """A simple quadratic MLP based on the Gromov (2023) paper."""

    def __init__(self, modular_base: int, hidden_dim: int):
        super().__init__()
        self.modular_base = modular_base
        self.input_dim = 2 * modular_base  # This is D in the paper.
        self.hidden_dim = hidden_dim  # This is N in the paper.
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


class NandaTransformer(nn.Module):
    """A simple 1-layer Transformer based on the Nanda et al. (2023) paper."""

    def __init__(
        self,
        modular_base: int = 113,
        embed_dim: int = 128,
        intermediate_mlp_dim: int = 512,
        max_sequence_len: int = 3,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        vocab_size = modular_base + 1
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=max_sequence_len, embedding_dim=embed_dim
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            bias=True,  # Want bias=False but PyTorch has a bug.
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, intermediate_mlp_dim),
            nn.ReLU(),
            nn.Linear(intermediate_mlp_dim, embed_dim),
        )
        self.token_unembeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Computes the forward pass of the network.

        Args:
            input_ids: LongTensor of shape (batch_size, max_sequence_len)
            containing the input token IDs for the binary modular task.

        Returns:
            FloatTensor of shape (batch_size, modular_base) containing the
            logits for the output token.
        """
        hidden_states = self.token_embeddings(input_ids) + self.positional_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        hidden_states = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            need_weights=False,
            is_causal=True,
        )[0]
        hidden_states = self.feedforward(hidden_states)
        # Only return the output logits for the last token.
        # The following has shape (batch_size, max_sequence_len, modular_base + 1)
        # before the slice.
        return F.linear(hidden_states, self.token_unembeddings.weight)[:, -1]

    def get_mlp_activations(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Computes the final-token activations of the MLP layer for a given input.

        Args:
            input_ids: LongTensor of shape (batch_size, max_sequence_len)
            containing the input token IDs for the binary modular task.

        Returns:
            FloatTensor of shape (batch_size, intermediate_mlp_dim)
            containing the activations of the MLP layer for the given input.
        """
        hidden_states = self.token_embeddings(input_ids) + self.positional_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        hidden_states = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            need_weights=False,
            is_causal=True,
        )[0]
        hidden_states = self.feedforward[0](
            hidden_states
        )  # Apply the first layer of the MLP.
        hidden_states = self.feedforward[1](hidden_states)  # Apply the nonlinearity.
        return hidden_states[:, -1, :]
