import torch
import torch.nn as nn
from typing import Tuple


class GenerativeModel(nn.Module):
    """LSTM-based generative model for molecular SMILES generation."""
    
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        embedding_dim: int, 
        n_layers: int,
        dropout: float = 0.2
    ):
        super(GenerativeModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            n_layers, 
            dropout=dropout,
            batch_first=False
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self, 
        input_tensor: torch.Tensor, 
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model."""
        batch_size = input_tensor.size(0)
        embedded = self.embedding(input_tensor)
        output, hidden = self.lstm(embedded.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for LSTM."""
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        )
        return hidden