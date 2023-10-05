import torch
from typing import Optional


class ModelConfig:
    def __init__(self,
                 num_layers: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 vocab_size: Optional[int] = None,
                 embedding_dim: Optional[int] = None,
                 criterion: Optional[torch.nn.Module] = None,
                 dropout: float = 0.5,
                 use_semantic: Optional[bool] = False,
                 ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.criterion = criterion
        self.dropout = dropout
        self.use_semantic = use_semantic


class ModelOutput:
    def __init__(self, logits, probabilities, loss=None, embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.loss = loss
        self.embeddings = embeddings
