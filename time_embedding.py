import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_embedding_size, output_dim):
        super(TimeEmbedding, self).__init__()
        self.output_dim = output_dim
        self.hidden_embedding_size = hidden_embedding_size

        # Initialize weights and biases
        self.emb_weights = nn.Parameter(torch.randn(hidden_embedding_size))
        self.emb_biases = nn.Parameter(torch.randn(hidden_embedding_size))
        self.emb_final = nn.Parameter(torch.randn(hidden_embedding_size, output_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = F.softmax(x * self.emb_weights + self.emb_biases, dim=-1)
        x = torch.einsum('bsv,vi->bsi', x, self.emb_final)
        return x

    def extra_repr(self):
        return 'time_dims={}, hidden_embedding_size={}'.format(self.output_dim, self.hidden_embedding_size)
