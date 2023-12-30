import torch
import torch.nn as nn
from time_embedding import TimeEmbedding
class ModelTest(nn.Module):
    def __init__(self):
        super(ModelTest, self).__init__()
        self.time_emb = TimeEmbedding(25, 40)

    def forward(self, x):
        emb = self.time_emb(x)
        return emb
model = ModelTest()
# Testing the model
input_tensor = torch.rand(7, 25)
results = model(input_tensor)
results
assert results.shape == torch.Size([7, 25, 40])
