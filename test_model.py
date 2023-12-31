# create ModelTest
class ModelTest(nn.Module):
    def __init__(self):
        super(ModelTest, self).__init__()
        self.time_emb = TimeEmbedding(20, 64)
    def forward(self, x):
        emb = self.time_emb(x)
        return emb
model_pt = ModelTest()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=1e-4)

# generated sample dataset and using MSE and Adam optimizer
inputs = torch.rand(100, 20)
targets = torch.rand(100, 20, 64)
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 15  #15 loops for example
for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in data_loader:
        outputs = model_pt(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        num_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / num_batches
    print(f'Epoch [{epoch+1}/{num_epochs}], Average MSE: {avg_loss}')
