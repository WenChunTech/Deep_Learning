from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):

    def __init__(self, g_input_dim, g_output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)

        return torch.sigmoid(self.fc4(x))


class Discriminator(nn.Module):

    def __init__(self, d_input_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        print('x.shape', x.shape)
        x = F.dropout(x, 0.3)
        print('x.shape', x.shape)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)

        return torch.sigmoid(self.fc4(x))

writer = SummaryWriter('./logs')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
print('mnist dim: ', mnist_dim)
criterion = nn.BCELoss()

G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
D = Discriminator(d_input_dim=mnist_dim).to(device)
# optimizer
lr = 0.0002
g_optimizer = torch.optim.Adam(G.parameters(), lr = lr)
d_optimizer = torch.optim.Adam(D.parameters(), lr = lr)


def d_train(x):
    D.zero_grad()

    x_real, y_real = x.view(-1, mnist_dim).to(device), torch.ones(batch_size, 1).to(device)

    d_output = D(x_real)
    d_real_loss = criterion(d_output, y_real)
    d_real_score = d_output

    z = torch.randn(batch_size, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    d_output = D(x_fake)
    d_fake_loss = criterion(d_output, y_fake)
    d_fake_score = d_output

    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()


def g_train(x):
    G.zero_grad()
    z = torch.randn(batch_size, z_dim).to(device)
    y = torch.randn(batch_size, 1).to(device)

    g_output = G(z)
    d_output =  D(g_output)
    g_loss = criterion(d_output, y)

    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()

if __name__ == "__main__":
    summary(G)
    summary(D)