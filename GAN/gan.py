import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchinfo import summary


class Generator(nn.Module):
    def __init__(self, in_dim, dim=28):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            # 28*4=112*2=224
            self.block(dim * 8, dim * 4),
            self.block(dim * 4, dim * 2),
            self.block(dim * 2, dim),
            # 128 1 3 3
            nn.ConvTranspose2d(dim, 1, 5, padding=4),
            nn.Tanh()
        )

    @staticmethod
    def block(in_dim, out_dim):
        return nn.Sequential(
            # 128, 112, 2, 2
            # in_dim=128, out_dim=1,
            # 1) kernel_size=5, stride=1, padding=5,
            nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    # x(b, 1, 28 * 28)
    def forward(self, x):
        print(x.shape)
        out = self.layer1(x)
        print('out shape:', out.shape)
        out = out.reshape(out.size(0), -1, 4, 4)
        print(out.shape)
        out = self.layer3(out)
        print('layer3 shape', out.shape)

        return out


class Discriminator(nn.Module):
    @staticmethod
    def block(in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def __init__(self, in_dim=1, dim=28 * 28):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3),
            nn.LeakyReLU(0.2),
            self.block(dim, dim * 2),
            self.block(dim * 2, dim * 4),
            self.block(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(x.shape)
        out = self.layer(x)
        out = out.reshape(-1)
        return out


if __name__ == '__main__':
    batch_size = 16
    num_epochs = 10
    noise_dim = 128
    learning_rate = 0.0001
    writer = SummaryWriter()

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    G = Generator(in_dim=128)
    D = Discriminator()

    writer.add_graph(G, input_to_model=torch.randn(batch_size, noise_dim))
    # writer.add_graph(D)
    summary(G, (batch_size, 128))
    summary(D, (batch_size, 1, 28, 28))

    writer.add_graph(D, input_to_model=torch.randn(batch_size, 1, 28, 28))
    writer.close()
    # criterion = nn.BCELoss()
    # optim_g = torch.optim.Adam(G.parameters(), lr=learning_rate)
    # optim_d = torch.optim.Adam(D.parameters(), lr=learning_rate)

    # total_d_loss = 0
    # total_g_loss = 0
    # sample_img = torch.randn(batch_size, noise_dim)
    # for step, data in enumerate(train_loader):
    #     noise_sample = torch.randn(batch_size, noise_dim)
    #     fake_img: torch.Tensor = G(noise_sample)

    #     real_label = torch.ones(batch_size)
    #     fake_label = torch.zeros(batch_size)

    #     real_logit = D(data[0])
    #     fake_logit = D(fake_img.detach())

    #     print(f'real_logit: {real_logit.shape}, real_label: {real_label.shape}')
    #     real_loss = criterion(real_logit, real_label)
    #     fake_loss = criterion(fake_logit, fake_label)
    #     d_loss = real_loss + fake_loss
    #     total_d_loss += d_loss

    #     D.zero_grad()
    #     d_loss.backward()
    #     optim_d.step()

    #     # train generator
    #     g_fake_sample = torch.randn(batch_size, 1, 28 * 28)
    #     g_fake_img = G(g_fake_sample)
    #     g_fake_logit = D(g_fake_img)

    #     g_loss = criterion(g_fake_logit, fake_label)
    #     total_g_loss += g_loss

    #     G.zero_grad()
    #     g_loss.backward()
    #     optim_g.step()

    #     if (step+1) % 5 == 0:
    #         print(f'step: {step}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    #         G.eval()
    #         writer.add_scalar("loss", {"d_loss": total_d_loss / 10, "g_loss": total_g_loss / 10}, global_step=step)
    #         grid_img = torchvision.utils.make_grid(sample_img, nrow=8)
    #         writer.add_image(f"sample_{step}", sample_img, global_step=step)
    #         G.train()
