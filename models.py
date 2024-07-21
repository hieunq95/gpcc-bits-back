import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Bernoulli


class ConvoVAE(nn.Module):
    def __init__(self, in_dim=(40, 40, 40), out_dim=(40, 40, 40), h_dim=100, latent_dim=20):
        super(ConvoVAE, self).__init__()
        self.in_dim = in_dim  # [d, h, w]
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        if self.in_dim[0] not in [32, 64, 128]:
            raise AttributeError('Unsupported resolution')

        # down-sampling strategies:
        # resolution = 128:  128 -> 9/3+2(42), 5/3+1(14), 5/3(4) -> 4 (3 layers)
        # resolution = 64:   64 -> 9/3+1(20), 5/3(6), 2/1(4) -> 4 (3 layers)
        # resolution = 32:   32 -> 5/3(10), 4/2(4), 2/1(3) -> 3 (3 layers)
        if self.in_dim[0] == 128:
            ksp = [[9, 3, 2], [5, 3, 1], [5, 3, 0]]  # [kernel_size, stride, padding] per layer
            new_dim = 4  # new dimensional size after convo layers
        elif self.in_dim[0] == 64:
            ksp = [[9, 3, 1], [5, 3, 0], [2, 1, 0]]
            new_dim = 4
        elif self.in_dim[0] == 32:
            ksp = [[5, 3, 0], [4, 2, 0], [2, 1, 0]]
            new_dim = 3

        self.conv_encode = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=ksp[0][0], stride=ksp[0][1], padding=ksp[0][2]),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=ksp[1][0], stride=ksp[1][1], padding=ksp[1][2]),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=ksp[2][0], stride=ksp[2][1], padding=ksp[2][2]),
            nn.ReLU(),
        )

        # fully-connected layers
        self.new_dim = new_dim
        self.flat_dim = 16 * self.new_dim**3
        self.fc1 = nn.Linear(self.flat_dim, self.h_dim)
        self.fc21 = nn.Linear(self.h_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.h_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.h_dim)
        self.fc4 = nn.Linear(self.h_dim, self.flat_dim)

        self.conv_decode = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=ksp[2][0], stride=ksp[2][1], padding=ksp[2][2]),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=ksp[1][0], stride=ksp[1][1], padding=ksp[1][2]),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=ksp[0][0], stride=ksp[0][1], padding=ksp[0][2]),
            nn.Sigmoid()
        )

    def encode(self, x):
        conv_encoded_x = self.conv_encode(x)
        enc_x_flat = conv_encoded_x.view(-1, self.flat_dim)
        h1 = F.relu(self.fc1(enc_x_flat))
        mu = self.fc21(h1)
        var = torch.exp(self.fc22(h1))
        return mu, var

    def reparameterize(self, mu, log_var):
        if self.training:
            eps = torch.rand_like(log_var)
            return eps.mul(log_var).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        z_new_shape = (h4.size()[0], 16, self.new_dim, self.new_dim, self.new_dim)
        h4_reshape = torch.reshape(h4, z_new_shape)
        conv_decoded_x = self.conv_decode(h4_reshape)
        return conv_decoded_x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat

    def generate(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim)
        x_hat = self.decode(z)
        return x_hat

    def loss(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_probs = self.decode(z)
        flat_dim = self.in_dim[0] * self.in_dim[1] * self.in_dim[2]
        dist = Bernoulli(x_probs.view(-1, flat_dim))
        l = torch.sum(dist.log_prob(x.view(-1, flat_dim)), dim=1)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(Normal(mu, log_var).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * 1.4425 / flat_dim