import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[128, 64, 32]):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.1)
            ]
            prev_dim = h

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(current_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.1)
            ]
            current_dim = h

        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_latent(self, x):
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def vae_loss(recon_x, x, mu, logvar, beta=0.5):
    recon = F.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl
