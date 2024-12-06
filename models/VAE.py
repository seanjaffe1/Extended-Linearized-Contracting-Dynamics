
from torch import nn
import torch

class VAE_traininer(nn.Module):
    def __init__(self, transform, input_dim=8, latent_dim=3):
        super().__init__()

        self.transform = transform

        self.var_net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, latent_dim),
            nn.Tanh(),
        )

        # initalize the var_net
        for m in self.var_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        mu, log_det = self.transform(x)
        logvar = self.var_net(x)

        z = self.reparameterize(mu, logvar) # takes exponential function (log var -> var)
        x_hat, _  = self.transform.inverse(z)
        
        return x_hat, mu, logvar