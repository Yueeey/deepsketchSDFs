import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        # x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = self.FC_output(h)
        # x_hat = F.normalize(x_hat)
        return x_hat