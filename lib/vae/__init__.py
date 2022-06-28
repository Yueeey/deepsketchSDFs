import torch
import torch.nn as nn

import numpy as np

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.mu_bn = nn.BatchNorm1d(256)
        self.mu_bn.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self, reset=False):
        gamma = 0.5
        if not reset:
            self.mu_bn.weight.fill_(gamma)
        else:
            print('reset bn!')
            self.mu_bn.weight.fill_(gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

        
    def reparameterization(self, mean, var):
        DEVICE = mean.device
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var * epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        mean = self.mu_bn(mean)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var, z

    def random_para(self, mean, var):
        DEVICE = mean.device
        mean_ = torch.zeros(mean.shape).data.normal_(0, 1).to(DEVICE) 
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean_ + var * epsilon 
        z = z.to(DEVICE)
        return z

    def random_vis(self, x):
        mean, log_var = self.Encoder(x)
        z = self.random_para(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

    def reparameterization_multi(self, mean, var):
        DEVICE = mean.device
        z = []
        z.append(mean)
        for i in range(5):
            epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
            z_ = mean + var * epsilon                          # reparameterization trick
            z.append(z_)
        return z
        
                
    def para_multi(self, x):
        mean, log_var = self.Encoder(x)
        mean = self.mu_bn(mean)
        z = self.reparameterization_multi(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = []
        for z_ in z:
            x_hat.append(self.Decoder(z_))
        return x_hat, mean, log_var, z

    # def random_para_multi(self, mean, var):
    #     DEVICE = mean.device
    #     mean_ = torch.zeros(mean.shape).data.normal_(0, 1).to(DEVICE)
    #     z = []
    #     z.append(mean_)
    #     for i in range(5): 
    #         epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
    #         z_ = mean_ + var * epsilon 
    #         z.append(z_)
        
    #     return z

    # def random_vis_multi(self, x):
    #     mean, log_var = self.Encoder(x)
    #     z = self.random_para_multi(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
    #     x_hat = []
    #     for z_ in z:
    #         x_hat.append(self.Decoder(z_))
    #     return x_hat, mean, log_var

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 