import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    def __init__(self,input_shape, args):
        super(VanillaVAE, self).__init__()
        self.args = args
        self.latent_dim = args.latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=args.vae_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=args.vae_hidden_dim, out_features=self.latent_dim)
        )
        self.fc_mu = nn.Linear(self.latent_dim, self.args.n_actions)
        self.fc_var = nn.Linear(self.latent_dim, self.args.n_actions)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.args.n_actions, out_features=args.vae_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=args.vae_hidden_dim, out_features=args.vae_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=args.vae_hidden_dim, out_features=input_shape),
        )

    def encode(self, inputs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inputs: (Tensor) Input tensor to encoder 
        :return: (Tensor) 
        """
        result = self.encoder(inputs)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor)
        :return: (Tensor) 
        """
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return: (Tensor) 
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), inputs, mu, log_var]

    def loss_function(self,args,recons,inputs,mu,log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons_loss =F.mse_loss(recons, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + args.kld_weight * kld_loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        return loss

    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) 
        :return: (Tensor) 
        """
        return self.forward(x)[0]

class Encoder(nn.Module):
    def __init__(self, args,input_size):
        super(Encoder, self).__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.vae_hidden_dim = args.vae_hidden_dim
        self.linear1 = nn.Linear(input_size, self.vae_hidden_dim)
        self.linear2 = nn.Linear(self.vae_hidden_dim, self.latent_dim)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear1(x)) #-> bs,hidden_size
        x = self.linear2(x) #-> bs,latent_size
        return x
class Decoder(nn.Module):
    def __init__(self, args, output_size):
        super(Decoder, self).__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.vae_hidden_dim = args.vae_hidden_dim
        self.linear1 = torch.nn.Linear(self.latent_dim, self.vae_hidden_dim)
        self.linear2 = torch.nn.Linear(self.vae_hidden_dim, output_size)        
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        x = self.linear2(x) #->bs,output_size
        # x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        return x
class AE(nn.Module):
    def __init__(self, args,input_size, output_size):
        super(AE, self).__init__()
        self.args = args
        self.encoder = Encoder(args,input_size)
        self.decoder = Decoder(args,output_size)
    def forward(self, x): #x: bs,input_size
        feat = self.encoder(x) #feat: bs,latent_size
        re_x = self.decoder(feat) #re_x: bs, output_size
        return re_x
    def loss(self,x,re_x):
        return F.mse_loss(re_x,x)

def smooth_step(x,gamma=1):
    '''
    Implement for `smooth step activation function` proposed in paper 
    `The Tree Ensemble Layer: Differentiability meets Conditional Computation`
    S(t) =
    \begin{cases}
    0, & \mbox{if } t \le -\gamma/2 \\
    - \frac{2}{\gamma^3}t^3 + \frac{3}{2\gamma}t + \frac{1}{2}, & \mbox{if } -\gamma/2 \le t  \le \gamma/2  \\
    1, & \mbox{if } t \ge \gamma/2 
    \end{cases}
    Args:
        x -- inputs Tensor
        gamma -- hyperparameter,should be positive
    '''
    bound = gamma / 2
    x = torch.where(x <= -bound, 0,x)
    x = torch.where(x >= bound, 1,x)
    return SmoothIndicator(x)
    
def SmoothIndicator(x,gamma=1):
    '''
    Implement for 
    -\frac{2}{\gamma^3}t^3+\frac{3}{2\gamma}t+\frac{1}{2}
    '''
    x1=-2*x**3
    x2=1.5*x
    y1=gamma**3
    return x1/y1+x2/gamma+0.5