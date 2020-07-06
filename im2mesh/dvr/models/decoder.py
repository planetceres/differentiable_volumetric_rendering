
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from im2mesh.layers import ResnetBlockFC


class Decoder(nn.Module):
    ''' Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        return out



class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output#, coords        

class SirenDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=128,
                 hidden_size=512, n_blocks=5, out_dim=4):
        super().__init__()

        # We simply ignore the latent code
        self.siren = Siren(dim, hidden_size, n_blocks, out_dim, outermost_linear=True)


        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

    def forward(self, p, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        out = self.siren(p)

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        return out