import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class ConvUnit(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,  # fixme constant
                               kernel_size=3,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)
    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    #
    # digit: in_units=8 in_channels=1152 num_units=10 unit_size=16 use_routing=Ture
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing
        self.weight_conv = nn.Conv2d(unit_size * self.num_units, 8, 1, 1, bias=True)
        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels,out_channels=unit_size)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)] #[128,32,1,1]
        u_all = u[0]
        # [The proposed SWM, here we adopt a gated convolution operation to automatically give the weights
        # for capsule convolutions. Therefore, it benefits to generate better primary activity vectors.]
        for i in range(self.num_units-1):
            u_all = torch.cat((u_all,u[i+1]),dim=1)
        weights = self.weight_conv(u_all)
        u = [u[i] * weights[:, [i], :, :] for i in range(self.num_units)]
        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1) #[128,8,32,1,1]
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1) #[128,8,32]
        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0) # [128,8,32]
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2) # [128,32,8]
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4) # [128,32,16,8,1]
        # (batch, features, num_units, unit_size, in_units)
        W = torch.cat([self.W] * batch_size, dim=0) # [128,32,16,16,8]
        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x) # [128,32,16,16,1]
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).cuda() #[1,32,16,1]
        # Iterative for dynamic routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij) #[1,32,16,1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4) #[128,32,16,1,1]
            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) #[128,1,16,16,1]
            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j) #[128,1,16,16,1]
            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1) #[128,32,16,16,1]
            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True) #[1,32,16,1]
            # Update b_ij (routing)
            b_ij = b_ij + u_vj1
        return v_j.squeeze(1) #[128,16,16,1]
