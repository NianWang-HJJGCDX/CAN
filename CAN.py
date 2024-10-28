import torch
import torch.nn as nn
from torch.autograd import Variable
from capsule_layer import CapsuleLayer

class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class AFE(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(AFE, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels, kernel_size=5, stride=1,bias=True)
        self.ReLU=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels, out_channels,  kernel_size=1, stride=1,bias=True)
        self.palayer=PA(in_channels)
    def forward(self, x):
        x = self.palayer(x)
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        return x

class CAN(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CAN, self).__init__()
        self.reconstructed_image_count = 0
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height
        self.AFE = AFE(in_channels=conv_inputs,out_channels=conv_outputs)
        self.Primarycaps = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)
        self.Classcaps = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)
        reconstruction_size = image_width * image_height * image_channels
        
        # define the three full connection layers for reconstruction 
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #[batch,70,5,5]
        x = self.AFE(x) #[batch,channel,3,3]
        x = self.Primarycaps(x)
        x = self.Classcaps(x)
        return x

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)
        if size_average:
            L_c = L_c.mean()
        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]
            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()
        return error