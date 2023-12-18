import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeparableConv2d(nn.Module):
    """
    Separable 2D Convolution module implementation.

    This module applies depthwise and pointwise convolutions sequentially.

    Args:
        in_ch (int): Input channel.
        out_ch (int): Output channel.
        kernel_size (int): Kernel size for depthwise convolution.
        padding (int, optional): Padding size. Defaults to None.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding=None):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, groups=in_ch, padding=kernel_size // 2 if not padding else padding)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EncBlock(nn.Module):
    """
    Block module used in the U-Net's encoder.

    Args:
        in_ch (int): Input channel.
        out_ch (int): Output channel.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(EncBlock, self).__init__()
        self.sepconv = SeparableConv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(x)
        x = self.sepconv(x)
        x = self.batchnorm(x)
        return x


class Encoder(nn.Module):
    """
    U-Net's encoder.

    Args:
        chs (tuple, optional): Channels for each block.
    """

    def __init__(self, chs):
        super(Encoder, self).__init__()
        self.enc_blocks1 = nn.ModuleList([EncBlock(chs[i], chs[i + 1]) for i in range(1, len(chs) - 1)])
        self.enc_blocks2 = nn.ModuleList([EncBlock(chs[i], chs[i]) for i in range(2, len(chs))])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res_convs = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 1, stride=2) for i in range(1, len(chs) - 1)])
        self.conv1 = nn.Conv2d(chs[0], chs[1], 3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(chs[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        previous = x
        for i in range(len(self.enc_blocks1)):
            x = self.enc_blocks1[i](x)
            x = self.enc_blocks2[i](x)
            x = self.maxpool(x)
            residual = self.res_convs[i](previous)
            x = torch.add(x, residual)
            previous = x
        return x


class DecBlock(nn.Module):
    """
    Block module used in the U-Net's decoder.

    Args:
        in_ch (int): Input channel.
        out_ch (int): Output channel.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(DecBlock, self).__init__()
        self.sepconv = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(x)
        x = self.sepconv(x)
        x = self.batchnorm(x)
        return x


class Decoder(nn.Module):
    """
    U-Net's decoder.

    Args:
        chs (tuple, optional): Channels for each block.
    """

    def __init__(self, chs):
        super(Decoder, self).__init__()
        self.dec_blocks1 = nn.ModuleList([DecBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.dec_blocks2 = nn.ModuleList([DecBlock(chs[i], chs[i]) for i in range(1, len(chs))])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.res_convs = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 1) for i in range(len(chs) - 1)])

    def forward(self, x):
        previous = x
        for i in range(len(self.dec_blocks1)):
            x = self.dec_blocks1[i](x)
            x = self.dec_blocks2[i](x)
            x = self.upsample(x)
            residual = self.res_convs[i](self.upsample(previous))
            x = torch.add(x, residual)
            previous = x
        return x


class UNet_Xception(nn.Module):
    r"""
    U-Net Xception-style from [Tamvakis et al., 2022](https://www.sciencedirect.com/science/article/pii/S2405896322027525).

    Args:
        enc_chs (tuple, optional): Encoder's channels. Defaults to (4, 32, 64, 128, 256).
        dec_chs (tuple, optional): Decoder's channels. Defaults to (256, 128, 64, 32).
        num_class (int, optional): Output channel size. Defaults to 2.
        source_ch (int, optional): Source output's channel size. Defaults to 0.
        smoothing_kernel (np.ndarray, optional): Gaussian kernel. Defaults to None.
    """

    def __init__(self, enc_chs=(4, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_class=2, source_ch=0, smoothing_kernel=None):
        super(UNet_Xception, self).__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head_fil = [dec_chs[-1], 32, 16, 8, 4, num_class]
        self.head = nn.ModuleList([nn.Conv2d(self.head_fil[i], self.head_fil[i + 1], 3, padding=1) for i in range(len(self.head_fil) - 1)])
        if source_ch > 0:
            self.head_source = nn.Conv2d(self.head_fil[0], source_ch, 3, padding=1)
        if smoothing_kernel is None:
            self.kernel = None
            self.bias = None
        else:
            self.kernel = torch.from_numpy(smoothing_kernel)
            self.bias = torch.Tensor([0., 0.])
        self.source_ch = source_ch

    def set_device(self, device: str or int):
        """
        Set the GPU id for the model.

        Args:
            device (str or int): GPU id, e.g. 'cuda:0', 'cuda:1', etc. or 'cpu'
        """
        if self.kernel is not None:
            self.kernel = self.kernel.to(device)
            self.bias = self.bias.to(device)
        else:
            pass

    def forward(self, x) -> torch.Tensor or tuple:
        """
        Forward method for the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor or tuple: Output tensor/s.
        """
        # Encoding and decoding
        enc_x = self.encoder(x)
        out = self.decoder(enc_x)
        # Add second head for the source term used in the PDE
        # source_ch is the number of channels of the source term output
        if self.source_ch:
            s = self.head_source(out)
        # Add head for output
        for i in range(len(self.head_fil) - 1):
            out = self.head[i](out)
        # Smoothing kernel for the first output (velocity field)
        if self.kernel is not None:
            u = F.conv2d(out, self.kernel, bias=self.bias, groups=2, padding=self.kernel.shape[-1] // 2)
        else:
            u = out

        if self.source_ch:
            return u, s
        else:
            return u
