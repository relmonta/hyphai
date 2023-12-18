import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Block module for U-Net.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    """
    Encoder module for U-Net.

    Args:
        chs (tuple, optional): Number of channels for each block.
    """

    def __init__(self, chs):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    """
    Decoder module for U-Net.

    Args:
        chs (tuple, optional): Number of channels for each block.
    """

    def __init__(self, chs):
        super(Decoder, self).__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    """
    Standard U-Net model from [O. Ronneberger et al.](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)

    Args:
        enc_chs (tuple, optional): Number of channels in the encoder. Defaults to (2, 64, 128, 256, 512).
        dec_chs (tuple, optional): Number of channels in the decoder. Defaults to (512, 256, 128, 64).
        num_class (int, optional): Number of output classes. Defaults to 2.
    """

    def __init__(self, enc_chs=(2, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64), num_class=2):
        super(UNet, self).__init__()
        self.encoder = Encoder(enc_chs).float()
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out
