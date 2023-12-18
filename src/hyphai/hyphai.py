from .unet_xception import UNet_Xception
import torch
import torch.nn as nn
import numpy as np

class HyPhAIModule(nn.Module):
    """
    Hybrid Physics-AI module for cloud cover nowcasting, the base class for all HyPhAI models.

    This module contains a U-Net, which is used to predict the velocity field, and a custom hard tanh function.
    Additionally, it contains a Gaussian smoothing kernel used to smooth the velocity field before solving the PDE.

    Attributes:
        leadtime (int): Prediction lead-time.
        n_classes (int): Number of classes.
        context_size (int): Context observation size.
        dx (float): Grid spacing in the x-direction (default value: 1./256).
        dy (float): Grid spacing in the y-direction (default value: 1./256).

    """

    def __init__(self, leadtime=6, n_classes=12, context_size=2):
        """
        Initialize the HyPhAIModule.

        Args:
            leadtime (int, optional): Prediction lead-time. Defaults to 6.
            n_classes (int, optional): Number of classes. Defaults to 12.
            context_size (int, optional): Context observation size. Defaults to 2.
        """
        super().__init__()
        self.leadtime = leadtime
        self.n_classes = n_classes
        self.context_size = context_size
        self.dx = 1. / 256
        self.dy = 1. / 256
        self.unet = UNet_Xception(
            enc_chs=(context_size, 32, 64, 128, 256),
            dec_chs=(256, 128, 64, 32),
            num_class=2,
            smoothing_kernel=HyPhAIModule.smoothing_kernel(33))

    def set_device(self, device: str):
        """Set GPU id to use

        Args:
            device (str): GPU id, e.g. 'cuda', 'cuda:0', 'cuda:1', etc. or 'cpu'
        """
        self.pde.set_device(device)
        self.unet.set_device(device)
        self.device = device

    @staticmethod
    def custom_tanh(x: torch.Tensor) -> torch.Tensor:
        r"""
        A custom hard tanh function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: :math:`\frac{tanh(4(x-0.6))+1}{2}`
        """
        return (torch.tanh(4 * (x - 0.6)) + 1) / 2

    @staticmethod
    def smoothing_kernel(kernel_size: int) -> np.ndarray:
        """
        A Gaussian smoothing kernel.

        Args:
            kernel_size (int): Kernel size.

        Returns:
            np.ndarray: Gaussian kernel.
        """
        if kernel_size < 2:
            return None
        t = np.linspace(-1, 1, kernel_size)
        t = t.reshape(kernel_size, 1) * np.ones((1, kernel_size))
        l = 0.3
        kernel = np.exp(-0.5 * (t**2 + t.T**2) / l**2)
        kernel /= kernel.sum()
        conv_kernel = np.ones((2, 1, kernel_size, kernel_size))
        conv_kernel[0, 0, :, :] = kernel
        conv_kernel[1, 0, :, :] /= conv_kernel[1, 0, :, :].sum()
        return conv_kernel.astype(np.float32)
