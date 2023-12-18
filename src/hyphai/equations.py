import torch
import torch.nn as nn
import torch.nn.functional as F
    
class PDEUpwind(torch.nn.Module):
    """Second optimized version of PDE class"""
    def __init__(self, dx: tuple = (1. / 256, 1. / 256), source: str='none'):
        super().__init__()
        self.dx = dx[0]
        self.dy = dx[1]
        self.source = source
        self.pad = {'pad': (0, 0, 1, 1, 1, 1), 'mode': "replicate"}
        # kernels using weights from finite differences formulas
        self.kernel_Dc_x1_plus = torch.nn.Parameter(torch.Tensor([[0., 0., 0.],
                                               [0., -1., 0.],
                                               [0., 1., 0.]]
                                              ).unsqueeze(0).unsqueeze(0) / (2 * self.dx))
        self.kernel_Dc_x1_minus = torch.nn.Parameter(torch.Tensor([[0., -1., 0.],
                                                [0., 1., 0.],
                                                [0., 0., 0.]]
                                               ).unsqueeze(0).unsqueeze(0) / (2 * self.dx))
        # (3,3) -> (1,1,3,3,1) : (in_channels, out_channels, kX, kY, kZ)
        self.kernel_Dc_y1_plus = torch.nn.Parameter(torch.Tensor([[0., 0., 0.],
                                               [0., -1., 1.],
                                               [0., 0., 0.]]
                                              ).unsqueeze(0).unsqueeze(0) / (2 * self.dy))
        self.kernel_Dc_y1_minus = torch.nn.Parameter(torch.Tensor([[0., 0., 0.],
                                                [-1., 1., 0.],
                                                [0., 0., 0.]]
                                               ).unsqueeze(0).unsqueeze(0) / (2 * self.dy))
        self.kernels = [self.kernel_Dc_x1_plus, self.kernel_Dc_x1_minus, self.kernel_Dc_y1_plus, self.kernel_Dc_y1_minus]
        self.conv_layers = [
            torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0, groups=12, bias=False, stride=1),
            torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0, groups=12, bias=False, stride=1),
            torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0, groups=12, bias=False, stride=1),
            torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0, groups=12, bias=False, stride=1)
        ]
        self.set_device('cpu') # default device is cpu    

    def set_device(self, device):
        self.device = device
        self.repeated_kernels = []
        for kernel in self.kernels:
            self.repeated_kernels.append(kernel.repeat(12, 1, 1, 1).to(device))
        for i, conv_layer in enumerate(self.conv_layers):
            conv_layer.to(device).weight = torch.nn.Parameter(self.repeated_kernels[i])
            conv_layer.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward method

        Args:
            inputs (torch.Tensor): Input tensors
        
        Returns:
            torch.Tensor: Trend tensor
        """
        p, u_plus, u_minus, v_plus,  v_minus = inputs[:5]
        padded_p = F.pad(p, self.pad['pad'], mode=self.pad['mode'])
        # Reshape padded_p from (batch_size, 1, 256, 256, n_classes) to
        # (batch_size, n_classes, 256, 256), so that we can feed it to the 2d conv layers
        padded_p = torch.swapaxes(padded_p, 1, -1).squeeze(-1)
        derivatives = []
        for conv_layer in self.conv_layers:
            derivative = conv_layer(padded_p)
            derivative = torch.swapaxes(derivative.unsqueeze(-1), 1, -1)
            derivatives.append(derivative)
        
        # u_minus * Dc_x_o1_plus - u_plus * Dc_x_o1_minus
        mul_u = derivatives[0] * u_minus
        mul_Dc_x_o1_minus_u_plus = derivatives[1] * u_plus
        mul_u -= mul_Dc_x_o1_minus_u_plus
        # v_minus * Dc_y_o1_plus - v_plus * Dc_y_o1_minus
        mul_v = derivatives[2] * v_minus
        mul_Dc_y_o1_minus_v_plus = derivatives[3] * v_plus
        mul_v -= mul_Dc_y_o1_minus_v_plus
        # sum
        trend_p = mul_u + mul_v

        # Source term (HyPhAI-2)
        if self.source.lower() == 'hyphai-2':
            trend_p = trend_p + inputs[-1]

        # Source term (HyPhAI-3)
        if self.source.lower() == 'hyphai-3':
            lambda_rate = inputs[-1]
            # = Sum_i P(C_(t+1)=j|C_t=i) * P(C_t=i)
            # Changing the shape of p from (batch_size, 1, 256, 256, n_classes) to
            # (batch_size, 1, 256, 256, n_classes, 1), so that we can multiply it with
            # s of shape (batch_size, 1, 256, 256, n_classes, n_classes)
            source_term = torch.matmul(lambda_rate, p.unsqueeze(-1))
            trend_p = trend_p + source_term.squeeze(-1)

        # Source term (HyPhAI-4)
        if self.source.lower() == 'hyphai-4':
            # weightings are of shape (batch_size, n_matrices,256,256)
            # transitions are of shape (n_matrices, 1, 1, 1, n_classes, n_classes)
            weightings, transitions = inputs[-1]
            # = Sum_i P(C_(t+1)=j|C_t=i) * P(C_t=i)
            # = Sum_j W_j * T_j * P
            # reshape of p from (batch_size, 1, 256, 256, n_classes) to (batch_size, 1, 256, 256, n_classes, 1)
            source_term = torch.matmul(torch.zeros_like(transitions[:1]).to(self.device), p.unsqueeze(-1)).squeeze(-1) * weightings[:,-1:,...]
            for i in range(transitions.shape[1]):
                source_term += torch.matmul(transitions[i:i+1], p.unsqueeze(-1)).squeeze(-1) * weightings[:,i:i+1,...]
            trend_p = trend_p + source_term            
        return trend_p


class PDECentralDiff(nn.Module):
    r"""A class to represent the Advection equation

    Attributes:
        dx (tuple): Grid spacing in x and y directions
    """

    def __init__(self, dx: tuple = (1. / 256, 1. / 256), source: str = 'none'):
        super().__init__()
        self.dx = dx[0]
        self.dy = dx[1]
        # Padding to add to tensors before applying derivatives
        self.pad = {'pad': (0, 0, 1, 1, 1, 1), 'mode': "replicate"}
        # kernels using weights from finite differences formulas
        self.kernel_Dc_x1 = torch.Tensor([[0., -1., 0.],
                                          [0., 0., 0.],
                                          [0., 1., 0.]]
                                         ).unsqueeze(0).unsqueeze(0).unsqueeze(-1) / (2 * self.dx)
        # (3,3) -> (1,1,3,3,1) : (in_channels, out_channels, kX, kY, kZ)
        self.kernel_Dc_y1 = torch.Tensor([[0., 0., 0.],
                                          [-1., 0., 1.],
                                          [0., 0., 0.]]
                                         ).unsqueeze(0).unsqueeze(0).unsqueeze(-1) / (2 * self.dy)
        self.bias = torch.Tensor([0.])
        self.source = source
    def set_device(self, device: str or int):
        """Set GPU ID

        Args:
            device (str or int): GPU id, e.g. 'cuda:0', 'cuda:1', etc. or 'cpu'
        """
        self.device = device
        self.bias = self.bias.to(device)
        self.kernel_Dc_x1 = self.kernel_Dc_x1.to(device)
        self.kernel_Dc_y1 = self.kernel_Dc_y1.to(device)
    def forward(self, inputs: torch.Tensor) -> tuple:
        r"""Forward method

        Args:
            inputs (torch.Tensor): Inputs equation variables

        Returns:
            tuple: .. math:: \frac{df}{dt}
        """
        p, minus_u, minus_v = inputs[0], inputs[1], inputs[2]

        # Derivatives
        padded_p = F.pad(p, self.pad['pad'], self.pad['mode'])
        Dc_x_o1 = F.conv3d(padded_p, self.kernel_Dc_x1,
                           bias=self.bias)  # dP(x,y)/dx
        Dc_y_o1 = F.conv3d(padded_p, self.kernel_Dc_y1,
                           bias=self.bias)  # dP(x,y)/dy

        # Advection
        mul_0 = minus_u * Dc_x_o1
        mul_1 = minus_v * Dc_y_o1
        trend_p = mul_0 + mul_1

        # Source term (HyPhAI-2)
        if self.source.lower() == 'hyphai-2':
            trend_p = trend_p + inputs[-1]
            
        # Source term (HyPhAI-3)
        if self.source.lower() == 'hyphai-3':
            lambda_rate = inputs[-1]
            # = Sum_i P(C_(t+1)=j|C_t=i) * P(C_t=i)
            # Changing the shape of p from (batch_size, 1, 256, 256, n_classes) to
            # (batch_size, 1, 256, 256, n_classes, 1), so that we can multiply it with
            # s of shape (batch_size, 1, 256, 256, n_classes, n_classes)
            source_term = torch.matmul(lambda_rate, p.unsqueeze(-1))
            trend_p = trend_p + source_term.squeeze(-1)

        # Source term (HyPhAI-4)
        if self.source.lower() == 'hyphai-4':
            # weightings are of shape (batch_size, n_matrices,256,256)
            # transitions are of shape (n_matrices, 1, 1, 1, n_classes, n_classes)
            weightings, transitions = inputs[-1]
            # = Sum_i P(C_(t+1)=j|C_t=i) * P(C_t=i)
            # = Sum_j W_j * T_j * P
            # reshape of p from (batch_size, 1, 256, 256, n_classes) to (batch_size, 1, 256, 256, n_classes, 1)
            source_term = torch.matmul(torch.zeros_like(transitions[:1]).to(self.device), p.unsqueeze(-1)).squeeze(-1) * weightings[:,-1:,...]
            for i in range(transitions.shape[1]):
                source_term += torch.matmul(transitions[i:i+1], p.unsqueeze(-1)).squeeze(-1) * weightings[:,i:i+1,...]
            trend_p = trend_p + source_term      
        return trend_p