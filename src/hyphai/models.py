from .unet import UNet as UNet
from .equations import *
from .hyphai import HyPhAIModule
from .schemes import RungeKutta4
from .unet_xception import UNet_Xception
import torch.nn.functional as F
import torch.nn as nn
import torch

class HyPhAI1(HyPhAIModule):
    r"""Hybrid Physics-AI model for cloud cover nowcasting, This model has two parts:

    1. The PDE, here the advection equation, The scheme used to solve the PDE is the 4th order Runge-Kutta.
    2. The U-Net, which is used to predict the velocity field.
        The U-Net takes as input a tensor of size `(batch_size, context_size, H, W)`,
        where `context_size` is the number of observations in the context.
        The U-Net's output is a tensor `U` of size `(batch_size, 2, H, W)`.
        The first channel of `U` is the west-east velocity u and the second channel is the south-north velocity v.

    """

    def __init__(self, leadtime=8, n_classes=12, context_size=4, ndt=10):
        """__init__ method

        Args:
            leadtime (int, optional): Prediction lead-time. Defaults to 8.
            n_classes (int, optional): Number of classes. Defaults to 12.
            context_size (int, optional): Context observations size. Defaults to 4.
        """
        super().__init__(
            leadtime=leadtime,
            n_classes=n_classes,
            context_size=context_size)
        # The PDE is solved on a grid of size (256, 256) with a cell size of (1/256, 1/256)
        self.pde = PDEUpwind(dx=(1. / 256, 1. / 256), source='none')
        # The time scheme integrator is a 4th order Runge-kutta.
        self.scheme = RungeKutta4(self.pde, delta_t=1, ndt=ndt)

    def forward(self, inputs: torch.Tensor):
        """Forward method

        Args:
            inputs (torch.Tensor): Input tensors

        Returns:
            torch.Tensor: Model(inputs)
        """
        x_context = inputs[0]
        X_0 = inputs[1]
        # The U-Net takes as input a tensor of size `(batch_size, context_size, H, W)`,
        # U = [u, v]
        # U = self.unet(x_context)
        # u = torch.mul(U[:, 1:, ...], self.dx).unsqueeze(-1)
        # v = torch.mul(U[:, :1, ...], self.dy).unsqueeze(-1)
        velocity_field = self.unet(x_context)
        # u = velocity_field[1] * dx
        u = torch.mul(velocity_field[:, 1:, ...], self.dx).unsqueeze(-1)
        # v = velocity_field[0] * dy
        v = torch.mul(velocity_field[:, :1, ...], self.dy).unsqueeze(-1)

        # # repeat u & v along the last dimension
        # # u = (batch_size, H, W, 1) -> (batch_size, H, W, n_classes)
        u = u.expand(*u.shape[:-1], self.n_classes)
        v = v.expand(*v.shape[:-1], self.n_classes)
        minus_u = -u
        minus_v = -v

        u_plus = F.relu(u)
        v_plus = F.relu(v)
        # Actually, v_minus = -F.relu(-v) but there is another negative sign considered in the
        # advective term, so we can just remove the negative sign in both terms as -(-F.relu(-v)) = F.relu(-v)
        # thus, 2 tensor operations are saved
        u_minus = F.relu(minus_u)
        v_minus = F.relu(minus_v)

        outputs = []
        # The initial state is the last observation in the context
        state = X_0
        # Time integration
        for _ in range(self.leadtime):
            # Time integration step
            state = self.scheme([state, u_plus, u_minus, v_plus, v_minus])
            
            # Normalization, this is not necessary but it is done to keep the
            # probabilities are between 0 and 1 in case of instabilities during the training
            # specifically, if the velocity values are too large, the CFL condition may not be satisfied
            state =  F.hardtanh(state, min_val=0, max_val=1)
            sum_p = torch.sum(state, dim=-1, keepdims=True)
            # Ensuring that the probabilities sum to 1
            state = torch.divide(state, sum_p)

            # The output maps represent is the probability of each class
            # The CrossEntropyLoss expects the input to be of shape [batch_size, n_classes, 256, 256]
            # So, we swap axis to get the right shape
            # Before: [batch_size, 1, 256, 256, n_classes]
            # After: [batch_size, n_classes, 256, 256]
            out = torch.swapaxes(state, -1, 1).squeeze(-1)
            outputs += [out]
        return outputs

class HyPhAI2(HyPhAIModule):
    def __init__(self, leadtime=8, n_classes=12, context_size=4, ndt=10):
        super().__init__(leadtime=leadtime, n_classes=n_classes,
                         context_size=context_size)
        # The PDE, here the advection equation with a source term
        # generated using a UNet
        self.pde = PDEUpwind(dx=(1. / 256, 1. / 256), source='hyphai-2')
        # Scheme used to solve the PDE, here 4th order Runge-Kutta.
        self.scheme = RungeKutta4(self.pde, delta_t=1, ndt=ndt)
        self.dt = 1 / ndt
        # The U-Net used to predict the source term
        self.unet_s = UNet(
            enc_chs=(context_size,32,64,128,256),
            dec_chs=(256,128,64,32),
            num_class=n_classes
        )

    def forward(self, inputs: torch.Tensor):
        x_context = inputs[0]
        X_0 = inputs[1]
        # velocity field = [u, v]
        velocity_field = self.unet(x_context)
        u = torch.mul(velocity_field[:, 1:, ...], self.dx).unsqueeze(-1)
        v = torch.mul(velocity_field[:, :1, ...], self.dy).unsqueeze(-1)

        # Repeat u & v along the last dimension
        # so u = [batch_size, 1, 256, 256, n_classes]
        u = u.expand(*u.shape[:-1], self.n_classes)
        v = v.expand(*v.shape[:-1], self.n_classes)

        u_plus = F.relu(u)
        v_plus = F.relu(v)
        # Actually, v_minus = -F.relu(-v) but the negative sign is considered in the
        # multiplication
        minus_u = -u
        minus_v = -v
        u_minus = F.relu(minus_u)
        v_minus = F.relu(minus_v)

        outputs = []
        # Initial condition
        state = X_0
        sliding_window = x_context        
        # Time integration
        for t in range(self.leadtime):
            second_member = self.unet_s(sliding_window)
            # bound the second member values to [-1, 1]
            second_member = torch.tanh(second_member)
            second_member = torch.swapaxes(second_member.unsqueeze(-1), 1, -1)  

            # Time integration step     
            state = self.scheme([state, u_plus,u_minus, v_plus, v_minus, second_member])

            # Normalization, this is not necessary but it is done to keep the
            # probabilities are between 0 and 1 in case of instabilities during the training
            # specifically, if the velocity values are too large, the CFL condition may not be satisfied
            state = HyPhAI2.custom_tanh(state)
            argmx = torch.argmax(state, dim=-1, keepdims=False)
            # Updating the sliding window
            sliding_window = torch.cat([sliding_window[:, 1:, ...], argmx], dim=1)
            # Ensuring that the sum of the probabilities is 1
            state = torch.divide(state, torch.sum(state, dim=-1, keepdims=True))
            
            # The CrossEntropyLoss expects the input to be of shape [batch_size, n_classes, 256, 256],
            # so, we swap axis to get the right shape 
            # Before: [batch_size, 1, 256, 256, n_classes]
            # After: [batch_size, n_classes, 256, 256]            
            out = torch.swapaxes(state, -1, 1).squeeze(-1)
            # Add the output to the forecast list
            outputs += [out]
        return outputs

class HyPhAI3(HyPhAIModule):
    def __init__(self, leadtime=8, n_classes=12, context_size=4, ndt=10):
        super().__init__(leadtime=leadtime, n_classes=n_classes,
                         context_size=context_size)
        # The PDE, here the advection equation with source term
        # The discretization of the PDE is done using the upwind scheme on a
        # grid of size (256, 256) with a cell size of (1/256, 1/256).
        self.pde = PDEUpwind(dx=(1. / 256, 1. / 256), source='hyphai-3')
        # Scheme used to solve the PDE, here 4th order Runge-Kutta.
        self.scheme = RungeKutta4(self.pde, delta_t=1, ndt=ndt)
        self.dt = 1 / ndt
        # UNet, here the UNet with 5 encoder layers and 4 decoder layers
        # This UNet is overridden to get two outputs, the first output is a tensor of size `(batch_size, source_ch, H, W)`.
        # The second output is a tensor of size `(batch_size, 2, H, W)`.
        self.unet = UNet_Xception(
            enc_chs=(context_size, 32, 64, 128, 256),
            dec_chs=(256, 128, 64, 32),
            num_class=2,
            smoothing_kernel=HyPhAIModule.smoothing_kernel(33)
        )
        self.unet_s = UNet(
            enc_chs=(context_size,256,512),
            dec_chs=(512,256),
            num_class=n_classes*n_classes
        )

    def forward(self, inputs: torch.Tensor):
        """Forward method

        Args:
            inputs (torch.Tensor): Input tensors

            Returns:
                torch.Tensor: Model(inputs)
        """
        # x_context, X_0, d
        x_context = inputs[0]
        X_0 = inputs[1]
        # velocity field = [u, v]
        # S = [P(C_(t+1)=j | C_t = j-1), P(C_(t+1)=j | C_t = j+1)]
        velocity_field = self.unet(x_context)
        # u = velocity_field[1] * dx
        u = torch.mul(velocity_field[:, 1:, ...], self.dx).unsqueeze(-1)
        # v = velocity_field[0] * dy
        v = torch.mul(velocity_field[:, :1, ...], self.dy).unsqueeze(-1)

        # Repeat u & v along the last dimension
        # so u = [batch_size, 1, 256, 256, n_classes]
        u = u.expand(*u.shape[:-1], self.n_classes)
        v = v.expand(*v.shape[:-1], self.n_classes)
        u_plus = F.relu(u)
        v_plus = F.relu(v)
        # Actually, v_minus = -F.relu(-v) but the negative sign is considered in the
        # multiplication
        minus_u = -u
        minus_v = -v
        u_minus = F.relu(minus_u)
        v_minus = F.relu(minus_v)
       
        m = self.unet_s(x_context)
        m = torch.moveaxis(m, 1, -1).unsqueeze(1)
        # Reshape from (batch_size, 1, 256, 256, n_classes * n_classes) to
        # (batch_size, 1, 256, 256, n_classes, n_classes)
        m = m.reshape(*m.shape[:-1], self.n_classes, self.n_classes)
        m = F.softmax(m, dim=-1)
        # M = [[P(C_(t+1)=1 | C_t = 1),..., P(C_(t+1)=1 | C_t = n_classes)],
        #      [P(C_(t+1)=2 | C_t = 1),..., P(C_(t+1)=2 | C_t = n_classes)],
        #                              ...
        #      [P(C_(t+1)=n_classes | C_t = 1),..., P(C_(t+1)=n_classes | C_t = n_classes)]]
        # m is the transition matrix, the source term is given by
        # lambda = (m-I)/dt        
        identity = torch.eye(self.n_classes).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
        # I is of shape (1, 1, 1, 1, n_classes, n_classes)
        # Repeat I to match the shape of m (batch_size, 1, 256, 256, n_classes, n_classes)
        identity = identity.repeat(*u.shape[:-1], 1, 1)
        lambda_rate = torch.sub(m, identity)/ self.dt
        
        outputs = []
        # Initial condition
        state = X_0
        # Time integration
        for _ in range(self.leadtime):
            # Time integration step
            state = self.scheme([state, u_plus,u_minus, v_plus, v_minus, lambda_rate])

            # Normalization, this is not necessary but it is done to keep the
            # probabilities are between 0 and 1 in case of instabilities during the training
            # specifically, if the velocity values are too large, the CFL condition may not be satisfied
            state = F.hardtanh(state, min_val=0, max_val=1)
            sum_p = torch.sum(state, dim=-1, keepdims=True)
            # Ensuring that the probabilities sum to 1
            state = torch.divide(state, sum_p)

            # The CrossEntropyLoss expects the input to be of shape [batch_size, n_classes, 256, 256]
            # For this reason we swap the the second and the last dimensions
            # Before: [batch_size, 1, 256, 256, n_classes]
            # After: [batch_size, n_classes, 256, 256]
            out = torch.swapaxes(state, -1, 1).squeeze(-1)
            # Appending the output to the list, each output is for a different
            # lead-time, 15 minutes apart
            outputs += [out]
        return outputs

class HyPhAI4(HyPhAIModule):
    def __init__(self, leadtime=8, n_classes=12, context_size=4, ndt=10, n_matrices=10):
        """__init__ method

        Args:
            leadtime (int, optional): Prediction lead-times. Defaults to 8.
            n_classes (int, optional): Number of classes. Defaults to 12.
            context_size (int, optional): Context observations size. Defaults to 4.
            n_matrices (int, optional): Number of matrices to use in the source term. Defaults to 10.
        """
        super().__init__(leadtime=leadtime, n_classes=n_classes,
                         context_size=context_size)
        # The PDE, here the advection equation with source term
        # The discretization of the PDE is done using the upwind scheme on a
        # grid of size (256, 256) with a cell size of (1/256, 1/256).
        self.pde = PDEUpwind(dx=(1. / 256, 1. / 256), source='hyphai-4')
        # Scheme used to solve the PDE, here 4th order Runge-Kutta.
        self.scheme = RungeKutta4(self.pde, delta_t=1, ndt=ndt)
        self.dt = 1 / ndt
        # UNet, here the UNet with 5 encoder layers and 4 decoder layers
        # This UNet is overridden to get two outputs, the first output is a tensor of size `(batch_size, source_ch, H, W)`.
        # The second output is a tensor of size `(batch_size, 2, H, W)`.
        self.unet = UNet_Xception(
            enc_chs=(context_size, 32, 64, 128, 256),
            dec_chs=(256, 128, 64, 32),
            num_class=2,
            smoothing_kernel=HyPhAIModule.smoothing_kernel(33)
        )
        self.unet_s = UNet(
            enc_chs=(context_size,64,128,256),
            dec_chs=(256,128,64),
            num_class=n_matrices+1
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transitions = nn.Parameter(torch.eye(self.n_classes).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device).repeat(n_matrices, 1,1,1, 1, 1), requires_grad=True)
        identity = torch.eye(self.n_classes).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        # I is of shape (1, 1, 1, 1, n_classes, n_classes)
        # Repeat I to match the shape of m (n_matrices, 1, 1, 1, n_classes, n_classes)
        self.identity = identity.repeat(n_matrices, 1,1,1, 1, 1)

    def forward(self, inputs: torch.Tensor):
        """Forward method

        Args:
            inputs (torch.Tensor): Input tensors

            Returns:
                torch.Tensor: Model(inputs)
        """
        # x_context, X_0, d
        x_context = inputs[0]
        X_0 = inputs[1]
        # velocity field = [u, v]
        # S = [P(C_(t+1)=j | C_t = j-1), P(C_(t+1)=j | C_t = j+1)]
        velocity_field = self.unet(x_context)
        # u = velocity_field[1] * dx
        u = torch.mul(velocity_field[:, 1:, ...], self.dx).unsqueeze(-1)
        # v = velocity_field[0] * dy
        v = torch.mul(velocity_field[:, :1, ...], self.dy).unsqueeze(-1)

        # Repeat u & v along the last dimension
        # so u = [batch_size, 1, 256, 256, n_classes]
        u = u.expand(*u.shape[:-1], self.n_classes)
        v = v.expand(*v.shape[:-1], self.n_classes)
        u_plus = F.relu(u)
        v_plus = F.relu(v)
        # Actually, v_minus = -F.relu(-v) but the negative sign is considered in the
        # multiplication
        minus_u = -u
        minus_v = -v
        u_minus = F.relu(minus_u)
        v_minus = F.relu(minus_v)

        matrices = (F.softmax(self.transitions, dim=-1) - self.identity)/self.dt
        T = self.unet_s(x_context)
        # T is of shape (batch_size, n_matrices, 256, 256)
        # making sure that the sum of the weights is 1 for each pixel
        weightings = (torch.relu(T) / torch.sum(torch.relu(T), dim=1, keepdim=True)).unsqueeze(-1)
        
        outputs = []
        # Initial condition
        state = X_0
        # Time integration 
        for _ in range(self.leadtime):
            # Time integration step
            state = self.scheme([state, u_plus,u_minus, v_plus, v_minus, [weightings, matrices]])

            # Normalization, this is not necessary but it is done to keep the
            # probabilities are between 0 and 1 in case of instabilities during the training
            # specifically, if the velocity values are too large, the CFL condition may not be satisfied
            state =  F.hardtanh(state, min_val=0, max_val=1)
            sum_p = torch.sum(state, dim=-1, keepdims=True)
            # Ensuring that the probabilities sum to 1
            state = torch.divide(state, sum_p)

            # The CrossEntropyLoss expects the input to be of shape [batch_size, n_classes, 256, 256]
            # For this reason we swap the the second and the last dimensions
            # Before: [batch_size, 1, 256, 256, n_classes]
            # After: [batch_size, n_classes, 256, 256]
            out = torch.swapaxes(state, -1, 1).squeeze(-1)
            # Appending the output to the list, each output is for a different
            # lead-time, 15 minutes apart
            outputs += [out]
        return outputs


class FullDLModel(nn.Module):
    """This module is used to train a standard U-Net for cloud cover nowcasting.
    """

    def __init__(self, leadtime=8, n_classes=12, context_size=4):
        """__init__ method

        Args:
            leadtime (int, optional): prediction lead-time. Defaults to 8.
            n_classes (int, optional): number of classes. Defaults to 12.
            context_size (int, optional): context observations size. Defaults to 4.
        """
        super().__init__()
        self.leadtime = leadtime
        self.n_classes = n_classes
        self.context_size = context_size
        self.unet = UNet(
            enc_chs=(context_size, 32, 64, 128, 256),
            dec_chs=(256, 128, 64, 32),
            num_class=n_classes)

    def set_device(self, device: str):
        pass

    def forward(self, inputs: torch.Tensor):
        """Forward method

        Args:
            inputs (torch.Tensor): Input tensors of shape (batch_size, context_size, H, W)

            Returns:
                list: List of tensors of shape (batch_size, n_classes, H, W), each tensor is the forecast for a different lead-time
        """
        x = inputs
        outputs = []
        # Get the last observation
        state = x[:, -1:, ...]
        # Iterate over multiple lead-times
        for _ in range(self.leadtime):
            # Concatenate the last observation / prediction with the previous observations
            x = torch.cat([x[:, 1:, ...], state], dim=1)
            # Get the prediction
            out = self.unet(x)
            # Get predicted labels for each pixel
            # out = torch.softmax(out, dim=1)
            state = torch.argmax(out, dim=1, keepdim=True)
            # Add the output to the forecast list
            outputs += [out]
        return outputs
