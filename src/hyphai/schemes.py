import torch
from torch.nn import Module
from torch.nn import functional as F


class RungeKutta4(Module):
    """
    PyTorch module implementing the 4th order Runge-Kutta (RK4) scheme.

    Args:
        trend (module): A Model that takes as inputs tensors representing the equation variables at time t
            and returns a tensor representing the trend at time t, i.e. the time derivative of the state.
        delta_t (float): The time step size.
        ndt (int): The number of sub-steps to take within each time step.
    """

    def __init__(self, trend: Module, delta_t=1, ndt=5):
        super().__init__()
        self.trend = trend
        self.dt = delta_t / ndt
        self.ndt = ndt

    def forward(self, inputs) -> torch.Tensor or tuple:
        """
        Computes one step of the 4th order Runge-Kutta scheme.

        Args:
            inputs (torch.Tensor): Equation variables at time t.

        Returns:
            torch.Tensor or tuple: The state at time t + delta_t
        """

        c = inputs[0]
        other_inputs = inputs[1:]
        for _ in range(self.ndt):
            # k1 = f(y_t)
            k1 = self.trend([c, *other_inputs])

            # k2 = f(y_t + dt * k1 / 2)
            input_k2 = torch.add(c, k1, alpha=self.dt / 2)  # out = c + dt/2*k1
            k2 = self.trend([input_k2, *other_inputs])

            # k3 = f(y_t + dt * k2 / 2)
            input_k3 = torch.add(c, k2, alpha=self.dt / 2)
            k3 = self.trend([input_k3, *other_inputs])

            # k4 = f(y_t + dt * k3)
            input_k4 = torch.add(c, k3, alpha=self.dt)
            k4 = self.trend([input_k4, *other_inputs])

            # output
            # k2 + k3
            _sum_k2_k3 = torch.add(k2, k3)
            # k1 + k4
            _sum_k1_k4 = torch.add(k1, k4)
            # k1 + k2/2 + k3/2 + k4
            _sum = torch.add(_sum_k1_k4, _sum_k2_k3, alpha=2.)
            # y_(t+dt) = y_t + (k1 + k2/2 + k3/2 + k4)*dt/6
            c = torch.add(c, _sum, alpha=self.dt / 6.)
            
        return c
    