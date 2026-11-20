import torch
import torch.nn as nn


class ODEWrapper(nn.Module):

    """
    Wrap the stochastic interpolants sample ODE and dlogp ODE into a single PyTorch module that is compatible with the
    torchdiffeq format.
    """

    def __init__(self, b: nn.Module, return_dlogp=False, reverse_ode=False) -> None:
        """
        Initialize the ODE wrapper.
        :param b: The velocity field neural network.
        :param return_dlogp: If True, return the log sample prabilities as well.
        :param reverse_ode: If True, reverse the ODE direction.
        :return: None
        """
        super(ODEWrapper, self).__init__()

        self.b = b
        self.return_dlogp = return_dlogp
        self.reverse_ode = reverse_ode

    def forward(self, integration_time: torch.tensor, states: tuple, beta0: torch.tensor, beta1: torch.tensor=None) -> tuple:

        """
        Forward pass of the ODE wrapper. This is the function that is called by the torchdiffeq odeint_adjoint.
        :param integration_time: The time at which the ODE is evaluated.
        :param states: Current integration states, i.e. (x_{t-1}, dlogp_{t-1}).
        :param batch: Torch Geometric batch of molecular data.
        :return: Tuple of value and divergence of the vector field.
        """

        if self.return_dlogp:
            xs, _ = states
            ts = torch.ones_like(xs[:, 0]).unsqueeze(-1)*integration_time

            b = self.b(xs, ts, beta0, beta1)

            divergence = self.compute_divergence(xs, ts, beta0, beta1)
            return (b, -divergence) if not self.reverse_ode else (-b, divergence)  # NOTE: reverse sampling is not implemented in integrator yet
        
        else:
            with torch.no_grad():
                xs, = states
                ts = torch.ones_like(xs[:, 0]).unsqueeze(-1)*integration_time
                b = self.b(xs, ts, beta0, beta1)
                return b

    def compute_divergence(self, xs: torch.tensor, ts: torch.tensor, beta0: torch.tensor, beta1: torch.tensor=None) -> torch.tensor:
        bs = xs.shape[0]

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            ts.requires_grad_(True)
            b_val = self.b(xs, ts, beta0, beta1)
            divergence = 0.0

            for i in range(xs.shape[1]):
                divergence += torch.autograd.grad(b_val[:, i].sum(), xs, create_graph=True)[0][:, i]
        return divergence.view(bs)

