#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchdiffeq import odeint_adjoint

from thermo.models.ode_wrapper import ODEWrapper


class StandardIntegrator:
    def __init__(self, b, method: str='euler', n_step: int=100, start: float=0.0, end: float=1.0, return_dlogp=False):
        """## Initialize the integrator. 

        ### Args:
            - `b (_type_)`: The velocity field network.
            - `method (str, optional)`: The ODE solver method. Defaults to 'euler'.
            - `n_step (int, optional)`: The number of integration steps. Defaults to 100.
            - `start (float, optional)`: Start of the integration interval. Defaults to 0.0.
            - `end (float, optional)`: End of the integration interval. Defaults to 1.0.
            - `return_dlogp (bool, optional)`: If True, return the dlogp ODE as well. Defaults to False.
        """

        assert method in {"euler"}, "Only 'euler' method is supported for now."

        self.ode_wrapper = ODEWrapper(b, return_dlogp=return_dlogp)
        self.start, self.end = start, end
        self.n_step = n_step
        self.method = method
        self.return_dlogp = return_dlogp

    def rollout(self, pos0, beta0, beta1=None) -> tuple:

        """
        Solve the sample and dlogp ODEs given a batch of initial values.
        :param batch: The batch of molecular data.
        :return: The sample and dlogp ODEs.
        """

        batch_size, _ = pos0.shape

        dlogp = torch.zeros(batch_size, 1).to(pos0.device) if self.return_dlogp else None

        if self.return_dlogp:
            states = (pos0, dlogp)
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(pos0.device)

            ode = lambda t, states: self.ode_wrapper(t, states, beta0) if beta1 is None else self.ode_wrapper(t, states, beta0, beta1)
            x, dlogp = odeint_adjoint(
                func=ode,
                y0=states,
                t=integration_times,
                method=self.method,
                adjoint_params=(self.ode_wrapper.parameters())
            )
            dlopg = -dlogp
        else:
            states = (pos0,)
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(pos0.device)

            ode = lambda t, states: self.ode_wrapper(t, states, pos0, beta0) if beta1 is None else self.ode_wrapper(t, states, beta0, beta1)
            x = odeint_adjoint(
                func=ode,
                y0=states,
                t=integration_times,
                method=self.method,
                adjoint_params=(self.ode_wrapper.parameters())
            )
            
        return x, dlogp
