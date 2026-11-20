import torch

from abc import ABC, abstractmethod


class BaseInterpolant(ABC):
    def __init__(self):
        super(BaseInterpolant, self).__init__()

    @abstractmethod
    def _calc_pos_t(self, t: torch.tensor, pos0: torch.tensor, pos1: torch.tensor):
        pass

    @abstractmethod
    def interpolate(self, pos0, pos1):
        pass


class OneSidedLinearInterpolant(BaseInterpolant):

    def __init__(self) -> None:
        super(OneSidedLinearInterpolant, self).__init__()

        # one-sided linear interpolant
        self.a = lambda t: (1-t)
        self.adot = lambda t: -1.0
        self.b = lambda t: t
        self.bdot = lambda t: 1.0
        
        self.It = lambda t, pos0, pos1: self.a(t)*pos0 + self.b(t)*pos1
        self.dtIt = lambda t, pos0, pos1: self.adot(t)*pos0 + self.bdot(t)*pos1

        self.t_distr = torch.distributions.Beta(1.5, 0.9)

    def _calc_pos_t(self, t: torch.tensor, pos0: torch.tensor, pos1: torch.tensor) -> torch.tensor:
        """## Calculate the interpolated states at time t for one-sided linear interpolant.

        ### Args:
            - `t (torch.tensor)`: time
            - `pos0 (torch.tensor)`: random standard Gaussian noise
            - `pos1 (torch.tensor)`: final state

        ### Returns:
            - `torch.tensor`: interpolated state
        """
        
        return self.It(t, pos0, pos1)

    def interpolate(self, pos0, pos1):
        t = self.t_distr.sample((pos0.shape[0], 1)).to(pos0.device).to(pos0.dtype)
        pos_t = self._calc_pos_t(t, pos0, pos1)
        pos_t = pos_t.to(pos0.device).to(pos0.dtype)
        return pos_t, t


class LinearInterpolant(BaseInterpolant):

    def __init__(self, gamma_strategy="sin2", a=None) -> None:
        super(LinearInterpolant, self).__init__()
        
        if gamma_strategy == "brownian":
            assert a is not None, "a must be provided for brownian gamma strategy"
            self.gamma = lambda t: torch.sqrt(a*t*(1-t))
            self.gamma_dot = lambda t: (1/(2*torch.sqrt(a*t*(1-t)))) * a*(1 -2*t)
            self.gg_dot = lambda t: (a/2)*(1-2*t)
            
        elif gamma_strategy == "sin2":
            self.gamma = lambda t: torch.sin(torch.pi * t)**2
            self.gamma_dot = lambda t: 2*torch.pi*torch.sin(torch.pi * t)*torch.cos(torch.pi*t)
            self.gg_dot = lambda t: self.gamma(t)*self.gamma_dot(t)

        # linear interpolant
        self.a = lambda t: (1-t)
        self.adot = lambda t: -1.0

        self.b = lambda t: t
        self.bdot = lambda t: 1.0

        self.It = lambda t, pos0, pos1: self.a(t)*pos0 + self.b(t)*pos1
        self.dtIt = lambda t, pos0, pos1: self.adot(t)*pos0 + self.bdot(t)*pos1

        self.t_distr = torch.distributions.Beta(1.5, 0.9)

    def _calc_pos_t(self, t: torch.tensor, pos0: torch.tensor, pos1: torch.tensor):
        """## Calculate the interpolated states at time t for one-sided linear interpolant with antithetic sampling.

        ### Args:
            - `t (torch.tensor)`: time
            - `pos0 (torch.tensor)`: random standard Gaussian noise
            - `pos1 (torch.tensor)`: final state

        ### Returns:
            - `torch.tensor`: positive interpolated state
            - `torch.tensor`: negative interpolated state
            - `torch.tensor`: time
            - `torch.tensor`: gaussian noise from antithetic sampling
        """

        z = torch.randn(pos0.shape).to(pos0.device).to(pos0.dtype)
        gamma = self.gamma(t)
        It = self.It(t, pos0, pos1)
        return It + gamma*z, It - gamma*z, z

    def interpolate(self, pos0, pos1):
        t = self.t_distr.sample((pos0.shape[0], 1)).to(pos0.device).to(pos0.dtype)

        pos_t_plus, pos_t_minus, z = self._calc_pos_t(t, pos0, pos1)
        pos_t_plus, pos_t_minus, z = pos_t_plus.to(pos0.device).to(pos0.dtype), pos_t_minus.to(pos0.device).to(pos0.dtype), z.to(pos0.device).to(pos0.dtype)

        return pos_t_plus, pos_t_minus, t, z