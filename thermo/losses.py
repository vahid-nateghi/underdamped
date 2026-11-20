import torch

import thermo.interpolants as interpolants

from typing import Callable
from abc import abstractmethod


class BaseVelocityLoss(torch.nn.Module):
    def __init__(self, interpolant) -> None:
        super(BaseVelocityLoss, self).__init__()
        self.interpolant = interpolant

    @abstractmethod
    def forward(self, b, pos0, pos1, beta0, beta1=None):
        pass
    
    @abstractmethod
    def loss_per_sample(self, t, pos0, pos1, bt):
        pass

    @abstractmethod
    def make_batch_loss(self) -> Callable:
        pass


class OneSidedVelocityLoss(BaseVelocityLoss):
    def __init__(self, interpolant) -> None:
        assert isinstance(interpolant, interpolants.OneSidedLinearInterpolant)
        super(OneSidedVelocityLoss, self).__init__(interpolant)

    def forward(self, b, pos0, pos1, beta0, beta1=None):
        pos_t, t = self.interpolant.interpolate(pos0, pos1)

        # get bt
        bt = b(pos_t, t, beta1)

        # calc loss
        loss_fn = self.make_batch_loss()
        loss_val = loss_fn(t, pos0, pos1, bt).mean()
        return loss_val

    def loss_per_sample(self, t, pos0, pos1, bt):
        dtIt = self.interpolant.dtIt(t, pos0, pos1)
        
        loss = 0.5*torch.sum(bt**2) - torch.sum((dtIt) * bt)
        return loss

    def make_batch_loss(self) -> Callable:
        in_dims = (0, 0, 0, 0)  # share batch dimension
        batched_loss = torch.vmap(self.loss_per_sample, in_dims=in_dims, randomness='different')
        return batched_loss


class TwoSidedVelocityLoss(BaseVelocityLoss):
    def __init__(self, interpolant) -> None:
        assert isinstance(interpolant, interpolants.LinearInterpolant)
        super(TwoSidedVelocityLoss, self).__init__(interpolant)

    def forward(self, b, pos0, pos1, beta0, beta1):
        pos_t_plus, pos_t_minus, t, z = self.interpolant.interpolate(pos0, pos1)

        # get bt_plus and bt_minus
        bt_plus = b(pos_t_plus, t, beta0, beta1)
        bt_minus = b(pos_t_minus, t, beta0, beta1)

        # calc loss
        loss_fn = self.make_batch_loss()
        loss_val = loss_fn(t, z, pos0, pos1, bt_plus, bt_minus).mean()
        return loss_val

    def loss_per_sample(self, t, z, pos0, pos1, bt_plus, bt_minus):
        dtIt = self.interpolant.dtIt(t, pos0, pos1)
        gamma_dot = self.interpolant.gamma_dot(t)

        # compute loss from two "directions" for improved numerical stability
        loss = 0.5*torch.sum(bt_plus**2) - torch.sum((dtIt + gamma_dot*z) * bt_plus)
        loss += 0.5*torch.sum(bt_minus**2) - torch.sum((dtIt - gamma_dot*z) * bt_minus)
        return loss
    
    def make_batch_loss(self) -> Callable:
        in_dims = (0, 0, 0, 0, 0, 0)  # share batch dimension
        batched_loss = torch.vmap(self.loss_per_sample, in_dims=in_dims, randomness='different')
        return batched_loss
    