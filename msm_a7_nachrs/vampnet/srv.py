import numpy as np

import torch

from ..deepmsm.deepmsm import *
from .vampnet import VAMPNet_Multimer

from typing import Optional, Union, Callable, Tuple
from deeptime.util.torch import disable_TF32, multi_dot


def sym_vampnet_loss(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6,
                 mode: str = 'trunc'):
    r"""Loss function that can be used to train SymVAMPNets. It evaluates as :math:`-\mathrm{score}`. The score
    is implemented in :meth:`score`."""
    return -1. * sym_vamp_score(data, data_lagged, method=method, epsilon=epsilon, mode=mode)

def sym_vamp_score(data: torch.Tensor, data_lagged: torch.Tensor, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on symmetrized data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert method in valid_score_methods, f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, f"Data and data_lagged must be of same shape but were {data.shape} " \
                                            f"and {data_lagged.shape}."
    out = None
    if method == 'VAMP1':
        koopman = sym_koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = sym_koopman_matrix(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00 = data.T @ data + data_lagged.T @ data_lagged
        ctt = c00
        c0t = data_lagged.T @ data + data.T @ data_lagged

        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        out = torch.trace(
            2. * multi_dot([s, u_t, c0t, v]) - multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
        )
    assert out is not None
    return 1 + out 

def sym_koopman_matrix(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-6, mode: str = 'trunc',
                        ) -> torch.Tensor:
    r""" Computes the symmetrized Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """

    c00 = x.T @ x + y.T @ y
    ctt = c00
    c0t = y.T @ x + x.T @ y

    c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
    koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

    return koopman


class VAMPNet_Multimer_Sym(VAMPNet_Multimer):

    def partial_fit(self, data, train_score_callback: Callable[[
                    int, torch.Tensor], None] = None, tb_writer=None):
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

        self.lobe.train()
        self.lobe_timelagged.train()

        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(
                data[0].astype(
                    self.dtype)).to(
                device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(
                data[1].astype(
                    self.dtype)).to(
                device=self.device)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        # x_0_aug = torch.concat([torch.roll(x_0, self.n_states * i, 1) for i in range(self.multimer)])
        # x_t_aug = torch.concat([torch.roll(x_t, self.n_states * i, 1) for i in range(self.multimer)])
#        loss_value = vampnet_loss(x_0_aug, x_t_aug, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

        loss_value = sym_vampnet_loss(
            x_0,
            x_t,
            method=self.score_method,
            epsilon=self.epsilon,
            mode=self.score_mode)

        loss_value.backward()
        self.optimizer.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
            tb_writer.add_scalars('VAMPE', {'train': -loss_value.item()}, self._step)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._step += 1

        return self

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:

        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                # augmenting validation set by permutation
                val_aug = torch.concat(
                    [torch.roll(val, self.n_states * i, 1) for i in range(self.multimer)])
                val_t_aug = torch.concat(
                    [torch.roll(val_t, self.n_states * i, 1) for i in range(self.multimer)])
                score_value = sym_vamp_score(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value