import warnings
from ..msm.MSM_a7 import MSMInitializer
from ..util.dataloader import MultimerTrajectoriesDataset

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from typing import Optional, List
from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset

from tqdm.notebook import tqdm  # progress bar
import deeptime
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.base import Model, Transformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import map_data
from deeptime.markov.tools.analysis import pcca_memberships
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.decomposition import VAMP, TICA
from torch.utils.data import DataLoader

from deeptime.decomposition.deep import VAMPNet
from ..deepmsm.deepmsm import *
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from sklearn import preprocessing
from scipy.stats import rankdata
from ..tica.sym_tica import SymTICA
from .vampnet import VAMPNet_Multimer, VAMPNet_Multimer_AUG


class VAMPNet_Multimer_SYM(VAMPNet_Multimer):
    def partial_fit(self, data, train_score_callback: Callable[[
                    int, torch.Tensor], None] = None, tb_writer=None):
        self.trained = True

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

        n_feat_per_sub = batch_0.shape[1] // self.multimer

        # augmenting training set by permutation
        batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
        batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)
        
        loss_value = -vamp_score_sym(x_0, x_t, symmetry_fold=self.multimer, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

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
                score_value = vamp_score_sym(
                    val_aug,
                    val_t_aug,
                    symmetry_fold=self.multimer,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value

class VAMPNet_Multimer_SYM_REV(VAMPNet_Multimer_SYM):
    def partial_fit(self, data, train_score_callback: Callable[[
                    int, torch.Tensor], None] = None, tb_writer=None):
        self.trained = True

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

        n_feat_per_sub = batch_0.shape[1] // self.multimer

        # augmenting training set by permutation
        batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
        batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)
        
        loss_value = -vamp_score_sym(x_0, x_t, symmetry_fold=self.multimer,
                                     reversible=True,
                                     method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

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
                score_value = vamp_score_sym(
                    val_aug,
                    val_t_aug,
                    symmetry_fold=self.multimer,
                    reversible=True,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value    

def vamp_score_sym(data: torch.Tensor, data_lagged: torch.Tensor, symmetry_fold: int, 
                   reversible: bool = False,
                   method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
    r"""Computes the VAMP score based on data and corresponding time-shifted data with symmetry.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    symmetry_fold : int
        fold of symmetry
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
        koopman = koopman_matrix_sym(data,
                                     data_lagged, symmetry_fold,
                                     reversible=reversible,
                                     epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix_sym(data,
                                     data_lagged, symmetry_fold,
                                     reversible=reversible,
                                     epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        if reversible:
            c0t = 0.5 * (c0t + c0t.t())
            c00 = 0.5 * (c00 + ctt.t())
            ctt = c00

        if c00.shape[0] % symmetry_fold != 0:
            raise ValueError(f"Number of features {c00.shape[0]} must" +
                             f"be divisible by symmetry_fold {symmetry_fold}.")
        subset_rank = c00.shape[0] // symmetry_fold

        cov_00_blocks = []
        cov_0t_blocks = []
        cov_tt_blocks = []
        for i in range(symmetry_fold):
            cov_00_blocks.append(c00[:subset_rank,
                                 i * subset_rank:(i + 1) * subset_rank])
            cov_0t_blocks.append(c0t[:subset_rank,
                                 i * subset_rank:(i + 1) * subset_rank])
            cov_tt_blocks.append(ctt[:subset_rank,
                                 i * subset_rank:(i + 1) * subset_rank])
                             
#        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
#        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
#        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
        cov_00 = torch.sum(torch.stack(cov_00_blocks), axis=0)
        cov_0t = torch.sum(torch.stack(cov_0t_blocks), axis=0)
        cov_tt = torch.sum(torch.stack(cov_tt_blocks), axis=0)


        c00_sqrt_inv = sym_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(cov_tt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, cov_0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        out = torch.trace(
            2. * multi_dot([s, u_t, cov_0t, v]) - multi_dot([s, u_t, cov_00, u, s, v_t, cov_tt, v])
        )
    assert out is not None
    return 1 + out
    

def koopman_matrix_sym(x: torch.Tensor, y: torch.Tensor, symmetry_fold: int,
                       reversible: bool = False,
                       epsilon: float = 1e-6, mode: str = 'trunc',
                       c_xx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
    r""" Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    symmetry_fold : int
        fold of symmetry
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True)
    if reversible:
        c0t = 0.5 * (c0t + c0t.t())
        c00 = 0.5 * (c00 + ctt.t())
        ctt = c00

    if c00.shape[0] % symmetry_fold != 0:
        raise ValueError(f"Number of features {c00.shape[0]} must" +
                         f"be divisible by symmetry_fold {symmetry_fold}.")
    subset_rank = c00.shape[0] // symmetry_fold

    cov_00_blocks = []
    cov_0t_blocks = []
    cov_tt_blocks = []
    for i in range(symmetry_fold):
        cov_00_blocks.append(c00[:subset_rank,
                                i * subset_rank:(i + 1) * subset_rank])
        cov_0t_blocks.append(c0t[:subset_rank,
                                i * subset_rank:(i + 1) * subset_rank])
        cov_tt_blocks.append(ctt[:subset_rank,
                                i * subset_rank:(i + 1) * subset_rank])
                            
#        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
#        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
#        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
    cov_00 = torch.sum(torch.stack(cov_00_blocks), axis=0)
    cov_0t = torch.sum(torch.stack(cov_0t_blocks), axis=0)
    cov_tt = torch.sum(torch.stack(cov_tt_blocks), axis=0)

    c00_sqrt_inv = sym_inverse(cov_00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(cov_tt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, cov_0t, ctt_sqrt_inv]).t()