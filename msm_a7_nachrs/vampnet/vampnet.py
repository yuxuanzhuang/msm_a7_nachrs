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


from deeptime.decomposition.deep import VAMPNet
from ..deepmsm.deepmsm import *
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from sklearn import preprocessing

class VAMPNETInitializer(MSMInitializer):
    prefix = "vampnet"

    def start_analysis(self):
        os.makedirs(self.filename, exist_ok=True)

        if (not os.path.isfile(self.filename + 'vampnet.pyemma')) or self.updating:
            print('Start new VAMPNET analysis')
            if self.in_memory:
                if not self.data_collected:
                    self.gather_feature_matrix()
            else:
                print('Partial fitting is not supported in VAMPNET')
                if not self.data_collected:
                    self.gather_feature_matrix()
                    
            #self.dataset = MultimerTrajectoriesDataset.from_numpy(
            #    self.lag, self.multimer, self.feature_trajectories)

            self.dataset = TrajectoriesDataset.from_numpy(lagtime=self.lag,
                                                          data=self.feature_trajectories)

            if not self.symmetrize:
                self.dataset_sym = MultimerTrajectoriesDataset.from_numpy(
                            self.lag, self.multimer, self.feature_trajectories)

            if self.dumping:
                self.dump_feature_trajectories()

#                with open(self.filename + 'vampnet_init.pickle', 'wb') as f:
#                    pickle.dump(self, f)
        else:
            print('Load old VAMPNET results')
#            self = pickle.load(open(self.filename + 'vampnet_init.pickle', 'rb'))


class MultimerNet(nn.Module):
    def __init__(self, data_shape, multimer, n_states):
        super().__init__()
        self.data_shape = data_shape
        self.multimer = multimer
        self.n_states = n_states

        self.n_feat_per_sub = self.data_shape // self.multimer
        self._construct_architecture()

    def _construct_architecture(self):
        self.batchnorm1d = nn.BatchNorm1d(self.n_feat_per_sub)

        # Fully connected layers into monomer part
        self.fc1 = nn.Linear(self.n_feat_per_sub, 200)
        self.elu1 = nn.ELU()

        self.fc2 = nn.Linear(200, 100)
        self.elu2 = nn.ELU()

        self.fc3 = nn.Linear(100, 50)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(50, 20)
        self.elu4 = nn.ELU()

        self.fc5 = nn.Linear(20, self.n_states)
        self.softmax = nn.Softmax(dim=1)

#        self.fc6 = nn.Linear(20, 2)
#        self.elu6 = nn.ELU()

#        self.fc7 = nn.Linear(2, self.n_states)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    # x represents our data
    def forward(self, x):
     #       x = self.batchnorm1d(x)

        batch_size = x.shape[0]

        n_feat_per_sub = int(self.data_shape / self.multimer)
        x_splits = x.reshape(batch_size, self.multimer, self.n_feat_per_sub)
        output = []

        x_stack = torch.permute(x_splits, (1, 0, 2)).reshape(
            batch_size * self.multimer, self.n_feat_per_sub)

        x_stack = self.batchnorm1d(x_stack)
        x_stack = self.fc1(x_stack)
        x_stack = self.elu1(x_stack)
        x_stack = self.dropout1(x_stack)
        x_stack = self.fc2(x_stack)
        x_stack = self.elu2(x_stack)
        x_stack = self.dropout2(x_stack)
        x_stack = self.fc3(x_stack)
        x_stack = self.elu3(x_stack)
        x_stack = self.fc4(x_stack)
        x_stack = self.elu4(x_stack)
        x_stack = self.fc5(x_stack)
#        x_stack = self.fc6(x_stack)
#        x_stack = self.elu6(x_stack)
#        x_stack = self.fc7(x_stack)
        x_stack = self.softmax(x_stack)

        x_splits = x_stack.reshape(
            self.multimer,
            batch_size,
            self.n_states).permute(
            1,
            0,
            2).reshape(
            batch_size,
            self.n_states * self.multimer)
        return x_splits


class VAMPNet_Multimer(VAMPNet):
    def __init__(self,
                 multimer: int,
                 n_states: int,
                 lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32,
                 trained=False):
        super().__init__(lobe,
                         lobe_timelagged,
                         device,
                         optimizer,
                         learning_rate,
                         score_method,
                         score_mode,
                         epsilon,
                         dtype)

        self.multimer = multimer
        self.n_states = n_states
        self.trained = trained

        """
        try:
            if self.multimer != self.lobe.module.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.module.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        except AttributeError:
            if self.multimer != self.lobe.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        """
        
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
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        # x_0_aug = torch.concat([torch.roll(x_0, self.n_states * i, 1) for i in range(self.multimer)])
        # x_t_aug = torch.concat([torch.roll(x_t, self.n_states * i, 1) for i in range(self.multimer)])
#        loss_value = vampnet_loss(x_0_aug, x_t_aug, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

        loss_value = vampnet_loss(
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
                score_value = vamp_score(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value

    def transform(self, data, **kwargs):
        r""" Transforms data with the encapsulated model.

        Parameters
        ----------
        data : array_like
            Input data
        **kwargs
            Optional arguments.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        if not self.trained:
            warnings.warn( 'VAMPNet not trained yet. Please call fit first.')
        model = self.fetch_model()
        if model is None:
            raise ValueError("This estimator contains no model yet, fit should be called first.")
        return model.transform(data, **kwargs)


class VAMPNet_Multimer_Aug(VAMPNet):
    def __init__(self,
                 multimer: int,
                 n_states: int,
                 lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32,
                 trained=False):
        super().__init__(lobe,
                         lobe_timelagged,
                         device,
                         optimizer,
                         learning_rate,
                         score_method,
                         score_mode,
                         epsilon,
                         dtype)

        self.multimer = multimer
        self.n_states = n_states
        self.trained = trained

        """
        try:
            if self.multimer != self.lobe.module.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.module.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        except AttributeError:
            if self.multimer != self.lobe.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        """
        
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

        n_feat_per_state = batch_0.shape[1] // self.multimer
        batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_state * i, 1) for i in range(self.multimer)])
        batch_t = torch.concat([torch.roll(batch_t, n_feat_per_state * i, 1) for i in range(self.multimer)])

        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        loss_value = vampnet_loss(
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
                score_value = vamp_score(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value

    def transform(self, data, **kwargs):
        r""" Transforms data with the encapsulated model.

        Parameters
        ----------
        data : array_like
            Input data
        **kwargs
            Optional arguments.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        if not self.trained:
            warnings.warn( 'VAMPNet not trained yet. Please call fit first.')
        model = self.fetch_model()
        if model is None:
            raise ValueError("This estimator contains no model yet, fit should be called first.")
        return model.transform(data, **kwargs)


class VAMPNet_Multimer_NOSYM(VAMPNet):
    def __init__(self,
                 multimer: int,
                 n_states: int,
                 lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32,
                 trained=False):
        super().__init__(lobe,
                         lobe_timelagged,
                         device,
                         optimizer,
                         learning_rate,
                         score_method,
                         score_mode,
                         epsilon,
                         dtype)

        self.multimer = multimer
        self.n_states = n_states
        self.trained = trained

        """
        try:
            if self.multimer != self.lobe.module.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.module.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        except AttributeError:
            if self.multimer != self.lobe.multimer:
                raise ValueError('Mismatch multimer between vampnet and lobe')
            if self.n_states != self.lobe.n_states:
                raise ValueError('Mismatch multimer between vampnet and lobe')
        """
        
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
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        x_0_aug = torch.concat([torch.roll(x_0, self.n_states * i, 1) for i in range(self.multimer)])
        x_t_aug = torch.concat([torch.roll(x_t, self.n_states * i, 1) for i in range(self.multimer)])
        loss_value = -vamp_score_nosym(x_0_aug, x_t_aug, symmetry_fold=self.multimer, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

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
                score_value = vamp_score_nosym(
                    val_aug,
                    val_t_aug,
                    symmetry_fold=self.multimer,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon)
                return score_value

    def transform(self, data, **kwargs):
        r""" Transforms data with the encapsulated model.

        Parameters
        ----------
        data : array_like
            Input data
        **kwargs
            Optional arguments.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        if not self.trained:
            warnings.warn( 'VAMPNet not trained yet. Please call fit first.')
        model = self.fetch_model()
        if model is None:
            raise ValueError("This estimator contains no model yet, fit should be called first.")
        return model.transform(data, **kwargs)


def vamp_score_nosym(data: torch.Tensor, data_lagged: torch.Tensor, symmetry_fold: int, method='VAMP2', epsilon: float = 1e-6, mode='trunc'):
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
        koopman = koopman_matrix_nosym(data, data_lagged, symmetry_fold, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p='nuc')
    elif method == 'VAMP2':
        koopman = koopman_matrix_nosym(data, data_lagged, symmetry_fold, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p='fro'), 2)
    elif method == 'VAMPE':
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)

        if c00.shape[0] % symmetry_fold != 0:
            raise ValueError(f"Number of features {c00.shape[0]} must" +
                             f"be divisible by symmetry_fold {symmetry_fold}.")
        subset_rank = c00.shape[0] // symmetry_fold

        c00 = c00[:subset_rank, :subset_rank]
        c0t = c0t[:subset_rank, :subset_rank]
        ctt = ctt[:subset_rank, :subset_rank]

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

def koopman_matrix_nosym(x: torch.Tensor, y: torch.Tensor, symmetry_fold: int, epsilon: float = 1e-6, mode: str = 'trunc',
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

    if c00.shape[0] % symmetry_fold != 0:
        raise ValueError(f"Number of features {c00.shape[0]} must" +
                         f"be divisible by symmetry_fold {symmetry_fold}.")
    subset_rank = c00.shape[0] // symmetry_fold

    c00 = c00[:subset_rank, :subset_rank]
    c0t = c0t[:subset_rank, :subset_rank]
    ctt = ctt[:subset_rank, :subset_rank]

    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()