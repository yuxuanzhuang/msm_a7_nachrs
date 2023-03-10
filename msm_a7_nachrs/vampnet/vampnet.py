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


class VAMPNETInitializer(MSMInitializer):
    prefix = "vampnet"

    def start_analysis(self):
        self._vampnets = []
        self._vampnet_dict = {}
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

    @property
    def vampnets(self):
        return self._vampnets
    
    @vampnets.setter
    def vampnets(self, value):
        self._vampnets = value
        self.select_vampnet(0)

    @property
    def state_list(self):
        return [f'{vampnet.n_states}_state_{vampnet.rep}_rep' for vampnet in self._vampnets]
    
    @property
    def vampnet_dict(self):
        if not self._vampnet_dict:
            self._vampnet_dict = {key: {} for key in self.state_list}
        return self._vampnet_dict
    
    def select_vampnet(self, index, update=False):
        self.active_vampnet = self.vampnets[index]
        self.active_vampnet_name = self.state_list[index]
        print('The activated VAMPNET # states:', self.active_vampnet.n_states)
        print('The activated VAMPNET # rep:', self.active_vampnet.rep)

        if not self.vampnet_dict[self.active_vampnet_name] or update:
            state_probabilities = [self.active_vampnet.transform(traj) for traj in self.dataset.trajectories]
            state_probabilities_concat = np.concatenate(state_probabilities)
            assignments = [stat_prob.argmax(1) for stat_prob in state_probabilities]
            assignments_concat = np.concatenate(assignments)

            self._vampnet_dict[self.active_vampnet_name]['state_probabilities'] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name]['state_probabilities_concat'] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]['assignments'] = assignments
            self._vampnet_dict[self.active_vampnet_name]['assignments_concat'] = assignments_concat
        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name]['state_probabilities']
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name]['state_probabilities_concat']
        self.assignments = self._vampnet_dict[self.active_vampnet_name]['assignments']
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name]['assignments_concat']

    def get_tica_model(self):
        print(f'Start TICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}')
        self.tica = TICA(lagtime=self.lag,
                         observable_transform=self.active_vampnet.fetch_model())
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):

            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
            batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

            self.tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.tica_model = self.tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]['tica_model'] = self.tica_model

        self.tica_output = [self.tica_model.transform(traj) for traj in self.dataset.trajectories]
        self.tica_concatenated = np.concatenate(self.tica_output)
        print('TICA shape:', self.tica_concatenated.shape)


class VAMPNETInitializer_Multimer(VAMPNETInitializer):
    def select_vampnet(self, index, update=False):
        self.active_vampnet = self.vampnets[index]
        self.active_vampnet_name = self.state_list[index]
        print('The activated VAMPNET # states:', self.active_vampnet.n_states)

        if not self.vampnet_dict[self.active_vampnet_name] or update:
            state_probabilities = [self.active_vampnet.transform(traj) for traj in self.dataset.trajectories]
            state_probabilities_concat = np.concatenate(state_probabilities)
            assignments = [stat_prob.reshape(stat_prob.shape[0],
                                 self.active_vampnet.multimer,
                                 self.active_vampnet.n_states).argmax(2) for stat_prob in state_probabilities]
            assignments_concat = np.concatenate(assignments)
            cluster_degen_dtrajs = []
            for sub_dtrajs in assignments:
            #    degenerated_traj = np.apply_along_axis(convert_state_to_degenerated, axis=1, arr=sub_dtrajs)
                sorted_sub_dtrajs = np.sort(sub_dtrajs, axis=1)[:,::-1]
                cluster_degen_dtrajs.append(np.sum(sorted_sub_dtrajs * (self.active_vampnet.n_states ** np.arange(self.active_vampnet.multimer)), axis=1))
            cluster_degen_concat = np.concatenate(cluster_degen_dtrajs)
            cluster_rank_concat = rankdata(cluster_degen_concat, method='dense') - 1
            print('# of cluster',cluster_rank_concat.max() + 1)
            self.n_clusters = cluster_rank_concat.max() + 1
            cluster_rank_dtrajs = []
            curr_ind = 0
            for sub_dtrajs in assignments:
                cluster_rank_dtrajs.append(cluster_rank_concat[curr_ind:curr_ind+sub_dtrajs.shape[0]])
                curr_ind += sub_dtrajs.shape[0]


            self._vampnet_dict[self.active_vampnet_name]['state_probabilities'] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name]['state_probabilities_concat'] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]['assignments'] = assignments
            self._vampnet_dict[self.active_vampnet_name]['assignments_concat'] = assignments_concat
            self._vampnet_dict[self.active_vampnet_name]['cluster_degen_dtrajs'] = cluster_degen_dtrajs
            self._vampnet_dict[self.active_vampnet_name]['cluster_degen_concat'] = cluster_degen_concat
            self._vampnet_dict[self.active_vampnet_name]['cluster_rank_dtrajs'] = cluster_rank_dtrajs
            self._vampnet_dict[self.active_vampnet_name]['cluster_rank_concat'] = cluster_rank_concat
        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name]['state_probabilities']
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name]['state_probabilities_concat']
        self.assignments = self._vampnet_dict[self.active_vampnet_name]['assignments']
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name]['assignments_concat']
        self.cluster_degen_dtrajs = self._vampnet_dict[self.active_vampnet_name]['cluster_degen_dtrajs']
        self.cluster_degen_concat = self._vampnet_dict[self.active_vampnet_name]['cluster_degen_concat']
        self.cluster_rank_dtrajs = self._vampnet_dict[self.active_vampnet_name]['cluster_rank_dtrajs']
        self.cluster_rank_concat = self._vampnet_dict[self.active_vampnet_name]['cluster_rank_concat']

    def get_tica_model(self):
        print(f'Start SymTICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}')
        self.tica = SymTICA(
                            symmetry_fold=self.active_vampnet.multimer,
                            lagtime=self.lag,
                            observable_transform=self.active_vampnet.fetch_model())
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):
            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
            batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

            self.tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.tica_model = self.tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]['tica_model'] = self.tica_model

        self.tica_output = [self.tica_model.transform(traj) for traj in self.dataset.trajectories]
        self.tica_concatenated = np.concatenate(self.tica_output)
        print('TICA shape:', self.tica_concatenated.shape)

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
                 rep: int = 0,
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
        self.rep = rep
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

    def save(self, folder, n_epoch, rep=None):
        if rep is None:
            rep = 0
            
        pickle.dump(self,
            open(f'{folder}/{self.__class__.__name__}/epoch_{n_epoch}_state_{self.n_states}_rep_{rep}.lobe', 'wb'))


class VAMPNet_Multimer_AUG(VAMPNet_Multimer):        
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