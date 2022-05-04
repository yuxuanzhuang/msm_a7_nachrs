from ..msm.MSM_a7 import MSMInitializer

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

from deeptime.decomposition.deep import VAMPNet
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data


class MultimerTrajectoriesDataset(TrajectoriesDataset):
    """
    A dataset for multimer trajectories.
    Warning: The features in this dataset should be n-fold symmetric.
    """
    def __init__(self, multimer: int, data: List[TrajectoryDataset]):
        self.multimer = multimer
        super().__init__(data)

    @staticmethod
    def from_numpy(lagtime, multimer, data: List[np.ndarray]):
        assert isinstance(data, list)
        assert len(data) > 0 and all(data[0].shape[1:] == x.shape[1:] for x in data), "Shape mismatch!"

        data_new = []
        total_shape = data[0].shape[1]
        per_shape = int(total_shape / multimer)

        for i in range(multimer):
            data_new.extend(
                [np.roll(traj.reshape(traj.shape[0],multimer, per_shape),
                                                i, axis=1).reshape(traj.shape[0],total_shape)
                for traj in data])
        return MultimerTrajectoriesDataset(multimer, [TrajectoryDataset(lagtime, traj) for traj in data_new])


class VAMPNETInitializer(MSMInitializer):
    prefix="vampnet"

    def start_analysis(self):
        if (not os.path.isfile(self.filename  + 'vampnet.pyemma')) or self.updating:
            print('Start new VAMPNET analysis')
            if not self.data_collected:
                self.gather_feature_matrix()

    #        self.dataset = MultimerTrajectoriesDataset.from_numpy(self.lag, self.multimer, self.feature_trajectories)
            self.dataset = TrajectoriesDataset.from_numpy(lagtime=self.lag,
                                                          data=self.feature_trajectories)


            with open(self.filename + 'vampnet_init.pickle', 'wb') as f:
                pickle.dump(self, f)
        else:
            print('Load old VAMPNET results')
            self = pickle.load(open(self.filename + 'vampnet_init.pickle', 'rb'))

        
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
        
        x_stack = torch.permute(x_splits, (1,0,2)).reshape(batch_size * self.multimer, self.n_feat_per_sub)

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
        
        x_splits = x_stack.reshape(self.multimer, batch_size, self.n_states).permute(1,0,2).reshape(batch_size, self.n_states * self.multimer)
        return x_splits

class VAMPNet_Multimer(VAMPNet):
    def __init__(self, 
                 multimer: int,
                 n_states: int,
                 lobe: nn.Module, lobe_timelagged: Optional[nn.Module] = None,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_method: str = 'VAMP2', score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32):
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
        
        if self.multimer != self.lobe.module.multimer:
            raise ValueError('Mismatch multimer between vampnet and lobe')
        if self.n_states != self.lobe.module.n_states:
            raise ValueError('Mismatch multimer between vampnet and lobe')
        
    def partial_fit(self, data, train_score_callback: Callable[[int, torch.Tensor], None] = None):
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
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)
        
        # x_0_aug = torch.concat([torch.roll(x_0, self.n_states * i, 1) for i in range(self.multimer)])
        # x_t_aug = torch.concat([torch.roll(x_t, self.n_states * i, 1) for i in range(self.multimer)])
#        loss_value = vampnet_loss(x_0_aug, x_t_aug, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

        loss_value = vampnet_loss(x_0, x_t, method=self.score_method, epsilon=self.epsilon, mode=self.score_mode)

        loss_value.backward()
        self.optimizer.step()

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
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
                val_aug = torch.concat([torch.roll(val, self.n_states * i, 1) for i in range(self.multimer)])
                val_t_aug = torch.concat([torch.roll(val_t, self.n_states * i, 1) for i in range(self.multimer)])
                score_value = vamp_score(val_aug, val_t_aug, method=self.score_method, mode=self.score_mode, epsilon=self.epsilon)
                return score_value