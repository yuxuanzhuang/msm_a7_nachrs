from MSM_a7 import *
from VAMPNET_init import VAMPNET_Initializer, MultimerTrajectoriesDataset
import pandas as pd
from random import seed
import random as rm
import itertools
import pyemma

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
torch.set_num_threads(32)

print(f"Using device {device}")

system_exclusion_dic = {'BGT_EPJPNU': [],
                   'BGT_EPJ': [],
                   'EPJPNU_BGT': [21],
                   'EPJPNU_EPJ': [],
                   'EPJ_BGT': [],
                   'EPJ_EPJPNU': []}

feature_name = 'domain_inverse_distance'
updating = False
n_states = 4


vampnet_obj = VAMPNET_Initializer(feature_name,
                         lag=60,
                         start=550,
                         multimer=5,
                         symmetrize=False,
                         domain_exclusion=['MX', 'MA', 'MC', 'M4', 'M3'],
                         system_exclusion_dic=system_exclusion_dic,
#                         system_exclusion=[],
                         updating=updating,
                         feature_file='msm_features_01dt.pickle',
                         extra_feature_file='msm_features_new.pickle')


if updating:
    vampnet_obj.start_vampnet_analysis()
    with open('vampnet_obj_new.pickle', 'wb') as f:
        pickle.dump(vampnet_obj, f)
else:
    vampnet_obj = pickle.load(open('vampnet_obj_new.pickle', 'rb'))

dataset = vampnet_obj.dataset
n_val = int(len(dataset)*.1)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
loader_train = DataLoader(train_data, batch_size=20000, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)


class MultimerNet(nn.Module):
    def __init__(self, data_shape, multimer, n_states):
        super().__init__()
        self.data_shape = data_shape
        self.multimer = multimer
        self.n_states = n_states
        
        self.n_feat_per_sub = self.data_shape // self.multimer


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

try:
    data_shape = dataset[0][0].shape[0]
except NameError:
    data_load = np.load(pd.read_pickle('msm_features_new.pickle')[feature_name][0])
    data_shape = data_load[:, vampnet_obj.feed_feature_indice].shape[1]
pentamer_lobe = torch.nn.DataParallel(MultimerNet(data_shape=data_shape, multimer=5, n_states=n_states))


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
        r""" Performs a partial fit on data. This does not perform any batching.
        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.
        Returns
        -------
        self : VAMPNet
            Reference to self.
        """

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
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.
        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.
        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])

                val_aug = torch.concat([torch.roll(val, self.n_states * i, 1) for i in range(self.multimer)])
                val_t_aug = torch.concat([torch.roll(val_t, self.n_states * i, 1) for i in range(self.multimer)])
                score_value = vamp_score(val_aug, val_t_aug, method=self.score_method, mode=self.score_mode, epsilon=self.epsilon)
                return score_value


n_epochs = [20] * 10
vampnets = [VAMPNet_Multimer(multimer=5,
                            n_states=n_states,
                            lobe=deepcopy(pentamer_lobe).to(device=device),
                            score_method='VAMPE',
                            learning_rate=5e-3,
                            device=device) for i in range(len(n_epochs))]

for i, (vampnet, n_epoch) in enumerate(zip(vampnets, n_epochs)):  
    vampnet.fit(loader_train,
                        n_epochs=n_epoch,
                        validation_loader=loader_val,
                        progress=tqdm)
    torch.save(vampnet.lobe.module.state_dict(), f'./test_lobe_nstates_{n_states}_epoch_{n_epoch}_{i}.lobe')
    pickle.dump(vampnet, open(f'./vampnet_nstates_{n_states}_epoch_{n_epoch}_{i}.pickle', 'wb'))
    print(i)

for i, vampnet in enumerate(vampnets):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.loglog(*vampnet.train_scores.T, label='training')
    ax.loglog(*vampnet.validation_scores.T, label='validation')
    ax.set_xlabel('step')
    ax.set_ylabel('score')
    ax.legend()
    plt.savefig(f'test_vampnet_train_val_scores_{i}.png')


""" import torch.multiprocessing as mp
def train_vampnet(i, vampnet, n_epoch):
    vampnet.fit(loader_train,
                        n_epochs=n_epoch,
                        validation_loader=loader_val,
                        progress=tqdm)
    torch.save(vampnet.lobe.module.state_dict(), './test_lobe_epoch_' + str(n_epoch) + '_' + str(i) + '.lobe')
    return vampnet

with mp.get_context('spawn').Pool(4) as p:
    vampnets = []
    for i, epoch in enumerate(n_epochs):
        device = torch.device('cuda:' + str(i))
        vampnets.append(VAMPNet_Multimer(multimer=5,
                                    n_states=3,
                                    lobe=deepcopy(pentamer_threestate_lobe).to(device=device),
                                    score_method='VAMPE',
                                    learning_rate=5e-3,
                                    device=device))
    vampnet_pool = [p.apply_async(train_vampnet, args=(i, vampnet, n_epoch)) for i, (vampnet, n_epoch) in enumerate(zip(vampnets, n_epochs))]
    vampnets = [vampnet_worker.get() for vampnet_worker in vampnet_pool] """