from MSM_a7 import *

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

class MultimerTrajectoriesDataset(TrajectoriesDataset):
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


class VAMPNET_Initializer(object):
    def __init__(self, vamp_file, updating=False, lag=20, start=0,
                 multimer=5,
                 symmetrize=False,
                 pathways=['BGT_EPJPNU',
                           'BGT_EPJ',
                           'EPJPNU_BGT',
                           'EPJPNU_EPJ',
                           'EPJ_BGT',
                           'EPJ_EPJPNU'],
                 system_exclusion_dic = system_exclusion_dic,
                 domain_exclusion = [],
                 feature_file = 'msm_features.pickle',
                 extra_feature_file = None,
                 extra_system_exclusion_dic = new_system_exclusion_dic):
        self.lag = lag
        self.vamp_file = vamp_file
        self.start = start
        self.multimer = multimer
        self.symmetrize = symmetrize
        self.pathways = pathways
        self.updating = updating
        self.feature_file = feature_file
        
        self.system_exclusion_dic = system_exclusion_dic
        
        self.md_data = pd.read_pickle(self.feature_file)
        self.md_data = self.md_data.applymap(lambda x: x.replace('/mnt/cephfs/projects/2020100800_alpha7_nachrs/MSM',
                           '/mnt/cephfs/projects/2021072000_nachrs_a7_msm/a7_nachrs') if isinstance(x, str) else x)
        self.dt = (self.md_data.traj_time[1] - self.md_data.traj_time[0]) / 1000
        system_exclusion = []
        offset = 0
        for ind, pathway in enumerate(self.pathways):
            for exclu_seed in system_exclusion_dic[pathway]:
                exclu_system_ind = exclu_seed + ind * 24 + offset
                system_exclusion.append(exclu_system_ind)
                
            if pathway == 'EPJPNU_BGT':
                offset = 1
    
        self.extra_feature_file = extra_feature_file
        self.extra_system_exclusion_dic = extra_system_exclusion_dic
        
        
        if self.extra_feature_file != None:
            extra_system_exclusion = []

            self.md_data_extra = pd.read_pickle(self.extra_feature_file)
            
            for ind, pathway in enumerate(self.pathways):
                for exclu_seed in self.extra_system_exclusion_dic[pathway]:
                    exclu_system_ind = exclu_seed + ind * 24 + offset
                    extra_system_exclusion.append(exclu_system_ind)

                if pathway == 'EPJPNU_BGT':
                    offset = 1
        
        self.system_exclusion = system_exclusion
        self.extra_system_exclusion = extra_system_exclusion

        self.domain_exclusion = domain_exclusion
        
        self.feature_info = np.load('./analysis_results/' + self.vamp_file + '_feature_info.npy')
        
        self.feed_feature_indice = []
        for ind, feat in enumerate(self.feature_info):
            if not any(exl_feat in feat for exl_feat in self.domain_exclusion):
                self.feed_feature_indice.append(ind)
                
        self.feature_info = self.feature_info[self.feed_feature_indice]
        
        
    def start_vampnet_analysis(self):
        if (not os.path.isfile(self.filename  + 'vampnet.pyemma')) or self.updating:
            print('Start new VAMPNET analysis')

            self.start_analysis()

        else:
            print('Load old VAMPNET results')
            pass
        
    def start_analysis(self):
        feature_data = []

        raw_data = np.concatenate([np.load(location, allow_pickle=True)
                        for location, df in self.md_data.groupby(self.vamp_file, sort=False)])
        md_data = self.md_data[self.md_data.pathway.isin(self.pathways)]
        
        print('Feed n.frames:' + str(md_data.shape[0]))
        
        
        self.pathway_seed_start = [0]
        old_pathway = self.pathways[0]
        sys_index = 1
        for sys, df in md_data.groupby(['system']):
            if sys not in self.system_exclusion:
                sys_data = raw_data[df.index[0] + self.start:df.index[-1]+1, self.feed_feature_indice]
                
                total_shape = sys_data.shape[1]
                per_shape = int(total_shape / 5)

                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            0, axis=1).reshape(sys_data.shape[0],total_shape))
                if self.symmetrize:
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                1, axis=1).reshape(sys_data.shape[0],total_shape))
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                2, axis=1).reshape(sys_data.shape[0],total_shape))
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                3, axis=1).reshape(sys_data.shape[0],total_shape))
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                4, axis=1).reshape(sys_data.shape[0],total_shape))                
                if df.pathway.iloc[0] != old_pathway:
                    self.pathway_seed_start.append(sys_index)
                old_pathway = df.pathway.iloc[0]
                sys_index += 1
            
            
        if self.extra_feature_file != None:
            raw_data = np.concatenate([np.load(location, allow_pickle=True)
                        for location, df in self.md_data_extra.groupby(self.vamp_file, sort=False)])
            md_data_extra = self.md_data_extra[self.md_data_extra.pathway.isin(self.pathways)]
            print('Feed extra n.frames:' + str(md_data_extra.shape[0]))

            self.pathway_seed_start_extra = [0]
            old_pathway = self.pathways[0]
            sys_index = 1
            for sys, df in md_data_extra.groupby(['system']):
                if sys not in self.extra_system_exclusion:
                    sys_data = raw_data[df.index[0] + self.start:df.index[-1]+1, self.feed_feature_indice]
                    
                    total_shape = sys_data.shape[1]
                    per_shape = int(total_shape / 5)

                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                0, axis=1).reshape(sys_data.shape[0],total_shape))
                    if self.symmetrize:
                        feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                    1, axis=1).reshape(sys_data.shape[0],total_shape))
                        feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                    2, axis=1).reshape(sys_data.shape[0],total_shape))
                        feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                    3, axis=1).reshape(sys_data.shape[0],total_shape))
                        feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                    4, axis=1).reshape(sys_data.shape[0],total_shape))
                    if df.pathway.iloc[0] != old_pathway:
                        self.pathway_seed_start_extra.append(sys_index)
                    old_pathway = df.pathway.iloc[0]
                    sys_index += 1
        
        # self.feature_data = feature_data

        self.dataset = MultimerTrajectoriesDataset.from_numpy(lagtime=self.lag,
                                                 multimer=self.multimer,
                                                 data=[traj.astype(np.float32) for traj in feature_data])
        del feature_data
        gc.collect()
        
    @property
    def filename(self):
        return './msmfile/' + self.vamp_file + '_lag' + str(self.lag) + '_start' + str(self.start) + '_pathway' + ''.join([str(pathway_ind_dic[pathway]) for pathway in self.pathways]) + '_'
    
    @property
    def cluster_filename(self):
        return self.filename + '_cluster' + str(self.n_clusters) + '_tic' + '_'.join([str(m_tic) for m_tic in self.meaningful_tic]) + '_'

