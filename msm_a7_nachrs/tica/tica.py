import os
from msm_a7_nachrs.msm.MSM_a7 import MSMInitializer
import numpy as np
import pandas as pd
import pyemma
from deeptime.decomposition import TICA
import pickle
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

import gc
import itertools
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align, pca, rms
from tqdm import tqdm

import dask.dataframe as dd

from ..util.utils import *
from ..datafiles import BGT, EPJ, EPJPNU
from ..msm.MSM_a7 import MSMInitializer
from ..util.dataloader import MultimerTrajectoriesDataset, get_symmetrized_data
from .sym_tica import SymTICA


pathways = [
    'BGT_EPJPNU',
    'BGT_EPJ',
    'EPJPNU_BGT',
    'EPJPNU_EPJ',
    'EPJ_BGT',
    'EPJ_EPJPNU'
]

pathway_ind_dic = {
    'BGT_EPJPNU': 1,
    'BGT_EPJ': 2,
    'EPJPNU_BGT': 3,
    'EPJPNU_EPJ': 4,
    'EPJ_BGT': 5,
    'EPJ_EPJPNU': 6
}

system_exclusion_dic = {'BGT_EPJPNU': [],
                        'BGT_EPJ': [],
                        'EPJPNU_BGT': [],
                        'EPJPNU_EPJ': [],
                        'EPJ_BGT': [],
                        'EPJ_EPJPNU': []}

class TICAInitializer(MSMInitializer):
    prefix = "tica"

    def start_analysis(self, block_size=10):
        os.makedirs(self.filename, exist_ok=True)
        if (not os.path.isfile(self.filename + 'tica.pickle')) or self.updating:
            print('Start new TICA analysis')
            if self.in_memory:
                if not self.data_collected:
                    self.gather_feature_matrix()

                self.tica = TICA(var_cutoff=0.8, lagtime=self.lag)
                self.tica.fit(self.feature_trajectories)
                pickle.dump(self.tica,
                open(
                    self.filename +
                    'tica.pickle',
                    'wb'))
                self.tica_output = [self.tica.transform(feature_traj) for feature_traj in self.feature_trajectories]

            else:
                self.tica = TICA(var_cutoff=0.8, lagtime=self.lag)
                self.partial_fit_tica(block_size=block_size)
                _ = self.tica.fetch_model()
                pickle.dump(self.tica,
                open(
                    self.filename +
                    'tica.pickle',
                    'wb'))
                self.tica_output = self.transform_feature_trajectories(self.md_dataframe,
                                        start=self.start)

            self.tica_concatenated = np.concatenate(self.tica_output)

            pickle.dump(
                self.tica_output,
                open(
                    self.filename +
                    'output.pickle',
                    'wb'))
#            if self.in_memory:
#                self.dump_feature_trajectories()
            gc.collect()

        else:
            print('Load old TICA results')
            self.tica = pickle.load(
                open(self.filename + 'tica.pickle', 'rb'))
            self.tica_output = pickle.load(
                open(self.filename + 'output.pickle', 'rb'))
            self.tica_concatenated = np.concatenate(self.tica_output)

    def partial_fit_tica(self, block_size=1):
        """
        Fit TICA to a subset of the data."""
        feature_df = self.md_dataframe.get_feature(self.feature_input_list,
                                in_memory=False)
        feature_trajectories = []
        for ind, (system, row) in tqdm(enumerate(feature_df.iterrows()), total=feature_df.shape[0]):
            if system not in self.system_exclusion:
                feature_trajectory = []
                for feat_loc, indice, feat_type in zip(row[self.feature_input_list].values,
                self.feature_input_indice_list, self.feature_type_list):
                    raw_data = np.load(feat_loc, allow_pickle=True)
                    raw_data = raw_data.reshape(raw_data.shape[0], -1)[self.start:, indice]
                    if feat_type == 'global':
                        # repeat five times
                        raw_data = np.repeat(raw_data, 5, axis=1).reshape(raw_data.shape[0], -1, 5).transpose(0, 2, 1)
                    else:
                        raw_data = raw_data.reshape(raw_data.shape[0], 5, -1)

                    feature_trajectory.append(raw_data)

                feature_trajectory = np.concatenate(feature_trajectory, axis=2).reshape(raw_data.shape[0], -1)
                if (ind+1) % block_size == 0:
                    feature_trajectories.append(feature_trajectory)
                    dataset = MultimerTrajectoriesDataset.from_numpy(
                        self.lag, self.multimer, feature_trajectories)
                    self.tica.partial_fit(dataset)
                    feature_trajectories = []
                else:
                    feature_trajectories.append(feature_trajectory)

        # fit the remaining data
        if len(feature_trajectories) > 0:
            dataset = MultimerTrajectoriesDataset.from_numpy(
                self.lag, self.multimer, feature_trajectories)
            self.tica.partial_fit(dataset)

    def transform_feature_trajectories(self, md_dataframe, start=0, end=-1):
        """
        Map new feature trajectories to the TICA space.
        """
        if end == -1:
            end = md_dataframe.dataframe.shape[0]
            
        mapped_feature_trajectories = []
        feature_df = md_dataframe.get_feature(self.feature_input_list,
                    in_memory=False)
        for system, row in tqdm(feature_df.iterrows(), total=feature_df.shape[0]):
            feature_trajectory = []
            for feat_loc, indice, feat_type in zip(row[self.feature_input_list].values,
                                                   self.feature_input_indice_list,
                                                   self.feature_type_list):
                raw_data = np.load(feat_loc, allow_pickle=True)
                raw_data = raw_data.reshape(raw_data.shape[0], -1)[start:end, indice]
                if feat_type == 'global':
                    # repeat five times
                    raw_data = np.repeat(raw_data, 5, axis=1).reshape(raw_data.shape[0], -1, 5).transpose(0, 2, 1)
                else:
                    raw_data = raw_data.reshape(raw_data.shape[0], 5, -1)

                feature_trajectory.append(raw_data)

            feature_trajectory = np.concatenate(feature_trajectory, axis=2).reshape(raw_data.shape[0], -1)
            feature_trajectories = get_symmetrized_data([feature_trajectory], self.multimer)
            for single_traj in feature_trajectories:
                mapped_feature_trajectories.append(self.tica.transform(single_traj))

        return mapped_feature_trajectories


    def get_correlation(self, feature, max_tic=3, stride=1):
        """
        Get the correlation between the feature and the TICA components.
        """
        feature_dataframe = self.md_dataframe.get_feature([feature])

        feature_data_concat = feature_dataframe[feature_dataframe.traj_time >= self.start \
        * self.dt * 1000].iloc[:, 4:].values
        tica_concat = np.concatenate([tica_traj[:, :max_tic] for tica_traj in \
        self.tica_output[::5]], axis=0)
        test_feature_TIC_correlation = np.zeros((feature_data_concat.shape[1], max_tic))
        for i in tqdm(range(feature_data_concat.shape[1]), total=feature_data_concat.shape[1]):
            for j in range(max_tic):
                test_feature_TIC_correlation[i, j] = pearsonr(
                    feature_data_concat[::stride, i], tica_concat[::stride, j])[0]

        test_feature_TIC_correlation_df = pd.DataFrame(
            test_feature_TIC_correlation,
            columns=['TIC' + str(i) for i in range(1, max_tic + 1)],
            index=feature_dataframe.columns[4:])

        del feature_dataframe
        del feature_data_concat
        gc.collect()

        return test_feature_TIC_correlation_df


    def get_mutual_information(self, feature, max_tic=3, stride=10):
        """
        Get the mutual information between the feature and the TICA components.
        """
        feature_dataframe = self.md_dataframe.get_feature([feature])

        feature_data_concat = feature_dataframe[feature_dataframe.traj_time >= self.start \
        * self.dt * 1000].iloc[:, 4:].values
        tica_concat = np.concatenate([tica_traj[:, :max_tic] for tica_traj in \
        self.tica_output[::5]], axis=0)
        test_feature_TIC_MI = np.zeros((feature_data_concat.shape[1], max_tic))
        for j in tqdm(range(max_tic), total=max_tic):
            mi = mutual_info_regression(feature_data_concat[::stride, :],
                                        tica_concat[::stride, j])
            test_feature_TIC_MI[:, j] = mi

        test_feature_TIC_MI_df = pd.DataFrame(
            test_feature_TIC_MI,
            columns=['TIC' + str(i) for i in range(1, max_tic + 1)],
            index=feature_dataframe.columns[4:])

        del feature_dataframe
        del feature_data_concat
        gc.collect()

        return test_feature_TIC_MI_df


class SymTICAInitializer(TICAInitializer):
    prefix = "sym_tica"

    def start_analysis(self, block_size=10):
        os.makedirs(self.filename, exist_ok=True)
        if (not os.path.isfile(self.filename + 'sym_tica.pickle')) or self.updating:
            print('Start new Sym TICA analysis')
            if self.in_memory:
                if not self.data_collected:
                    self.gather_feature_matrix()

                self.tica = SymTICA(symmetry_fold=self.multimer,
                            var_cutoff=0.8, lagtime=self.lag)
                self.tica.fit(self.feature_trajectories)
                pickle.dump(self.tica,
                open(
                    self.filename +
                    'sym_tica.pickle',
                    'wb'))
                self.tica_output = [self.tica.transform(feature_traj) for feature_traj in self.feature_trajectories]

            else:
                self.tica = SymTICA(symmetry_fold=self.multimer,
                            var_cutoff=0.8, lagtime=self.lag)
                self.partial_fit_tica(block_size=block_size)
                _ = self.tica.fetch_model()
                pickle.dump(self.tica,
                open(
                    self.filename +
                    'sym_tica.pickle',
                    'wb'))
                self.tica_output = self.transform_feature_trajectories(self.md_dataframe,
                                        start=self.start)

            self.tica_concatenated = np.concatenate(self.tica_output)

            pickle.dump(
                self.tica_output,
                open(
                    self.filename +
                    'output.pickle',
                    'wb'))
#            if self.in_memory:
#                self.dump_feature_trajectories()
            gc.collect()

        else:
            print('Load old sym TICA results')
            self.tica = pickle.load(
                open(self.filename + 'sym_tica.pickle', 'rb'))
            self.tica_output = pickle.load(
                open(self.filename + 'output.pickle', 'rb'))
            self.tica_concatenated = np.concatenate(self.tica_output)