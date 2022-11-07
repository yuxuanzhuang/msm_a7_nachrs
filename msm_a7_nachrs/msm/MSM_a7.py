import os
import numpy as np
import pandas as pd
import pyemma
import pickle
from scipy.stats import pearsonr

import gc
import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm
from ..util.utils import *
from ..util.dataloader import MultimerTrajectoriesDataset, get_symmetrized_data

from ..datafiles import BGT, EPJ, EPJPNU


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

#  exclused seeds from MSM
system_exclusion_dic = {'BGT_EPJPNU': [],
                        'BGT_EPJ': [],
                        'EPJPNU_BGT': [],
                        'EPJPNU_EPJ': [],
                        'EPJ_BGT': [],
                        'EPJ_EPJPNU': []}


def score_cv(data, dim, lag, number_of_splits=10, validation_fraction=0.5):
    """Compute a cross-validated VAMP2 score.

    We randomly split the list of independent trajectories into
    a training and a validation set, compute the VAMP2 score,
    and repeat this process several times.

    Parameters
    ----------
    data : list of numpy.ndarrays
        The input data.
    dim : int
        Number of processes to score; equivalent to the dimension
        after projecting the data with VAMP2.
    lag : int
        Lag time for the VAMP2 scoring.
    number_of_splits : int, optional, default=10
        How often do we repeat the splitting and score calculation.
    validation_fraction : int, optional, default=0.5
        Fraction of trajectories which should go into the validation
        set during a split.
    """
    # we temporarily suppress very short-lived progress bars
    with pyemma.util.contexts.settings(show_progress_bars=True):
        nval = int(len(data) * validation_fraction)
        scores = np.zeros(number_of_splits)
        for n in range(number_of_splits):
            ival = np.random.choice(len(data), size=nval, replace=False)
            vamp = pyemma.coordinates.vamp(
                [d for i, d in enumerate(data) if i not in ival], lag=lag, dim=dim)
            scores[n] = vamp.score(
                [d for i, d in enumerate(data) if i in ival])
    return scores


class MSMInitializer(object):
    prefix = "msm"

    def __init__(self,
                 md_dataframe,
                 lag,
                 start=0,
                 end=-1,
                 multimer=5,
                 symmetrize=True,
                 updating=True,
                 dumping=False,
                 system_exclusion=[],
                 interval=1,
                 prefix=None,
                 in_memory=True):

        #TODO: Add deepcopy to datarfame
        self.md_dataframe = md_dataframe

        # lag for # of frames
        self.lag = lag
        self.start = start
        self.end = end
        self.multimer = multimer
        self.symmetrize = symmetrize
        self.updating = updating
        self.dumping = dumping

        self.system_exclusion = system_exclusion

        self.interval = interval
        if prefix!=None:
            self.prefix=prefix
        self.in_memory = in_memory
        self.data_collected = False


        self.md_dataframe.dataframe = self.md_dataframe.dataframe[self.md_dataframe.dataframe.system.isin(self.system_exclusion) == False].reset_index()
        self.md_data = self.md_dataframe.dataframe
        # dt: ns
        self.dt = (
            self.md_data.traj_time[1] - self.md_data.traj_time[0]) / 1000 * self.interval
        print(f'lag time is {self.lag * self.dt} ns')
        print(f'start time is {self.start * self.dt} ns')
        if self.end != -1:
            print(f'end time is {self.end * self.dt} ns')

        self.feature_input_list = []
        self.feature_input_info_list = []
        self.feature_input_indice_list = []
        self.feature_type_list = []

        os.makedirs(self.filename, exist_ok=True)


    def add_feature(self, feature_selected, excluded_indices=[], feat_type='subunit'):
        if feature_selected not in self.md_dataframe.analysis_list:
            raise ValueError(
                f'feature selection {feature_selected} not available\n'
                f'Available features are {self.md_dataframe.analysis_list}')
        self.feature_input_list.append(feature_selected)

        feature_info = np.load(
            self.md_dataframe.analysis_results.filename + feature_selected + '_feature_info.npy'
            )

        self.feature_input_info_list.append(np.delete(feature_info, excluded_indices))
        self.feature_input_indice_list.append(np.delete(np.arange(len(feature_info)), excluded_indices))
        self.feature_type_list.append(feat_type)
        print(
            f'added feature selection {feature_selected} type: {feat_type}, # of features: {len(self.feature_input_info_list[-1])}')
    
    def gather_feature_matrix(self):
        """load feature matrix into memory"""
        self.feature_trajectories = []
        feature_df = self.md_dataframe.get_feature(self.feature_input_list,
                    in_memory=False)
        for system, row in tqdm(feature_df.iterrows(), total=feature_df.shape[0]):
            feature_trajectory = []
            for feat_loc, indice, feat_type in zip(row[self.feature_input_list].values,
                                                   self.feature_input_indice_list,
                                                   self.feature_type_list):
                raw_data = np.load(feat_loc, allow_pickle=True)
                if self.end == -1:
                    end = raw_data.shape[0]
                else:
                    end = self.end
                raw_data = raw_data.reshape(raw_data.shape[0], -1)[self.start:end, indice]
                if feat_type == 'global':
                    # repeat five times
                    raw_data = np.repeat(raw_data, 5, axis=1).reshape(raw_data.shape[0], -1, 5).transpose(0, 2, 1)
                else:
                    raw_data = raw_data.reshape(raw_data.shape[0], 5, -1)

                feature_trajectory.append(raw_data)
            feature_trajectory = np.concatenate(feature_trajectory, axis=2).reshape(raw_data.shape[0], -1)
            self.feature_trajectories.extend(get_symmetrized_data([feature_trajectory], self.multimer))
        self.data_collected = True
                

    def dump_feature_trajectories(self):
        if self.data_collected:
            if self.dumping:
                for ind, feature_matrix in enumerate(
                        self.feature_trajectories):
                    np.save(
                        self.filename +
                        f'feature_traj_{ind}.npy',
                        feature_matrix)
            print('Feature matrix saved')
            self.data_collected = False

        del self.feature_trajectories
        gc.collect()


    def get_feature_trajectories(self):
        if self.data_collected:
            return self.feature_trajectories
        else:
            self.feature_trajectories = []
            for ind in range(self.n_trajectories):
                self.feature_trajectories.append(
                    np.load(
                        self.filename +
                        f'feature_traj_{ind}.npy',
                        allow_pickle=True))

    def start_analysis(self):
        os.makedirs(self.filename, exist_ok=True)
        raise NotImplementedError("Should be overridden by subclass")
        if not self.data_collected:
            self.gather_feature_matrix()

    def clustering_with_pyemma(
            self, meaningful_tic, n_clusters, updating=False):
        self.n_clusters = n_clusters
        self.meaningful_tic = meaningful_tic

        self.tica_output_filter = [
            np.asarray(output)[
                :, meaningful_tic] for output in self.tica_output]

        if not (os.path.isfile(self.cluster_filename +
                '_pyemma.pickle')) or updating:
            print('Start new cluster analysis')

            self.cluster = pyemma.coordinates.cluster_kmeans(self.tica_output_filter,
                                                             k=self.n_clusters,
                                                             max_iter=100,
                                                             stride=100)
            self.cluster_dtrajs = self.cluster.dtrajs

            if self.dumping:
                pickle.dump(
                    self.cluster,
                    open(
                        self.cluster_filename +
                        '_pyemma.pickle',
                        'wb'))
                pickle.dump(
                    self.cluster_dtrajs,
                    open(
                        self.cluster_filename +
                        '_output_pyemma.pickle',
                        'wb'))

        else:
            print('Load old cluster results')
            self.cluster = pickle.load(
                open(
                    self.cluster_filename +
                    '_pyemma.pickle',
                    'rb'))
            self.cluster_dtrajs = pickle.load(
                open(
                    self.cluster_filename +
                    '_output_pyemma.pickle',
                    'rb'))

        self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

    def get_its(self, cluster='pyemma'):
        if cluster == 'dask':
            self.its = pyemma.msm.its(
                self.d_cluster_dtrajs,
                lags=int(
                    200 / self.dt),
                nits=10,
                errors='bayes',
                only_timescales=True)
        elif cluster == 'pyemma':
            self.its = pyemma.msm.its(
                self.cluster_dtrajs,
                lags=int(
                    200 / self.dt),
                nits=10,
                errors='bayes',
                only_timescales=True)

        return self.its

    def get_msm(self, lag, cluster='pyemma'):
        self.msm_lag = lag

        if cluster == 'dask':
            self.msm = pyemma.msm.bayesian_markov_model(
                self.d_cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')
        elif cluster == 'pyemma':
            self.msm = pyemma.msm.bayesian_markov_model(
                self.cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')

        return self.msm


    def get_correlation(self, feature):
        md_data = pd.read_pickle(self.feature_file)

        raw_data = np.concatenate([np.load(location, allow_pickle=True)
                                   for location, df in md_data.groupby(feature, sort=False)])

        md_data = md_data[md_data.pathway.isin(self.pathways)]

        feature_data = []

        for sys, df in md_data.groupby(['system']):
            if sys not in self.system_exclusion:
                if self.end == -1:
                    end = df.index[-1]
                else:
                    end = df[df.frame < self.end].index[-1]
                sys_data = raw_data[df.index[0] +
                                    self.start:end + 1, :]
                total_shape = sys_data.shape[1]
                per_shape = int(total_shape / 5)

                #  symmetrization of features
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0], 5, per_shape),
                                            0, axis=1).reshape(sys_data.shape[0], total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0], 5, per_shape),
                                            1, axis=1).reshape(sys_data.shape[0], total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0], 5, per_shape),
                                            2, axis=1).reshape(sys_data.shape[0], total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0], 5, per_shape),
                                            3, axis=1).reshape(sys_data.shape[0], total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0], 5, per_shape),
                                            4, axis=1).reshape(sys_data.shape[0], total_shape))

        feature_data_concat = np.concatenate(feature_data)

        max_tic = 20
        test_feature_TIC_correlation = np.zeros(
            (feature_data_concat.shape[1], max_tic))

        for i in range(feature_data_concat.shape[1]):
            for j in range(max_tic):
                test_feature_TIC_correlation[i, j] = pearsonr(
                    feature_data_concat[:, i],
                    self.tica_concatenated[:, j])[0]

        feature_info = np.load(
            './analysis_results/' +
            feature +
            '_feature_info.npy')

        del feature_data
        del feature_data_concat
        gc.collect()

        return test_feature_TIC_correlation, feature_info

    @property
    def filename(self):
        if self.feature_input_list == []:
            feature_list_str = ''
        else:
            feature_list_str = '__'.join([f'{feature}_{len(feat_n)}' for feature, feat_n in zip(self.feature_input_list, self.feature_input_info_list)])

        if self.end == -1:
            return f'{self.md_dataframe.filename}/msmfile/{self.prefix}/{self.lag}/{self.start}/{feature_list_str}/'
        else:
            return f'{self.md_dataframe.filename}/msmfile/{self.prefix}/{self.lag}/{self.start}_{self.end}/{feature_list_str}/'

    @property
    def cluster_filename(self):
        return self.filename + 'cluster' + \
            str(self.n_clusters) + '_tic' + \
            '_'.join([str(m_tic) for m_tic in self.meaningful_tic]) + '_'
