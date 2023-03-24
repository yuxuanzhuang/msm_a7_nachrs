import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyemma
from deeptime.clustering import KMeans, RegularSpace
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.plots import plot_implied_timescales, plot_ck_test
from deeptime.util.validation import implied_timescales, ck_test
from joblib import Parallel, delayed
import pickle
from scipy.stats import pearsonr
from pydantic import BaseModel
from datetime import datetime
from copy import copy, deepcopy
from sklearn.neighbors import KNeighborsClassifier

from typing import List, Optional

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

        self.md_dataframe = deepcopy(md_dataframe)

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

        self.ck_test = {}
        self.rerun_msm = False


        self.md_dataframe.dataframe = self.md_dataframe.dataframe[self.md_dataframe.dataframe.system.isin(self.system_exclusion) == False].reset_index(drop=True)
        system_array = self.md_dataframe.dataframe.system.to_numpy()
        self.n_trajectories = len(np.unique(system_array))

        def fill_missing_values(system_array):
            diff_arr = (np.diff(system_array, prepend=0) != 0) & (np.diff(system_array, prepend=0) != 1)
            if all(diff_arr == False):
                return system_array
            start_index_update = np.arange(system_array.shape[0])[diff_arr][0]
            system_array[start_index_update:] = system_array[start_index_update:] -1
            return fill_missing_values(system_array)
        
        system_array = fill_missing_values(system_array)
        self.md_dataframe.dataframe.system = system_array
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
        if self.feature_input_list == []:
            raise ValueError('No feature selected yet, please use add_feature() first')
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
            
            if self.symmetrize:
                self.feature_trajectories.extend(get_symmetrized_data([feature_trajectory], self.multimer))
            else:
                self.feature_trajectories.append(feature_trajectory)
        self.data_collected = True

    def start_analysis(self):
        os.makedirs(self.filename, exist_ok=True)
        raise NotImplementedError("Should be overridden by subclass")
        if not self.data_collected:
            self.gather_feature_matrix()

    def clustering_with_deeptime(self, 
                                 n_clusters,
                                 meaningful_tic=None,
                                 updating=False,
                                 max_iter=1000):
        # if attr tica_output is None, then tica is not performed
        if not hasattr(self, 'tica_output'):
            raise ValueError('TICA output not available')
        self.n_clusters = n_clusters

        if meaningful_tic is None:
            meaningful_tic = np.arange(self.tica_output[0].shape[1])
        self.meaningful_tic = meaningful_tic
        print('Meaningful TICs are', meaningful_tic)
        
        self.tica_output_filter = [
            np.asarray(output)[
                :, meaningful_tic] for output in self.tica_output]

        if not (os.path.isfile(self.cluster_filename +
                '_deeptime.pickle')) or updating:
            print('Start new cluster analysis')
            self.rerun_msm = True
            self.kmean = KMeans(
                n_clusters=self.n_clusters,
                init_strategy='kmeans++',
                max_iter=max_iter,
                n_jobs=24,
                progress=tqdm)
            self.cluster = self.kmean.fit(self.tica_output_filter).fetch_model()
            self.cluster_dtrajs = [self.cluster.transform(tic_output_traj) for tic_output_traj in self.tica_output_filter]
            self.cluster_centers = self.cluster.cluster_centers
            self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

            os.makedirs(self.filename, exist_ok=True)
            pickle.dump(
                    self.cluster,
                    open(
                        self.cluster_filename +
                        '_deeptime.pickle',
                        'wb'))
        else:
            print('Loading old cluster analysis')

            self.cluster = pickle.load(
                open(
                    self.cluster_filename +
                    '_deeptime.pickle',
                    'rb'))
            self.cluster_dtrajs = [self.cluster.transform(tic_output_traj) for tic_output_traj in self.tica_output_filter]
            self.cluster_centers = self.cluster.cluster_centers
            self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)
            
    def clustering_with_pyemma(
            self, meaningful_tic, n_clusters, updating=False):
        self.clustering_with_deeptime(meaningful_tic, n_clusters, updating=updating)
        return None
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

    def assigning_cluster(self,
                          cluster_dtrajs,
                          n_clusters=None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        self.cluster_dtrajs = cluster_dtrajs
        self.meaningful_tic = 'all'
        self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

    @staticmethod
    def bayesian_msm_from_traj(cluster_dtrajs, lagtime, n_samples):
            counts = TransitionCountEstimator(lagtime=lagtime, count_mode='effective').fit_fetch(cluster_dtrajs)
            return BayesianMSM(n_samples=n_samples).fit_fetch(counts)
    
    def get_its(self, cluster='deeptime', lag_max=200, n_samples=1000, n_jobs=10, updating=False):

        if cluster == 'deeptime':
            if not (os.path.isfile(self.cluster_filename +
                    f'_deeptime_its_{n_samples}.pickle')) or updating or self.rerun_msm:
                print('Start new ITS analysis')
                lagtimes = np.linspace(1, lag_max / self.dt, 10).astype(int)

                if n_jobs != 1:
                    with tqdm_joblib(tqdm(desc="ITS", total=10)) as progress_bar:
                        models = Parallel(n_jobs=n_jobs)(delayed(self.bayesian_msm_from_traj)(self.cluster_dtrajs,
                                                                    lagtime, n_samples) for lagtime in lagtimes)
                else:
                    models = []
                    for lagtime in tqdm(lagtimes, desc='lagtime', total=len(lagtimes)):
                        counts = TransitionCountEstimator(lagtime=lagtime, count_mode='effective').fit_fetch(self.cluster_dtrajs)
                        models.append(BayesianMSM(n_samples=n_samples).fit_fetch(counts))
                
                print('Keep ITS analysis')
                self.its_models = models
                self.its = implied_timescales(models)

                pickle.dump(
                        self.its,
                        open(
                            self.cluster_filename +
                            f'_deeptime_its_{n_samples}.pickle',
                            'wb'))
                pickle.dump(
                        self.its_models,
                        open(
                            self.cluster_filename +
                            f'_deeptime_its_models_{n_samples}.pickle',
                            'wb'))
            else:
                print('Loading old ITS analysis')

                self.its = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_its_{n_samples}.pickle',
                        'rb'))
                self.its_models = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_its_models_{n_samples}.pickle',
                        'rb'))
            
        elif cluster == 'pyemma':
            self.its = pyemma.msm.its(
                self.cluster_dtrajs,
                lags=int(
                    lag_max / self.dt),
                nits=10,
                errors='bayes',
                only_timescales=True)
        else:
            raise ValueError('Cluster method not recognized')

        return self.its
    
    def plot_its(self, n_its=10):
        fig, ax = plt.subplots(figsize=(18, 10))
        plot_implied_timescales(self.its, n_its=n_its, ax=ax)
        ax.set_yscale('log')
        ax.set_title('Implied timescales')
        ax.set_xlabel('lag time (steps)')
        ax.set_ylabel('timescale (steps)')
        plt.show()

        
    def get_ck_test(self, n_states, lag, mlags=6, n_jobs=6, n_samples=20, updating=False):
        if not updating and not self.rerun_msm and (os.path.isfile(self.cluster_filename +
                                                                   f'_deeptime_cktest.pickle')):
            print('Loading old CK test')
            self.ck_test = pickle.load(
                open(
                    self.cluster_filename +
                    f'_deeptime_cktest.pickle',
                    'rb'))
            
        if (n_states, lag, mlags) not in self.ck_test or updating:            
            print('CK models building')
            model =  BayesianMSM(n_samples=n_samples).fit_fetch(TransitionCountEstimator(lagtime=lag,
                                                        count_mode='effective').fit_fetch(self.cluster_dtrajs))
            lagtimes = np.arange(1, mlags+1) * lag
            print('Estimating lagtimes', lagtimes)

            if n_jobs != 1:
                with tqdm_joblib(tqdm(desc="ITS", total=len(lagtimes))) as progress_bar:
                    test_models = Parallel(n_jobs=n_jobs)(delayed(self.bayesian_msm_from_traj)(self.cluster_dtrajs,
                                                                lagtime, n_samples) for lagtime in lagtimes)
            else:
                test_models = []
                for lagtime in tqdm(lagtimes, desc='lagtime', total=len(lagtimes)):
                    counts = TransitionCountEstimator(lagtime=lagtime, count_mode='effective').fit_fetch(self.cluster_dtrajs)
                    test_models.append(BayesianMSM(n_samples=n_samples).fit_fetch(counts))
            print('Start CK test')
            self.ck_test[n_states, lag, mlags] = {
                    'model': model,
                    'ck_test': model.ck_test(test_models, n_states, progress=tqdm),
                    'models': test_models
            }
            pickle.dump(
                    self.ck_test,
                    open(
                        self.cluster_filename +
                        f'_deeptime_cktest.pickle',
                        'wb'))
            
        return plot_ck_test(self.ck_test[n_states, lag, mlags]['ck_test'])


    def get_maximum_likelihood_msm(self, lag, cluster='deeptime', updating=False):
        self.msm_lag = lag
        if cluster == 'deeptime':
            if not (os.path.isfile(self.cluster_filename +
                    f'_deeptime_max_msm_{lag}.pickle')) or updating or self.rerun_msm:
                print('Start new MSM analysis')
                self.msm = MaximumLikelihoodMSM(
                                reversible=True,
                                stationary_distribution_constraint=None)
                self.msm.fit(self.cluster_dtrajs, lagtime=lag)
                self.msm_model = self.msm.fetch_model()
                pickle.dump(
                        self.msm,
                        open(
                            self.cluster_filename +
                            f'_deeptime_max_msm_{lag}.pickle',
                            'wb'))
                pickle.dump(
                        self.msm_model,
                        open(
                            self.cluster_filename +
                            f'_deeptime_max_msm_model_{lag}.pickle',
                            'wb'))
            else:
                print('Loading old MSM analysis')
                self.msm = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_max_msm_{lag}.pickle',
                        'rb'))
                self.msm_model = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_max_msm_model_{lag}.pickle',
                        'rb'))
            self.trajectory_weights = self.msm_model.compute_trajectory_weights(self.cluster_dtrajs)

                
        elif cluster == 'pyemma':
            self.msm_model = pyemma.msm.bayesian_markov_model(
                self.cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')
        
        return self.msm_model

    def get_bayesian_msm(self, lag, n_samples=100, cluster='deeptime', updating=False):
        self.msm_lag = lag
        if cluster == 'deeptime':
            if not (os.path.isfile(self.cluster_filename +
                    f'_deeptime_bayesian_msm_{lag}.pickle')) or updating or self.rerun_msm:
                print('Start new MSM analysis')
                self.counts = TransitionCountEstimator(lagtime=lag,
                                                       count_mode='effective').fit_fetch(self.cluster_dtrajs)
                self.msm = BayesianMSM(n_samples=n_samples).fit(self.counts)

                self.msm_model = self.msm.fetch_model()

                from deeptime.markov.tools.analysis import stationary_distribution

                pi_samples = []
                traj_weights_samples = []
                for sample in self.msm_model.samples:
                    pi_samples.append(stationary_distribution(sample.transition_matrix))
                    traj_weights_samples.append(sample.compute_trajectory_weights(self.cluster_dtrajs))

                self.pi_samples = np.array(pi_samples, dtype=object)
                self.traj_weights_samples = np.array(traj_weights_samples, dtype=object)

                self.stationary_distribution = np.mean(self.pi_samples, axis=0)
                self.pi = self.stationary_distribution
                self.trajectory_weights = np.mean(self.traj_weights_samples, axis=0)


                pickle.dump(
                        self.counts,
                        open(
                            self.cluster_filename +
                            f'_deeptime_bayesian_counts_{lag}.pickle',
                            'wb'))
                pickle.dump(
                        self.msm,
                        open(
                            self.cluster_filename +
                            f'_deeptime_bayesian_msm_{lag}.pickle',
                            'wb'))
                pickle.dump(
                        self.msm_model,
                        open(
                            self.cluster_filename +
                            f'_deeptime_bayesian_msm_model_{lag}.pickle',
                            'wb'))
                            
            else:
                print('Loading old MSM analysis')
                self.counts = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_bayesian_counts_{lag}.pickle',
                        'rb'))
                self.msm = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_bayesian_msm_{lag}.pickle',
                        'rb'))
                self.msm_model = pickle.load(
                    open(
                        self.cluster_filename +
                        f'_deeptime_bayesian_msm_model_{lag}.pickle',
                        'rb'))

                from deeptime.markov.tools.analysis import stationary_distribution

                pi_samples = []
                traj_weights_samples = []
                for sample in self.msm_model.samples:
                    pi_samples.append(stationary_distribution(sample.transition_matrix))
                    traj_weights_samples.append(sample.compute_trajectory_weights(self.cluster_dtrajs))

                self.pi_samples = np.array(pi_samples, dtype=object)
                self.traj_weights_samples = np.array(traj_weights_samples, dtype=object)

                self.stationary_distribution = np.mean(self.pi_samples, axis=0)
                self.pi = self.stationary_distribution
                self.trajectory_weights = np.mean(self.traj_weights_samples, axis=0)

        elif cluster == 'pyemma':
            self.msm_model = pyemma.msm.bayesian_markov_model(
                self.cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')
        
        return self.msm_model


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


class MSMMetaData(BaseModel):
    create_time: Optional[datetime] = None
    id: int = 0
    name = 'MSM'
    lag: int
    start: int
    end: int
    multimer: int
    symmetrize: bool
    system_exclusion: Optional[List[int]] = []
    interval: int
    prefix: Optional[str] = None
    feature_input_info_list: Optional[List[str]] = []