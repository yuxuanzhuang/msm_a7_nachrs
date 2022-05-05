import os
from msm_a7_nachrs.msm.MSM_a7 import MSMInitializer
import numpy as np
import pandas as pd
import pyemma
import pickle
from scipy.stats import pearsonr

import gc
import itertools
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align, pca, rms

import dask.dataframe as dd

from ..util.utils import *
from ..datafiles import BGT, EPJ, EPJPNU
from ..msm.MSM_a7 import MSMInitializer

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

    def start_analysis(self):

        if (not os.path.isfile(self.filename + 'tica.pyemma')) or self.updating:
            print('Start new TICA analysis')
            if not self.data_collected:
                self.gather_feature_matrix()

            self.tica = pyemma.coordinates.tica(
                self.feature_trajectories, lag=self.lag)

            self.tica_output = self.tica.get_output()
            self.tica_concatenated = np.concatenate(self.tica_output)
            self.tica.save(self.filename + 'tica.pyemma', overwrite=True)
            pickle.dump(
                self.tica_output,
                open(
                    self.filename +
                    'output.pickle',
                    'wb'))
            self.dump_feature_trajectories()
            gc.collect()

        else:
            print('Load old TICA results')
            self.tica = pyemma.load(self.filename + 'tica.pyemma')
            self.tica_output = pickle.load(
                open(self.filename + 'output.pickle', 'rb'))
            self.tica_concatenated = np.concatenate(self.tica_output)

    def clustering_with_dask(self, meaningful_tic, n_clusters, updating=False):
        self.n_clusters = n_clusters
        self.meaningful_tic = meaningful_tic

        self.tica_output_filter = [
            np.asarray(output)[
                :, meaningful_tic] for output in self.tica_output]

        if not (os.path.isfile(self.cluster_filename + '_dask.pickle')) or updating:
            print('Start new cluster analysis')

            # not implemented yet
            self.d_cluster = dask.KMeans(
                n_clusters=self.n_clusters, init='k-means++', n_jobs=64)
            self.d_cluster.fit(np.concatenate(self.tica_output_filter)[::10])
            self.d_cluster_dtrajs = [self.d_cluster.predict(
                tica_out_traj).compute() for tica_out_traj in self.tica_output_filter]

            pickle.dump(
                self.d_cluster,
                open(
                    self.cluster_filename +
                    '_dask.pickle',
                    'wb'))
            pickle.dump(
                self.d_cluster_dtrajs,
                open(
                    self.cluster_filename +
                    '_output_dask.pickle',
                    'wb'))

        else:
            print('Load old cluster results')

            self.d_cluster = pickle.load(
                open(
                    self.cluster_filename +
                    '_dask.pickle',
                    'rb'))
            self.d_cluster_dtrajs = pickle.load(
                open(self.cluster_filename + '_output_dask.pickle', 'rb'))

        self.d_dtrajs_concatenated = np.concatenate(self.d_cluster_dtrajs)

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

    def get_its(self, cluster='dask'):
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

    def get_msm(self, lag, cluster='dask'):
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
                sys_data = raw_data[df.index[0] +
                                    self.start:df.index[-1] + 1, :]
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
