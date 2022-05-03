import os
import numpy as np
import pandas as pd
import pyemma
import pickle
from scipy.stats import pearsonr

#from dask_ml.cluster import KMeans

import gc
import itertools
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align, pca, rms

#import pmda.rms
import nglview as nl

import dask.dataframe as dd

from ..util.utils import *
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

n_seeds_dic = {
              'BGT_EPJPNU': 24,
              'BGT_EPJ': 24,
              'EPJPNU_BGT': 25,
              'EPJPNU_EPJ': 24,
              'EPJ_BGT': 24,
              'EPJ_EPJPNU': 24
}

"""
1: 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
2: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
3: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
4: 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96
5: 97 98 99 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
6: 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
"""

#  exclused seeds from MSM
system_exclusion_dic = {'BGT_EPJPNU': [],
                   'BGT_EPJ': [19],
                   'EPJPNU_BGT': [19, 20, 21],
                   'EPJPNU_EPJ': [],
                   'EPJ_BGT': [],
                   'EPJ_EPJPNU': []}

new_system_exclusion_dic = {'BGT_EPJPNU': [],
                   'BGT_EPJ': [],
                   'EPJPNU_BGT': [],
                   'EPJPNU_EPJ': [],
                   'EPJ_BGT': [],
                   'EPJ_EPJPNU': []}




pdb = '../PRODUCTION/BGT_EPJPNU/SEEDS_0/protein.pdb'

traj_notes = [
              'BGT_EPJPNU',
              'BGT_EPJ',
              'EPJPNU_BGT',
              'EPJPNU_EPJ',
              'EPJ_BGT',
              'EPJ_EPJPNU'
              ]

production_dic = {'traj_note': traj_notes,
                 'load_location': ["".join(i) for i in itertools.product([pwd + '/../PRODUCTION/'], traj_notes)],
                 'save_location': [pwd + '/../ANALYSIS/two_traj/'] * len(traj_notes),
                 'skip': [1] * len(traj_notes)}

traj_note_dic = production_dic

trajectory_list = []
for traj_note, load_location in zip(
    traj_note_dic["traj_note"], traj_note_dic["load_location"]
):
    for seed in sorted(os.listdir(load_location), key=natural_keys):
        if seed.startswith("SEEDS"):
            trajectory_list.append(load_location + "/" + seed + "/protein.xtc")

resid_selection = "name CA"

structure_list = ["BGT_EPJPNU", "EPJ_BGT", "EPJPNU_EPJ"]
pdb_file = []
for structure in BGT, EPJ, EPJPNU:
    pdb_file.append(structure)
u_ref = mda.Universe(pdb_file[0], *pdb_file)

aligner_ref = align.AlignTraj(
    u_ref, u_ref, select=resid_selection, in_memory=True
).run()

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
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
    return scores



class MSMInitializer(object):
    def __init__(self, tica_file, updating=False, lag=20, start=0,
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
        self.tica_file = tica_file
        self.start = start
        self.pathways = pathways
        self.updating = updating
        self.feature_file = feature_file
        self.system_exclusion_dic = system_exclusion_dic
        
        self.md_data = pd.read_pickle(self.feature_file)
#        self.md_data = self.md_data.applymap(lambda x: x.replace('/mnt/cephfs/projects/2020100800_alpha7_nachrs/MSM',
#                           '/mnt/cephfs/projects/2021072000_nachrs_a7_msm/a7_nachrs') if isinstance(x, str) else x)
        # dt: ns
        self.dt = (self.md_data.traj_time[1] - self.md_data.traj_time[0]) / 1000
        print(f'lag time is {self.lag * self.dt} ns')
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
        
        self.feature_info = np.load('./analysis_results/' + self.tica_file + '_feature_info.npy')
        
        self.feed_feature_indice = []
        for ind, feat in enumerate(self.feature_info):
            if not any(exl_feat in feat for exl_feat in self.domain_exclusion):
                self.feed_feature_indice.append(ind)
                
        self.feature_info = self.feature_info[self.feed_feature_indice]
        
        
    def start_tica_analysis(self):
        if (not os.path.isfile(self.filename  + 'tica.pyemma')) or self.updating:
            print('Start new TICA analysis')

            self.start_analysis()

        else:
            print('Load old TICA results')
            self.tica = pyemma.load(self.filename  + 'tica.pyemma')
            self.tica_output = pickle.load(open(self.filename  + 'output.pickle', 'rb'))
            self.tica_concatenated = np.concatenate(self.tica_output)
            
            self.pathway_seed_start = [0]
            old_pathway = self.pathways[0]
            sys_index = 1
            for sys, df in self.md_data[self.md_data.pathway.isin(self.pathways)].groupby(['system']):
                if sys not in self.system_exclusion:
                    if df.pathway.iloc[0] != old_pathway:
                        self.pathway_seed_start.append(sys_index)
                    old_pathway = df.pathway.iloc[0]
                    sys_index += 5
        
    def start_analysis(self):
        feature_data = []

        raw_data = np.concatenate([np.load(location, allow_pickle=True)
                        for location, df in self.md_data.groupby(self.tica_file, sort=False)])
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
                
                #  symmetrization of features
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            0, axis=1).reshape(sys_data.shape[0],total_shape))
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
                sys_index += 5
            
            
        if self.extra_feature_file != None:
            raw_data = np.concatenate([np.load(location, allow_pickle=True)
                        for location, df in self.md_data_extra.groupby(self.tica_file, sort=False)])
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

                    #  symmetrization of features
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                                0, axis=1).reshape(sys_data.shape[0],total_shape))
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
                    sys_index += 5
                

        
        self.tica = pyemma.coordinates.tica(feature_data, lag=self.lag)
                
        self.tica_output = self.tica.get_output()
        self.tica_concatenated = np.concatenate(self.tica_output)
        self.tica.save(self.filename + 'tica.pyemma', overwrite=True)
        pickle.dump(self.tica_output, open(self.filename  + 'output.pickle', 'wb'))

       
        
        del feature_data
        del raw_data
        gc.collect()
        
        
    def clustering_with_dask(self, meaningful_tic, n_clusters, updating=False):
        self.n_clusters = n_clusters
        self.meaningful_tic = meaningful_tic

        self.tica_output_filter = [np.asarray(output)[:,meaningful_tic] for output in self.tica_output]
        
        if not (os.path.isfile(self.cluster_filename + '_dask.pickle')) or updating:
            print('Start new cluster analysis')

            # not implemented yet
            self.d_cluster = dask.KMeans(n_clusters=self.n_clusters, init='k-means++', n_jobs=64)
            self.d_cluster.fit(np.concatenate(self.tica_output_filter)[::10])
            self.d_cluster_dtrajs = [self.d_cluster.predict(tica_out_traj).compute() for tica_out_traj in self.tica_output_filter]
            
            pickle.dump(self.d_cluster, open(self.cluster_filename + '_dask.pickle', 'wb'))
            pickle.dump(self.d_cluster_dtrajs, open(self.cluster_filename + '_output_dask.pickle', 'wb'))

        else:
            print('Load old cluster results')

            self.d_cluster = pickle.load(open(self.cluster_filename + '_dask.pickle', 'rb'))
            self.d_cluster_dtrajs = pickle.load(open(self.cluster_filename + '_output_dask.pickle', 'rb'))
               
        self.d_dtrajs_concatenated = np.concatenate(self.d_cluster_dtrajs)
        
        
    def clustering_with_pyemma(self, meaningful_tic, n_clusters, updating=False):
        self.n_clusters = n_clusters
        self.meaningful_tic = meaningful_tic

        self.tica_output_filter = [np.asarray(output)[:,meaningful_tic] for output in self.tica_output]
        
        if not (os.path.isfile(self.cluster_filename + '_pyemma.pickle')) or updating:
            print('Start new cluster analysis')

            self.cluster = pyemma.coordinates.cluster_kmeans(self.tica_output_filter,
                                                             k=self.n_clusters,
                                                             max_iter=100,
                                                             stride=100)
            self.cluster_dtrajs = self.cluster.dtrajs
            
            pickle.dump(self.cluster, open(self.cluster_filename + '_pyemma.pickle', 'wb'))
            pickle.dump(self.cluster_dtrajs, open(self.cluster_filename + '_output_pyemma.pickle', 'wb'))

        else:
            print('Load old cluster results')
            self.cluster = pickle.load(open(self.cluster_filename + '_pyemma.pickle', 'rb'))
            self.cluster_dtrajs = pickle.load(open(self.cluster_filename + '_output_pyemma.pickle', 'rb'))
               
        self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)
        

    def get_its(self, cluster='dask'):
        if cluster == 'dask':
            self.its = pyemma.msm.its(self.d_cluster_dtrajs, lags=int(200 / self.dt), nits=10, errors='bayes', only_timescales=True)
        elif cluster == 'pyemma':
            self.its = pyemma.msm.its(self.cluster_dtrajs, lags=int(200 / self.dt), nits=10, errors='bayes', only_timescales=True)

        return self.its
        
    def get_msm(self, lag, cluster='dask'):
        self.msm_lag = lag
        
        if cluster == 'dask':
            self.msm = pyemma.msm.bayesian_markov_model(self.d_cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')
        elif cluster == 'pyemma':
            self.msm = pyemma.msm.bayesian_markov_model(self.cluster_dtrajs, lag=self.msm_lag, dt_traj=str(self.dt) + ' ns')
        
        return self.msm
        

    def get_correlation(self, feature):
        md_data = pd.read_pickle(self.feature_file)

        raw_data = np.concatenate([np.load(location, allow_pickle=True)
                        for location, df in md_data.groupby(feature, sort=False)])
        
        md_data = md_data[md_data.pathway.isin(self.pathways)]
                
        feature_data = []
        
        for sys, df in md_data.groupby(['system']):
            if sys not in self.system_exclusion:
                sys_data = raw_data[df.index[0] + self.start:df.index[-1]+1, :]
                total_shape = sys_data.shape[1]
                per_shape = int(total_shape / 5)
                
                #  symmetrization of features
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            0, axis=1).reshape(sys_data.shape[0],total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            1, axis=1).reshape(sys_data.shape[0],total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            2, axis=1).reshape(sys_data.shape[0],total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            3, axis=1).reshape(sys_data.shape[0],total_shape))
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],5,per_shape),
                                            4, axis=1).reshape(sys_data.shape[0],total_shape))
                        
        feature_data_concat = np.concatenate(feature_data)
        
        max_tic = 20
        test_feature_TIC_correlation = np.zeros((feature_data_concat.shape[1], max_tic))

        for i in range(feature_data_concat.shape[1]):
            for j in range(max_tic):
                test_feature_TIC_correlation[i, j] = pearsonr(
                    feature_data_concat[:, i],
                    self.tica_concatenated[:, j])[0]
              
            
        feature_info = np.load('./analysis_results/' + feature + '_feature_info.npy')
        
        del feature_data
        del feature_data_concat
        gc.collect()
        
        return test_feature_TIC_correlation, feature_info


    @property
    def filename(self):
        return './msmfile/' + self.tica_file + '_lag' + str(self.lag) + '_start' + str(self.start) + '_pathway' + ''.join([str(pathway_ind_dic[pathway]) for pathway in self.pathways]) + '_'
    
    @property
    def cluster_filename(self):
        return self.filename + '_cluster' + str(self.n_clusters) + '_tic' + '_'.join([str(m_tic) for m_tic in self.meaningful_tic]) + '_'
