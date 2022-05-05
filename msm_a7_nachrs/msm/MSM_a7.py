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
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])
    return scores


class MSMInitializer(object):
    prefix="msm"

    def __init__(self,
                 feature_selection,
                 multimer=5,
                 symmetrize=True,
                 updating=True,
                 dumping=True,
                 lag=100,
                 start=0,
                 pathways = traj_notes,
                 system_exclusion_dic = system_exclusion_dic,
                 domain_exclusion = [],
                 feature_file = 'msm_features.pickle',
                 interval=1):
        # lag for # of frames
        self.lag = lag
        self.feature_selection = feature_selection
        self.start = start
        self.dumping = dumping
        self.multimer = multimer
        self.symmetrize = symmetrize

        self.pathways = pathways
        self.updating = updating
        self.feature_file = feature_file
        self.system_exclusion_dic = system_exclusion_dic
        self.interval = interval
        self.md_data = pd.read_pickle(self.feature_file)
#        self.md_data = self.md_data[self.md_data.frame % (3 * self.interval) == 0]

        # dt: ns
        self.dt = (self.md_data.traj_time[1] - self.md_data.traj_time[0]) / 1000 * self.interval
        print(f'lag time is {self.lag * self.dt} ns')

        system_exclusion = []
        for pathway in self.pathways:
            for exclu_seed in system_exclusion_dic[pathway]:
                exclu_system_ind = self.md_data[self.md_data.pathway == pathway][self.md_data.seed == exclu_seed]['system'].values[0]
                system_exclusion.append(exclu_system_ind)


        if feature_selection not in self.md_data.columns:
            raise ValueError(
                f'feature selection {feature_selection} not in'
                f'{self.feature_file}\n'
                f'Available features are {self.md_data.columns}')

        self.feature_info = []
        self.all_feed_feature_indice = []
        self.get_feature_info(feature_selection, domain_exclusion)
        print(f'added feature selection {feature_selection}, # of features: {len(self.feature_info[-1])}')

        self.all_feature_selection = [feature_selection]
        self.all_feature_file = [feature_file]
        self.all_pathways = [pathways]
        self.all_start = [start]
        self.all_system_exclusion_dic = [system_exclusion_dic]
        self.all_domain_exclusion = [domain_exclusion]
        self.all_system_exclusion = [system_exclusion]
        self.all_md_data = [self.md_data]

        self.data_collected = False

        os.makedirs(self.filename, exist_ok=True)

    def add_features(self,
                     feature_selection,
                     domain_exclusion=[]):

        if self.data_collected:
            raise ValueError('Feature already collected, create new instance')
        self.all_feature_selection.append(feature_selection)
        self.all_domain_exclusion.append(domain_exclusion)

        if feature_selection not in self.md_data.columns:
            raise ValueError(
                f'feature selection {feature_selection} not in'
                f'{self.feature_file}\n'
                f'Available features are {self.md_data.columns[8:]}')

        for ind, md_data in enumerate(self.all_md_data):
            if feature_selection not in md_data.columns:
                raise ValueError(
                    f'feature selection {feature_selection} not in'
                    f'{self.add_features[ind]}\n'
                    f'Available features are {md_data.columns[8:]}')
        self.get_feature_info(feature_selection, domain_exclusion)
        print(f'added feature selection {feature_selection}, # of features: {len(self.feature_info[-1])}')

    def add_trajectories(self,
                         feature_file= 'msm_features_new.pickle',
                         pathways = traj_notes,
                         start=None,
                         system_exclusion_dic = system_exclusion_dic,
                         domain_exclusion= []):
        
        if start is None:
            start=self.start
            
        if self.data_collected:
            raise ValueError('Feature already collected, create new instance')

        self.all_feature_file.append(feature_file)
        self.all_pathways.append(pathways)
        self.all_start.append(start)
        self.all_system_exclusion_dic.append(system_exclusion_dic)
        self.all_domain_exclusion.append(domain_exclusion)

        md_data = pd.read_pickle(feature_file)
#        md_data = md_data[md_data.frame % (3 * self.interval) == 0]

        if self.dt != (md_data.traj_time[1] - md_data.traj_time[0]) / 1000 * self.interval:
            raise ValueError('dt is not the same in new feature file ' + feature_file)

        self.all_md_data.append(md_data)
        
        system_exclusion = []
        for pathway in pathways:
            for exclu_seed in system_exclusion_dic[pathway]:
                exclu_system_ind = self.md_data[self.md_data.pathway == pathway][self.md_data.seed == exclu_seed]['system'].values[0]
                system_exclusion.append(exclu_system_ind)
        
        self.all_system_exclusion.append(system_exclusion)
        
    def get_feature_info(self, feature_selection, domain_exclusion):
        feature_info = np.load('./analysis_results/' + feature_selection + '_feature_info.npy', allow_pickle=True)
        feed_feature_indice = []
        for ind, feat in enumerate(feature_info):
            if not any(exl_feat in feat for exl_feat in domain_exclusion):
                feed_feature_indice.append(ind)
                
        self.feature_info.append(feature_info[feed_feature_indice])
        self.all_feed_feature_indice.append(feed_feature_indice)        

    def load_raw_data(self, feature_selection, feature_index, ensemble_index):
        #TODO: feature dim 1 not work
        md_data = self.all_md_data[ensemble_index]

        raw_data = np.concatenate([np.load(location, allow_pickle=True)[:, self.all_feed_feature_indice[feature_index]]
                        for location, df in md_data.groupby(feature_selection, sort=False)])
        
        #TODO: make sure float16 is enough accuracy
        raw_data = raw_data.astype(np.float32)
        md_data = md_data[md_data.pathway.isin(self.all_pathways[ensemble_index])]
        print('Feed n.frames: ' + str(md_data.shape[0]))

        md_data = md_data[~md_data.system.isin(self.all_system_exclusion[ensemble_index])]

        feature_data = []
        for sys, df in md_data.groupby(['system']):
            sys_data = raw_data[df.index[0] + self.all_start[ensemble_index]:df.index[-1]+1][::self.interval]
            total_shape = sys_data.shape[1]
            per_shape = int(total_shape / self.multimer)
            
            if self.symmetrize:
            #  symmetrization of features
                for permutation in range(self.multimer):
                    feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],self.multimer,per_shape),
                        permutation, axis=1).reshape(sys_data.shape[0],total_shape))
            else:
                feature_data.append(np.roll(sys_data.reshape(sys_data.shape[0],self.multimer,per_shape),
                        0, axis=1).reshape(sys_data.shape[0],total_shape))
                
        del raw_data
        del sys_data
        gc.collect()
        return feature_data

    def gather_feature_matrix(self):
        all_feature_data = []
        for ensemble_index, ensemble in enumerate(self.all_md_data):
            print('run for ' + self.all_feature_file[ensemble_index])
            feature_ensemble = []
            for feature_index, feature_selection in enumerate(self.all_feature_selection):
                print('run for ' + feature_selection)
                feature_ensemble.append(self.load_raw_data(feature_selection, feature_index, ensemble_index))
                gc.collect()
            concat_feature_data = []
            for system in range(0, len(feature_ensemble[0])):
                concat_feature_data.append(np.concatenate([feat_sys[system].astype(np.float32) for feat_sys in feature_ensemble], axis=1, dtype=np.float32))
            all_feature_data.append(concat_feature_data)

            del feature_ensemble
            del concat_feature_data

            gc.collect()
        self.data_collected = True
        self.feature_trajectories = list(itertools.chain.from_iterable(all_feature_data))
        self.n_trajectories = len(self.feature_trajectories)
    #    self.feature_trajectories = list(np.concatenate(all_feature_data))


    def dump_feature_trajectories(self):
        if self.data_collected:
            if self.dumping:
                for ind, feature_matrix in enumerate(self.feature_trajectories):
                    np.save(self.filename + f'feature_traj_{ind}.npy', feature_matrix)
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
                self.feature_trajectories.append(np.load(self.filename + f'feature_traj_{ind}.npy', allow_pickle=True))

    def start_analysis(self):
        raise NotImplementedError("Should be overridden by subclass")
        if not self.data_collected:
            self.gather_feature_matrix()       
        
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
            
            if self.dumping:
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
        return './msmfile/' + self.prefix + '_' + self.feature_selection + '_lag' + str(self.lag) + '_start' + str(self.start) + '_pathway' + ''.join([str(pathway_ind_dic[pathway]) for pathway in self.pathways]) + '/'
    
    @property
    def cluster_filename(self):
        return self.filename + 'cluster' + str(self.n_clusters) + '_tic' + '_'.join([str(m_tic) for m_tic in self.meaningful_tic]) + '_'