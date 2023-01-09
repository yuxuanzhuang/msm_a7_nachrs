'''
Annotations for nicotinic receptor alpha-7 project
'''
import numpy as np
import dask
import pickle
import re
import itertools
import os
pwd = os.getcwd()

subunit_iter_dic = {'A': 'B',
                    'B': 'C',
                    'C': 'D',
                    'D': 'E',
                    'E': 'A',
                    'F': 'G',
                    'G': 'H',
                    'H': 'I',
                    'I': 'J',
                    'J': 'F'}
                    
#  used for mdtraj
subunit_dic = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, -1: 4}

#  used for MDAnalysis
subunit_dic_mda = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'A'}

subunit_type = {0: 'a7', 1: 'a7', 2: 'a7', 3: 'a7', 4: 'a7', 5: 'a7'}

#  pore lining residues
pore_annotation = {
    'a7': {
        '-1': '(resid 237 and resname GLU)',
        '2': '(resid 240 and resname SER)',
        '6': '(resid 244 and resname THR)',
        '9': '(resid 247 and resname LEU)',
        '13': '(resid 251 and resname VAL)',
        '16': '(resid 254 and resname LEU)',
        '20': '(resid 258 and resname GLU)'
    }
}
secondary_structure_annotation = {
    'a7': {
        'ECD': '(resSeq 10 to 206)',
        'TMD': '(resSeq 207 to 401)'
    }
}

domain_structure_annotation = {
    'a7': {
        'M1': '(resSeq 207 to 232)',
        'M2': '(resSeq 235 to 261)',
        'M3': '(resSeq 267 to 298)',
        'MX': '(resSeq 300 to 321)',
        'MA': '(resSeq 331 to 358)',
        'M4': '(resSeq 359 to 390)',
        'MC': '(resSeq 391 to 401)',
        'loop_C': '(resSeq 176 to 205)',
        'M2_M3_loop': '(resSeq 260 to 268)',
    }
}

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
                  'save_location': [pwd + '/../ANALYSIS/TRAJECTORY/'] * len(traj_notes),
                  'skip': [1] * len(traj_notes)}

production_dic_2 = {'traj_note': traj_notes,
                    'load_location': ["".join(i) for i in itertools.product([pwd + '/../PRODUCTION_new/'], traj_notes)],
                    'save_location': [pwd + '/../ANALYSIS/TRAJECTORY_new/'] * len(traj_notes),
                    'skip': [1] * len(traj_notes)}

equilibration_dic = {'traj_note': traj_notes,
                     'load_location': ["".join(i) for i in itertools.product([pwd + '/../EQUILIBRATION/'], traj_notes)],
                     'save_location': [pwd + '/../EQUILIBRATION/'] * len(traj_notes),
                     'skip': [1] * len(traj_notes)}

climber_dic = {'traj_note': ['BGT_EPJPNU', 'BGT_EPJ', 'EPJ_EPJPNU', 'EPJ_BGT', 'EPJPNU_BGT', 'EPJPNU_EPJ'],
               'load_location': ["".join(i) for i in itertools.product([pwd + '/../Climber/'], ['BGT_EPJPNU', 'BGT_EPJ', 'EPJ_EPJPNU', 'EPJ_BGT', 'EPJPNU_BGT', 'EPJPNU_EPJ'])]}


import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()