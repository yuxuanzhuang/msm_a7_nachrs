import sys
sys.path.insert(1, '/opt/tcbsys/gromacs/gmxapi/2021.4/for_gmx/AVX2_128')

import multiprocessing
from multiprocessing import Pool
import os
import subprocess
from functools import partial
import MDAnalysis as mda
import numpy as np

import gmxapi as gmx

import itertools
import linecache
import glob
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument

parser.add_argument('--system', type=str, required=True)
parser.add_argument('--seed_loc', type=str, required=False, default='SEEDS')

# Parse the argument
args = parser.parse_args()

seed_loc = args.seed_loc

def mdrun_run(mdrun):
    mdrun.run()
    return 'Finished'

def mdrun(seed_dir, load_dir, note, gpu_id):
    return gmx.commandline_operation(
                        executable='gmx',
                        arguments=['mdrun', '-gpu_id', str(gpu_id), '-ntmpi','1','-ntomp','4', '-bonded', 'gpu', '-update', 'gpu',
                                   '-pme','gpu'],
                        input_files={
                                '-s': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.tpr',
                                '-cpi': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.cpt',
                        },
                        output_files={
                                '-c': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.gro',
                                '-o': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.trr',
                                '-x': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.xtc',
                                '-g': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.log',
                                '-e': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.edr',
                                '-cpo': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.cpt',

                        })

def mdrun_em(seed_dir, load_dir, note, gpu_id):
    return gmx.commandline_operation(
                        executable='gmx',
                        arguments=['mdrun', '-ntmpi','1','-ntomp','16'],
                        input_files={
                                '-s': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.tpr'},
                        output_files={
                                '-c': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.gro',
                                '-o': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.trr',
                                '-x': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.xtc',
                                '-g': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.log',
                                '-e': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + note + '.edr'
                        })

def grompp_em(seed_dir, load_dir):
    gmx_run = gmx.commandline_operation('gmx',
                      arguments=['grompp','-maxwarn','1'],
                      input_files={
                            '-f': '../mdp/em.mdp',
                            '-c': '../' + seed_loc + '/' + load_dir + '/' + seed_dir + '/system/FINAL/start.pdb',
                            '-r': '../' + seed_loc + '/' + load_dir + '/' + seed_dir + '/system/FINAL/start.pdb',
                            '-p': '../' + seed_loc + '/' + load_dir + '/' + seed_dir + '/system/FINAL/topol.top',
                      },
                      output_files={
                            '-o': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/em.tpr'
                      }
                     )
    gmx_run.run()
    if gmx_run.output.erroroutput.result() != '':
        raise RuntimeError('GMX failes with ' + gmx_run.output.erroroutput.result())


def grompp_eq(seed_dir, load_dir, init, end):

    gmx_run = gmx.commandline_operation('gmx',
                      arguments=['grompp','-maxwarn','1'],
                      input_files={
                            '-f': '../mdp/' + end + '.mdp',
                            '-c': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + init + '.gro',
                            '-r': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/' + init + '.gro',
                            '-n': '../' + seed_loc + '/' + load_dir + '/' + seed_dir + '/system/FINAL/index.ndx',
                            '-p': '../' + seed_loc + '/' + load_dir + '/' + seed_dir + '/system/FINAL/topol.top'
                      },
                      output_files={
                            '-o': '../EQUILIBRATION/' + load_dir + '/' + seed_dir + '/'+ end + '.tpr'
                      }
                     )
    gmx_run.run()
    if gmx_run.output.erroroutput.result() != '':
        print('GMX failes with ' + gmx_run.output.erroroutput.result())


def run_command(seed_gpu_id, system):
    seed_dir = seed_gpu_id[0]
    load_dir = system
    gpu_id = seed_gpu_id[1]
    os.makedirs(f'../EQUILIBRATION/{system}/{seed_dir}', exist_ok=True)

    if os.path.isfile(f'../EQUILIBRATION/{system}/{seed_dir}/em.gro'):
        print(f'{system}/{seed_dir}/em exists')
    else:
        grompp_em(seed_dir, load_dir)
        run = mdrun_em(seed_dir, load_dir, 'em', gpu_id)
        mdrun_run(run)

    if os.path.isfile(f'../EQUILIBRATION/{system}/{seed_dir}/eq.gro'):
        print(f'{system}/{seed_dir}/eq exists')
    else:
        grompp_eq(seed_dir, load_dir, 'em', 'eq')
        run = mdrun(seed_dir, load_dir, 'eq', gpu_id)
        mdrun_run(run)

    if os.path.isfile(f'../EQUILIBRATION/{system}/{seed_dir}/heavy.gro'):
        print(f'{system}/{seed_dir}/heavy exists')
    else:
        grompp_eq(seed_dir, load_dir, 'eq', 'heavy')
        run = mdrun(seed_dir, load_dir, 'heavy', gpu_id)
        mdrun_run(run)

    if os.path.isfile(f'../EQUILIBRATION/{system}/{seed_dir}/backbone.gro'):
        print(f'{system}/{seed_dir}/backbone exists')
    else:
        grompp_eq(seed_dir, load_dir, 'heavy', 'backbone')
        run = mdrun(seed_dir, load_dir, 'backbone', gpu_id)
        mdrun_run(run)

    if True:
        if os.path.isfile(f'../EQUILIBRATION/{system}/{seed_dir}/ca.gro'):
            print(f'{system}/{seed_dir}/ca exists')
        else:
            grompp_eq(seed_dir, load_dir, 'backbone', 'ca')
            run = mdrun(seed_dir, load_dir, 'ca', gpu_id)
            mdrun_run(run)

    print('Finished equilibration for ' + system + '/' + seed_dir)
#    with open('finished.txt', 'a') as f:
#        f.write('Finished equilibration for ' + system + '/' + seed_dir)
    return 0


if __name__ == '__main__':
    pool = Pool(4)
    system = args.system
    seeds = os.listdir(f'../SEEDS/{system}/')

    # more than enough gpu_ids to iterate over all seeds
    # four GPU in each node
    gpu_ids = [0, 1, 2, 3] * 20
    run_command_para = partial(run_command,
                            system=system)
    pool.map(run_command_para, zip(seeds, gpu_ids), chunksize=1)
    print('finished')

