import multiprocessing
from multiprocessing import Pool
import os
import shutil
import subprocess
from functools import partial

from ..util.ring_penetration import check_universe_ring_penetration
from MDAnalysis.transformations.wrap import unwrap
import MDAnalysis as mda
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument

parser.add_argument('--system', type=str, required=True)
parser.add_argument('--seed_loc', type=str, required=False, default='SEEDS')

# Parse the argument
args = parser.parse_args()

seed_loc = args.seed_loc


def run_command(seed_gpu_id, system):
    seed = seed_gpu_id[0]
    gpu_id = seed_gpu_id[1]
    # TODO: make this a function
    command = f"/nethome/yzhuang/anaconda3/envs/deeplearning/bin/python build_system.py --system {system} --seed {seed} --gpu_id {gpu_id}"
    print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()

    # This makes the wait possible
    p_status = p.wait()

    process = multiprocessing.current_process()
    with open('prep_output/ ' + system + '_' + seed + '_' + str(process.pid) + '.output', 'w') as f:
        f.write('finish')

    # This will give you the output of the command being executed
    print(f"{system} {seed}")

    # check ring penetration
    u = mda.Universe('../' + seed_loc + '/' + system + '/' + seed + '/system/FINAL/em.tpr',
                     '../' + seed_loc + '/' + system + '/' + seed + '/system/FINAL/start.pdb')
    u.trajectory.add_transformations(unwrap(u.atoms))
    if len(check_universe_ring_penetration(u)) > 3:
        # rebuild
        shutil.rmtree('../' + seed_loc + '/' + system + '/' + seed + '/system')
        command = f"/nethome/yzhuang/anaconda3/envs/deeplearning/bin/python build_system.py --system {system} --seed {seed} --gpu_id {gpu_id}"
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()

        # This makes the wait possible
        p_status = p.wait()

        process = multiprocessing.current_process()
        with open('prep_output/ ' + system + '_' + seed + '_' + str(process.pid) + '.output', 'w') as f:
            #        f.writelines(output)
            #        f.writelines(err)
            f.write('finish')

        # This will give you the output of the command being executed
    #    print("Command output: " + output)
        print(f"{system} {seed} has ring penetration")
        with open('ring_penetration.txt', 'a') as f:
            f.write(f"{system} {seed}\n")


pool = Pool(4)
system = args.system
seeds = os.listdir('../' + seed_loc + f'/{system}/')

# need more than enough GPU_ids to iterate
gpu_ids = [0, 1, 2, 3] * 20
run_command_para = partial(run_command,
                           system=system)
pool.map(run_command_para, zip(seeds, gpu_ids), chunksize=1)
