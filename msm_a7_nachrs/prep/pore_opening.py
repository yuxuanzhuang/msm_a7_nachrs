import MDAnalysis as mda
import nglview as nv
import numpy as np
import numpy as np
import MDAnalysis as mda
import os
from MDAnalysis.lib.distances import calc_bonds
import gmxapi as gmx

import shutil
import MDAnalysis.transformations as trans
from ENPMDA.utils import GroupHug

class PoreOpeningSimluationIteration(object):
    def __init__(self,
                 location,
                 simulation_name,
                 starting_file,
                 n_iterations,
                 pore_annotation,
                 gpu_id,
                 r0_shift=0.02):
        self.location = location
        self.simulation_name = simulation_name
        self.starting_file = starting_file
        self.n_iterations = n_iterations
        self.pore_annotation = pore_annotation
        self.gpu_id = gpu_id
        self.r0_shift = r0_shift

        self.iteration = 0
        self.iteration_files = []
        self.iteration_traj_files = []
        self.iteration_files.append(self.starting_file)
        self.iteration_traj_files.append(self.starting_file)
        self.bond_dist_list = []

        self.top_file = f'{self.location}/topol.top'
        self._add_restraint()
        self.mdp_file = f'{self.location}/mdp/pore_opening.mdp'
        self._create_mdp()
        self.index_file = f'{self.location}/index.ndx'
        self.bonded_topology = f'{self.location}/ca.tpr'
    
    def start_simulation(self):
        print('Starting simulation')
        os.makedirs(f'{self.location}/{self.simulation_name}/', exist_ok=True)
        os.makedirs(f'{self.location}/toppar/{self.simulation_name}', exist_ok=True)

        self.u = mda.Universe(self.iteration_files[-1])
        for i in range(self.n_iterations):
            print(f'Iteration {i}')
            self.iteration += 1
            self._get_pore_dist()
            print(self.bond_dist_list[-1])
            self._generate_pore_res()
            self._run_simulation()
            self.iteration_files.append(f'{self.location}/{self.simulation_name}/iter_{self.iteration}.pdb')
            self.iteration_traj_files.append(f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.xtc')

    def _get_pore_dist(self):
        self.u = mda.Universe(self.iteration_files[-1], self.iteration_traj_files[-1])

        u_prot = self.u.select_atoms('protein')

        u_bond = mda.Universe(self.bonded_topology)
        self.u.add_bonds(u_bond.bonds.to_indices())

        prot_chain_list = []

        # group all the protein chains
        for chain in u_prot.segments:
            prot_chain_list.append(chain.atoms)

        prot_group = GroupHug(*prot_chain_list)
        unwrap = trans.unwrap(self.u.atoms)
        center_in_box = trans.center_in_box(u_prot)
        self.u.trajectory.add_transformations(
                *[unwrap, prot_group, center_in_box])

        bond_dist = np.zeros([self.u.trajectory.n_frames, len(self.pore_annotation.keys()), 5])
        for i, ts in enumerate(self.u.trajectory):
            for j, (prime_not, selection) in enumerate(self.pore_annotation.items()):
                for k, (atom1, atom2) in enumerate(zip(self.u.select_atoms(selection),
                                        np.roll(self.u.select_atoms(selection),2))):
                    bond_dist[i, j, k] = calc_bonds(atom1.position, atom2.position) / 10
        self.bond_dist_list.append(bond_dist)

        np.save(f'{self.location}/{self.simulation_name}/bond_dist_list.npy',
                np.asarray(self.bond_dist_list))


    def _generate_pore_res(self):
        # how much to shift to left from initial distance
        # unit: nm
    #    r0_shift = 0.02
        # how much to shift to right from initial distance
        r1 = 10
        r2 = 20
        for i, (prime_not, selection) in enumerate(self.pore_annotation.items()):
            with open(f'{self.location}/toppar/{self.simulation_name}/pore_res_{prime_not}.itp', 'w') as f:
                f.write('[ bonds ]\n')
                for atom1, atom2 in zip(self.u.select_atoms(selection),
                                    np.roll(self.u.select_atoms(selection),2)):
                    bond_dist = calc_bonds(atom1.position, atom2.position) / 10
                    # gromacs index starts at 1
                    f.write(f'{atom1.index + 1} {atom2.index + 1} 10 {bond_dist-self.r0_shift:.4f} {bond_dist+r1:.4f} {bond_dist+r2:.4f} 5000\n')

    def _run_simulation(self):
        # Pore_restraint
        os.makedirs(f'{self.location}/{self.simulation_name}/iter_{self.iteration}', exist_ok=True)
        gmx_run = gmx.commandline_operation('gmx',
                                            arguments=['grompp', '-maxwarn', '1'],
                                            input_files={
                                                '-f': self.mdp_file,
                                                '-c': self.iteration_files[-1],
                                                '-r': self.iteration_files[-1],
                                                '-p': self.top_file,
                                                '-n': self.index_file,
                                            },
                                            output_files={
                                                '-o': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.tpr'
                                            }
                                            )
        gmx_run.run()
        if gmx_run.output.returncode.result() != 0:
            raise RuntimeError(
                'GMX failes with ' +
                gmx_run.output.stderr.result())

        gmx_run = gmx.commandline_operation(
                executable='gmx',
                arguments=['mdrun', '-gpu_id', str(self.gpu_id), '-ntmpi', '1', '-ntomp', '4', '-bonded', 'gpu', '-update', 'gpu',
                        '-pme', 'gpu'],
                input_files={
                    '-s': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.tpr',
                    '-cpi': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.cpt',
                },
                output_files={
                    '-c': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.gro',
                    '-x': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.xtc',
                    '-g': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.log',
                    '-e': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.edr',
                    '-cpo': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.cpt',
                })
        if not os.path.exists(f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.gro'):
            gmx_run.run()
            if gmx_run.output.returncode.result() != 0:
                raise RuntimeError(
                    'GMX failes with ' +
                    gmx_run.output.stderr.result())

        gmx_run = gmx.commandline_operation('gmx',
                                            arguments=['trjconv'],
                                            input_files={
                                                '-f': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.gro',
                                                '-s': f'{self.location}/{self.simulation_name}/iter_{self.iteration}/{self.simulation_name}.tpr',
                                            },
                                            output_files={
                                                '-o': f'{self.location}/{self.simulation_name}/iter_{self.iteration}.pdb'
                                            },
                                            stdin='0\n',
                                            )
        gmx_run.run()
        if gmx_run.output.returncode.result() != 0:
            raise RuntimeError(
                'GMX failes with ' +
                gmx_run.output.stderr.result())

    def _add_restraint(self):
        shutil.copy(self.top_file,
                    self.top_file.replace('topol',f'topol_{self.simulation_name}'))
        self.top_file = self.top_file.replace('topol',f'topol_{self.simulation_name}')
        with open(self.top_file, 'a') as f:
            f.write(f"#ifdef {self.simulation_name}\n")
            f.write("[ intermolecular_interactions]\n")
            for prime_not, selection in self.pore_annotation.items():
                f.write(f'#include "./toppar/{self.simulation_name}/pore_res_{prime_not}.itp"\n')
            f.write('#endif\n')

    def _create_mdp(self):
        shutil.copy(self.mdp_file,
                    self.mdp_file.replace('pore_opening',f'mdp_{self.simulation_name}'))
        self.mdp_file = self.mdp_file.replace('pore_opening',f'mdp_{self.simulation_name}')
        with open(self.mdp_file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"define    = -D{self.simulation_name}\n" + '\n' + content)