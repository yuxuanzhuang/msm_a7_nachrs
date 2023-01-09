from ENPMDA.analysis import *
from ENPMDA.analysis.base import DaskChunkMdanalysis


from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.analysis.distances import dist
from MDAnalysis.core.groups import AtomGroup, ResidueGroup

import MDAnalysis as mda
import itertools
import pandas as pd

from ..util.utils import subunit_iter_dic

domain_dict = {
    'ECD': '(resid 0 to 207)',
    'M1': '(resid 207 to 232)',
    'M1_upper': '(resid 207 to 220)',
    'M1_lower': '(resid 220 to 232)',
    'M2': '(resid 235 to 261)',
    'M2_upper': '(resid 235 to 248)',
    'M2_lower': '(resid 248 to 261)',
    '9prime': '(resid 247)',
    'M3': '(resid 267 to 298)',
    'M3_upper': '(resid 267 to 282)',
    'M3_lower': '(resid 282 to 298)',
    'MX': '(resid 300 to 321)',
    'MA': '(resid 331 to 358)',
    'M4': '(resid 359 to 390)',
    'M4_upper': '(resid 359 to 375)',
    'M4_lower': '(resid 375 to 390)',
    'MC': '(resid 391 to 401)',
    'loop_C': '(resid 183 to 195)',
    'M2_M3_loop': '(resid 260 to 268)',
    'loop_F': '(resid 168 to 175)',
    'pro_loop': '(resid 129 to 137)'
}

pore_annotation_prime = {
            '-1': '(resid 237 and resname GLU)',
            '2': '(resid 240 and resname SER)',
            '6': '(resid 244 and resname THR)',
            '9':  '(resid 247 and resname LEU)',
            '13': '(resid 251 and resname VAL)',
            '16': '(resid 254 and resname LEU)',
            '20': '(resid 258 and resname GLU)'
}


class get_domain_position(DaskChunkMdanalysis):
    name = 'domain_position'

    def set_feature_info(self, universe):
        domain_info_list = []

        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_info_list.append(
                    domain_name + '_' + domain_subunit.segids[0] + '_pos')

        domain_info_list = np.asarray(
            domain_info_list,
            dtype=object).reshape(
            len(domain_dict),
            5).T.ravel()

        return domain_info_list

    def run_analysis(self, universe, start, stop, step):
        CA_atoms = universe.select_atoms('name CA')

        domain_ag_list = []

        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_ag_list.append(domain_subunit)
                    
        domain_ag_list = np.asarray(
            domain_ag_list,
            dtype=object).reshape(
            len(domain_dict),
            5).T.ravel()


        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(np.asarray([calc_bonds(domain_ag.center_of_geometry(),
                                      CA_atoms.center_of_geometry())
                                      for domain_ag in domain_ag_list]))

        return result


class get_domain_interdistance(DaskChunkMdanalysis):
    name = 'domain_distance'

    def set_feature_info(self, universe):
        domain_info_list = []
        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_info_list.append(
                    domain_name + '_' + domain_subunit.segids[0] + '_pos')

        domain_info_list = np.asarray(
            domain_info_list, dtype=object).reshape(
            len(domain_dict), 5).T

        feature_info = []
        for feature_chain in [[0, 1, 2], [1, 2, 3],
                              [2, 3, 4], [3, 4, 0], [4, 0, 1]]:
            for d1_inf, d2_inf in itertools.product(
                    domain_info_list[feature_chain[0]],
                    domain_info_list[feature_chain].ravel()):
                if d1_inf != d2_inf:
                    feature_info.append('_'.join([d1_inf, d2_inf]))

        return feature_info        

    def run_analysis(self, universe, start, stop, step):
        domain_ag_list = []
        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_ag_list.append(domain_subunit)

        domain_ag_list = np.asarray(
            domain_ag_list, dtype=object).reshape(
            len(domain_dict), 5).T

        result = []
        for ts in universe.trajectory[start:stop:step]:
            domain_ag_c_gom_list = np.zeros(domain_ag_list.shape + (3,))
            for i, domain_ag in np.ndenumerate(domain_ag_list):
                domain_ag_c_gom_list[i] = domain_ag.center_of_geometry()
            
            ag_gom_list1 = []
            ag_gom_list2 = []
            for feature_chain in [[0, 1, 2], [1, 2, 3],
                                  [2, 3, 4], [3, 4, 0], [4, 0, 1]]:
                for domain_ag1, domain_ag2 in itertools.product(
                        domain_ag_c_gom_list[feature_chain[0]], domain_ag_c_gom_list[feature_chain].reshape(-1, 3)):
                    if (domain_ag1 != domain_ag2).any():
                        ag_gom_list1.append(domain_ag1)
                        ag_gom_list2.append(domain_ag2)
            r_ts = calc_bonds(np.vstack(ag_gom_list1), np.vstack(ag_gom_list2))
            result.append(list(r_ts))

        return result


class get_domain_intradistance(DaskChunkMdanalysis):
    name = 'domain_intra_distance'

    def set_feature_info(self, universe):
        domain_info_list = []
        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_info_list.append(
                    domain_name + '_' + domain_subunit.segids[0] + '_pos')

        domain_info_list = np.asarray(
            domain_info_list, dtype=object).reshape(
            len(domain_dict), 5).T

        feature_info = []
        for feature_chain in range(5):
            for d1_inf, d2_inf in itertools.product(
                    domain_info_list[feature_chain],
                    domain_info_list[feature_chain].ravel()):
                if d1_inf != d2_inf:
                    feature_info.append('_'.join([d1_inf, d2_inf]))
        return feature_info

    def run_analysis(self, universe, start, stop, step):
        domain_ag_list = []
        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_ag_list.append(domain_subunit)

        domain_ag_list = np.asarray(
            domain_ag_list, dtype=object).reshape(
            len(domain_dict), 5).T

        result = []
        for ts in universe.trajectory[start:stop:step]:

            domain_ag_c_gom_list = np.zeros(domain_ag_list.shape + (3,))
            for i, domain_ag in np.ndenumerate(domain_ag_list):
                domain_ag_c_gom_list[i] = domain_ag.center_of_geometry()
            
            ag_gom_list1 = []
            ag_gom_list2 = []
            for feature_chain in range(5):
                for domain_ag1, domain_ag2 in itertools.product(
                        domain_ag_c_gom_list[feature_chain], domain_ag_c_gom_list[feature_chain].reshape(-1, 3)):
                    if (domain_ag1 != domain_ag2).any():
                        ag_gom_list1.append(domain_ag1)
                        ag_gom_list2.append(domain_ag2)
            r_ts = calc_bonds(np.vstack(ag_gom_list1), np.vstack(ag_gom_list2))
            result.append(list(r_ts))

        return result


#arg_comp_candidates = np.load('arg_comp_candidates.npy')
#feature_list = np.load('feature_list.npy')


class get_c_alpha_distance(DaskChunkMdanalysis):
    name = 'ca_distance'

    def run_analysis(self, universe, start, stop, step):

        selection_comb = [['A F', 'A F B G C H'],
                          ['B G', 'B G C H D I'],
                          ['C H', 'C H D I E J'],
                          ['D I', 'D I E J A F'],
                          ['E J', 'E J A F B G']]

        result = []
        for ts in universe.trajectory[start:stop:step]:
            r_ts = []

            for selection1, selection2 in selection_comb:

                ag_sel1_collection = []
                ag_sel2_collection = []

                for sel in selection1.split(' '):
                    ag_sel1_collection.append(
                        universe.select_atoms(
                            'name CA and chainid ' + sel))
                for sel in selection2.split(' '):
                    ag_sel2_collection.append(
                        universe.select_atoms(
                            'name CA and chainid ' + sel))

                distance_matrix = distance_array(np.concatenate([ag.positions for ag in ag_sel1_collection]),
                                                 np.concatenate([ag.positions for ag in ag_sel2_collection]))
                filtered_distance_matrix = []
                for i, j in arg_comp_candidates:
                    filtered_distance_matrix.append(distance_matrix[i, j])

                r_ts.extend(filtered_distance_matrix)
            result.append(r_ts)

        self._feature_info = []
        for selection1, selection2 in selection_comb:
            for feature_chain in feature_list:
                self._feature_info.append(
                    selection1.split()[0] + '_' + feature_chain)

        return result


class get_rmsd_ref(DaskChunkMdanalysis):
    name = 'rmsd_to_stat'

    def set_feature_info(self, universe):
        return ['rmsd_2_bgt', 'rmsd_2_epj', 'rmsd_2_pnu']

    def run_analysis(self, universe, start, stop, step):
        name_ca = universe.select_atoms('name CA')
        rmsd_bgt = RMSD(
            name_ca,
            u_ref.select_atoms('name CA'),
            ref_frame=0).run(
            start,
            stop,
            step)
        rmsd_epj = RMSD(
            name_ca,
            u_ref.select_atoms('name CA'),
            ref_frame=1).run(
            start,
            stop,
            step)
        rmsd_pnu = RMSD(
            name_ca,
            u_ref.select_atoms('name CA'),
            ref_frame=2).run(
            start,
            stop,
            step)

        n_frames = rmsd_bgt.results['rmsd'].T[2].shape[0]
        return np.concatenate([rmsd_bgt.results['rmsd'],
                               rmsd_epj.results['rmsd'],
                               rmsd_pnu.results['rmsd']]).T[2].reshape(3, n_frames).T


class get_pore_hydration(DaskChunkMdanalysis):
    name = 'pore_hydration'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['pore_hydration']

    def run_analysis(self, universe, start, stop, step):
        pore_hydrat_n = universe.select_atoms(
            "(cyzone 9 6 -6 resid 247) and resname TIP3 and name OH2", updating=True)

        n_hydration = []
        for ts in universe.trajectory[start:stop:step]:
            n_hydration.append(pore_hydrat_n.n_atoms)
        return n_hydration

class get_pore_hydration_prime(DaskChunkMdanalysis):
    name = 'pore_hydration_prime'
    universe_file = 'system'

    def set_feature_info(self, universe):
        feats = []
        for annotation, sel in pore_annotation_prime.items():
            feats.append(annotation)
        return feats

    def run_analysis(self, universe, start, stop, step):
        pore_hydrat_ags = [universe.select_atoms(
            f"(cyzone 9 2 -2 {selection}) and resname TIP3 and name OH2", updating=True)
            for annotation, selection in pore_annotation_prime.items()]

        n_hydration = []
        for ts in universe.trajectory[start:stop:step]:
            n_hydration.append([pore_hydrat_ag.n_atoms for pore_hydrat_ag in pore_hydrat_ags])
        return n_hydration


class get_c_alpha_distance_10A(DaskChunkMdanalysis):
    name = 'ca_distance_10A'
    
    def set_feature_info(self, universe):
        pair_indices_union_df = pd.read_pickle('pair_indices_union_df.pickle')
        feat_info = []
        ag1 = universe.atoms[[]]
        ag2 = universe.atoms[[]]
        for subunit in range(5):
            for ind, row in pair_indices_union_df.iterrows():
                ag1 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a1_chain, row.a1_resid))
                ag2 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a2_chain, row.a2_resid))
            pair_indices_union_df = pair_indices_union_df.replace({"a1_chain": subunit_iter_dic})
            pair_indices_union_df = pair_indices_union_df.replace({"a2_chain": subunit_iter_dic}) 
            
        for ca_ag1, ca_ag2 in zip(ag1, ag2):
            feat_info.append(f'{ca_ag1.segid}_{ca_ag1.resid}_{ca_ag2.segid}_{ca_ag2.resid}')
        self.ag1_indices = ag1.indices
        self.ag2_indices = ag2.indices
        return feat_info

    def run_analysis(self, universe, start, stop, step):
        result = []
        ag1 = universe.atoms[self.ag1_indices]
        ag2 = universe.atoms[self.ag2_indices]
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(ag1, ag2)[2])
        return result



class get_c_alpha_distance_10A_2diff(DaskChunkMdanalysis):
    name = 'ca_distance_10A_2diff'
    
    def set_feature_info(self, universe):
        pair_indices_union_df = pd.read_pickle('pair_indices_union_df_2div.pickle')
        feat_info = []
        ag1 = universe.atoms[[]]
        ag2 = universe.atoms[[]]
        for subunit in range(5):
            for ind, row in pair_indices_union_df.iterrows():
                ag1 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a1_chain, row.a1_resid))
                ag2 += universe.select_atoms('name CA and segid {} and resid {}'.format(row.a2_chain, row.a2_resid))
            pair_indices_union_df = pair_indices_union_df.replace({"a1_chain": subunit_iter_dic})
            pair_indices_union_df = pair_indices_union_df.replace({"a2_chain": subunit_iter_dic}) 
            
        for ca_ag1, ca_ag2 in zip(ag1, ag2):
            feat_info.append(f'{ca_ag1.segid}_{ca_ag1.resid}_{ca_ag2.segid}_{ca_ag2.resid}')
        self.ag1_indices = ag1.indices
        self.ag2_indices = ag2.indices
        return feat_info

    def run_analysis(self, universe, start, stop, step):
        result = []
        ag1 = universe.atoms[self.ag1_indices]
        ag2 = universe.atoms[self.ag2_indices]
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(ag1, ag2)[2])
        return result


class get_acho_contact_from_92(DaskChunkMdanalysis):
    name = 'acho_dist_92'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['acho_dist_92_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        acho_ag = universe.select_atoms("resname ACHO and name N")
        protein_ag = universe.select_atoms("name CA and resid 92")

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(acho_ag, protein_ag)[2])
        return result

class get_acho_dist_from_54(DaskChunkMdanalysis):
    name = 'acho_dist_54'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['acho_dist_54_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        acho_ag = universe.select_atoms("resname ACHO and name N")
        protein_ag = universe.select_atoms("name CA and resid 54")
        protein_ag = AtomGroup(np.roll(protein_ag, -1))

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(dist(acho_ag, protein_ag)[2])
        return result


class get_acho_contact_from_92(DaskChunkMdanalysis):
    name = 'acho_contact_92'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['acho_contact_92_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        acho_ag = universe.select_atoms("resname ACHO").residues
        protein_ag = universe.select_atoms("protein and resid 92").residues

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(np.asarray([np.min(distance_array(acho.atoms, res.atoms)) for acho, res in zip(acho_ag, protein_ag)]))
        return result

class get_acho_contact_from_54(DaskChunkMdanalysis):
    name = 'acho_contact_54'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['acho_contact_54_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        acho_ag = universe.select_atoms("resname ACHO").residues
        protein_ag = universe.select_atoms("protein and resid 54").residues
        protein_ag = np.roll(protein_ag, -1)

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(np.asarray([np.min(distance_array(acho.atoms, res.atoms)) for acho, res in zip(acho_ag, protein_ag)]))
        return result

class get_acho_contact_from_187(DaskChunkMdanalysis):
    name = 'acho_contact_187'
    universe_file = 'system'

    def set_feature_info(self, universe):
        return ['acho_contact_187_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        acho_ag = universe.select_atoms("resname ACHO").residues
        protein_ag = universe.select_atoms("protein and resid 187").residues

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result.append(np.asarray([np.min(distance_array(acho.atoms, res.atoms)) for acho, res in zip(acho_ag, protein_ag)]))
        return result

class get_chol_contact_from_253(DaskChunkMdanalysis):
    name = 'chol_contact_253'
    universe_file = 'system'
    output = 'object'

    def set_feature_info(self, universe):
        return ['chol_contact_253_{}'.format(i) for i in range(5)]

    def run_analysis(self, universe, start, stop, step):
        chl1_ag = universe.select_atoms("resname CHL1").residues
        protein_ag = universe.select_atoms("protein and resid 253").residues

        result = []
        for ts in universe.trajectory[start:stop:step]:
            result_ts = []
            for res in protein_ag:
                result_ts.append(np.asarray([[chl1.resid, np.min(distance_array(chl1.atoms, res.atoms))] for chl1 in chl1_ag]))
            result.append(result_ts)
        return result