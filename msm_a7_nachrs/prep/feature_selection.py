from ENPMDA.analysis import *
from ENPMDA.analysis.base import DaskChunkMdanalysis

from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.analysis.distances import self_distance_array

import MDAnalysis as mda
import itertools

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
            r_ts = []
            for feature_chain in [[0, 1, 2], [1, 2, 3],
                                  [2, 3, 4], [3, 4, 0], [4, 0, 1]]:
                for domain_ag1, domain_ag2 in itertools.product(
                        domain_ag_list[feature_chain[0]], domain_ag_list[feature_chain].ravel()):
                    if domain_ag1 != domain_ag2:
                        r_ts.append(calc_bonds(domain_ag1.center_of_geometry(),
                                               domain_ag2.center_of_geometry()))
            result.append(r_ts)

        return result


class get_domain_inverse_interdistance(DaskChunkMdanalysis):
    name = 'domain_inverse_distance'

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
        domain_info_list = []
        for domain_name, selection in domain_dict.items():
            for domain_subunit in universe.select_atoms(
                    selection).split('segment'):
                domain_ag_list.append(domain_subunit)
                domain_info_list.append(
                    domain_name + '_' + domain_subunit.segids[0] + '_pos')

        domain_ag_list = np.asarray(
            domain_ag_list, dtype=object).reshape(
            len(domain_dict), 5).T
        domain_info_list = np.asarray(
            domain_info_list, dtype=object).reshape(
            len(domain_dict), 5).T

        self._feature_info = []
        for feature_chain in [[0, 1, 2], [1, 2, 3],
                              [2, 3, 4], [3, 4, 0], [4, 0, 1]]:
            for d1_inf, d2_inf in itertools.product(
                    domain_info_list[feature_chain[0]], domain_info_list[feature_chain].ravel()):
                if d1_inf != d2_inf:
                    self._feature_info.append('_'.join([d1_inf, d2_inf]))

        result = []
        for ts in universe.trajectory[start:stop:step]:
            r_ts = []
            for feature_chain in [[0, 1, 2], [1, 2, 3],
                                  [2, 3, 4], [3, 4, 0], [4, 0, 1]]:
                for domain_ag1, domain_ag2 in itertools.product(
                        domain_ag_list[feature_chain[0]], domain_ag_list[feature_chain].ravel()):
                    if domain_ag1 != domain_ag2:
                        r_ts.append(1.0 / calc_bonds(domain_ag1.center_of_geometry(),
                                                     domain_ag2.center_of_geometry()))
            result.append(r_ts)

        return result


class get_domain_intradistance(DaskChunkMdanalysis):
    name = 'domain_inverse_intra_distance'

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
            r_ts = []
            for feature_chain in range(5):
                for domain_ag1, domain_ag2 in itertools.product(
                        domain_ag_list[feature_chain], domain_ag_list[feature_chain].ravel()):
                    if domain_ag1 != domain_ag2:
                        r_ts.append(1.0 / calc_bonds(domain_ag1.center_of_geometry(),
                                                     domain_ag2.center_of_geometry()))
            result.append(r_ts)

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


class get_c_alpha_distance_filtered(DaskChunkMdanalysis):
    name = 'ca_distance_filtered'

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
                for i, j in filtered_candidatess:
                    filtered_distance_matrix.append(distance_matrix[i, j])

                r_ts.extend(filtered_distance_matrix)
            result.append(r_ts)

        self._feature_info = []
        for selection1, selection2 in selection_comb:
            for feature_chain in feature_list:
                self._feature_info.append(
                    selection1.split()[0] + '_' + feature_chain)

        return result


class get_c_alpha_distance_filtered_inverse(DaskChunkMdanalysis):
    name = 'inverse_ca_distance_filtered'

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
                for i, j in filtered_candidatess:
                    filtered_distance_matrix.append(
                        1.0 / distance_matrix[i, j])

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
