# adapted from pentest.py by CHARMM-GUI

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import networkx as nx
import re
import numpy as np
import MDAnalysis.transformations as trans
import MDAnalysis as mda

_s = re.compile('\s+')
_p = re.compile('(\d+)\s+(\d+)')

def lsqp(atoms):
    com = atoms.mean(axis=0)
    #u, d, v = np.linalg.svd(atoms-com)

    axes = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        p1 = atoms[i]
        if i == len(atoms)-1:
            p2 = atoms[0]
        else:
            p2 = atoms[i+1]
        a = np.cross(p1, p2)
        axes += a
    u, d, v = np.linalg.svd(axes)
    i = 0
    d = -np.dot(v[i], com)
    n = -np.array((v[i,0], v[i,1], d))/v[i,2]
    return v[i], com, n

def intriangle(triangle, axis, u, p):
    # http://www.softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
    p1, p2, p3 = triangle
    w0 = p - p1
    a = -np.dot(axis, w0)
    b = np.dot(axis, u)
    if (abs(b) < 0.01): return False

    r = a / b
    if r < 0.0: return False
    if r > 1.0: return False

    I = p + u * r

    u = p2 - p1
    v = p3 - p1
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    w = I - p1
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv * uv - uu * vv

    s = (uv * wv - vv * wu)/D
    if (s < 0 or s > 1): return False
    t = (uv * wu - uu * wv)/D
    if (t < 0 or (s+t) > 1): return False
    return True

def build_topology(universe, selection):
    g = nx.Graph()

    #  Atoms
    natom = universe.atoms.n_atoms
    for atom in universe.select_atoms(selection).atoms:  #  might be buggy
        g.add_node(atom.index + 1, **{'segid': atom.segid,
                                      'resname': atom.resname,
                                      'name': atom.name,
                                      'resid': atom.resid})
    #  Bonds
    for bond in universe.select_atoms(selection).bonds:
        num1, num2 = bond.atoms.indices + 1
        if g.has_node(num1) and g.has_node(num2):
            g.add_edge(num1, num2)
    return g

def check_ring_penetration(top, coord, pbc=[], xtl='rect', verbose=0):
    # ring penetration test
    # 1. find rings
    # 2. build least square plane
    # 3. project atoms ring constituent atoms onto the plane and build convex
    # 4. find two bonded atoms that are at the opposite side of the plane
    # 5. determine the point of intersection is enclosed in the ring
    #
    from networkx.algorithms.components import connected_components
    molecules =  (top.subgraph(c) for c in connected_components(top))

    allatoms = np.array([coord[num] for num in top.nodes()])
    atoms_map = np.array([num for num in top.nodes()])
    natoms = len(allatoms)
    if pbc:
        atoms_map_reverse = {}
        for i,num in enumerate(top.nodes()):
            atoms_map_reverse[num] = i

        a = float(pbc[0])
        b = float(pbc[1])
        n = len(allatoms)
        if xtl == 'rect':
            allatoms = np.tile(allatoms, (9,1))
            op = ((a,0),(a,b),(0,b),(-a,b),(-a,0),(-a,-b),(0,-b),(a,-b))
            for i in range(8):
                x,y = op[i]
                allatoms[n*(i+1):n*(i+2),0] += x
                allatoms[n*(i+1):n*(i+2),1] += y
            atoms_map = np.tile(atoms_map, 9)
        if xtl =='hexa':
            allatoms = np.tile(allatoms, (7,1))
            rot = lambda theta: np.matrix(((np.cos(np.radians(theta)), -np.sin(np.radians(theta))),
                                           (np.sin(np.radians(theta)),  np.cos(np.radians(theta)))))
            op = (rot(15), rot(75), rot(135), rot(195), rot(255), rot(315))
            d = np.array((a, 0))
            for i in range(6):
                xy = np.dot(d, op[i])
                allatoms[n*(i+1):n*(i+2),:2] = allatoms[n*(i+1):n*(i+2),:2] + xy
            atoms_map = np.tile(atoms_map, 7)

    # print out image atoms
    #fp = open('image.pdb', 'w')
    #for i,atom in enumerate(allatoms):
    #    x, y, z = atom
    #    fp.write("HETATM%5d  %-3s %3s  %4d    %8.3f%8.3f%8.3f  0.00  0.00      \n" % (i, 'C', 'DUM', i, x, y, z))

    pen_pairs = []
    pen_cycles = []

    for m in molecules:
        cycles = nx.cycle_basis(m)
        if not cycles: continue
        for cycle in cycles:
            flag = False
            atoms = np.array([coord[num] for num in cycle])
            if len(set([top.nodes[num]['resid'] for num in cycle])) > 1: continue
            if verbose:
                num = cycle[0]
                print('found ring:', top.nodes[num]['segid'], top.nodes[num]['resid'], top.nodes[num]['resname'])

            # build least square fit plane
            axis, com, n = lsqp(atoms)

            # project atoms to the least square fit plane
            for i,atom in enumerate(atoms):
                w = np.dot(axis, atom-com)*axis + com
                atoms[i] = com + (atom - w)

            maxd = np.max(np.sqrt(np.sum(np.square(atoms - com), axis=1)))

            d = np.sqrt(np.sum(np.square(allatoms-com), axis=1))
            nums = np.squeeze(np.argwhere(d < 3))

            # find two bonded atoms that are at the opposite side of the plane
            for num in nums:
                num1 = atoms_map[num]

                for num2 in top[num1]:
                    if num1 in cycle or num2 in cycle: continue
                    if num > natoms:
                        # image atoms
                        offset = int(num / natoms)
                        coord1 = allatoms[num]
                        coord2 = allatoms[atoms_map_reverse[num2] + offset * natoms]
                    else:
                        coord1 = coord[num1]
                        coord2 = coord[num2]

                    v1 = np.dot(coord1 - com, axis)
                    v2 = np.dot(coord2 - com, axis)
                    if v1 * v2 > 0: continue

                    # point of intersection of the least square fit plane
                    s = -np.dot(axis, coord1-com)/np.dot(axis, coord2-coord1)
                    p = coord1 + s*(coord2-coord1)

                    d = np.sqrt(np.sum(np.square(p-com)))
                    if d > maxd: continue
                    if verbose:
                        print('found potentially pentrarting bond:',
                              top.nodes[num1]['segid'],
                              top.nodes[num1]['resid'],
                              top.nodes[num1]['resname'],
                              top.nodes[num1]['name'],
                              top.nodes[num2]['name'])

                    d = 0
                    for i in range(0, len(atoms)):
                        p1 = atoms[i] - p
                        try: p2 = atoms[i+1] - p
                        except: p2 = atoms[0] - p
                        d += np.arccos(np.dot(p1, p2)/np.linalg.norm(p1)/np.linalg.norm(p2))

                    wn = d/2/np.pi
                    if wn > 0.9 and wn < 1.1:
                        # we have a case
                        pen_pairs.append((num1, num2))
                        pen_cycles.append(cycle)
                        flag = True
                        break

                if flag: break

    return pen_pairs, pen_cycles



def check_universe_ring_penetration(universe,
                                    selection='not resname TIP3 and not (name H*)',
                                    verbose=False):
    output_text = []

    top = build_topology(universe, selection)
    ag = universe.select_atoms(selection)
    for frame, ts in enumerate(universe.trajectory):
        output_text.append('In frame %d' % frame)
        coord = dict(zip(ag.ids + 1, ag.positions))
        if len(top.nodes()) != len(coord):
            raise ValueError('Number of atoms does not match')
        #  only rect pbc have been tested
        pairs, rings = check_ring_penetration(top, coord, verbose=verbose)
        if pairs:
            output_text.append('found a ring penetration:')
            output_text.append('segid index resid resname atomname')

            for i, cycle in enumerate(rings):
                output_text.append('- %s %s %s %s %s | %s %s %s %s %s' % (
                top.nodes[pairs[i][0]]['segid'],
                top.nodes[pairs[i][0]]['index'],
                top.nodes[pairs[i][0]]['resid'],
                top.nodes[pairs[i][0]]['resname'],
                ' '.join([top.nodes[num]['name'] for num in pairs[i]]),
                top.nodes[cycle[0]]['segid'],
                top.nodes[cycle[0]]['index'],
                top.nodes[cycle[0]]['resid'],
                top.nodes[cycle[0]]['resname'],
                ' '.join([top.nodes[num]['name'] for num in cycle])))
    return output_text


def main():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", default=None, required=True,
                        help="trajectory or pdb file")
    parser.add_argument("--topology", default=None, required=True,
                        help="topology file that contains boned information e.g. tpr")
    parser.add_argument("--structure", default=None, required=False,
                        help="structure file e.g. gro or pdb format")
    parser.add_argument("-s", "--selection", default="not resname TIP3 and not (name H*)",
                        help="selection for atoms to check for pentration")
    parser.add_argument("-o", "--output", default=None,
                        help="output file")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    args = parser.parse_args()

    if args.verbose:
        print('checking pentration of %s' % args.selection)

    if args.structure:
        universe = mda.Universe(args.structure, args.trajectory)
        universe_bonded = mda.Universe(args.topology)
        universe.add_bonds(universe_bonded.bonds.to_indices())
    else:
        universe = mda.Universe(args.topology, args.trajectory)

    # unwrap/make_whole all selected atoms.
    universe.trajectory.add_transformations(trans.unwrap(universe.select_atoms(args.selection)))

    output_text = check_universe_ring_penetration(universe, args.selection, verbose=args.verbose)

    if args.output:
        fp = open(args.output, 'w')
        for line in output_text:
            fp.write(line + '\n')
        fp.close()
    else:
        for line in output_text:
            print(line)
        if len(output_text) == 1:
            print('No pentration found')

if __name__ == '__main__':
    sys.exit(main())