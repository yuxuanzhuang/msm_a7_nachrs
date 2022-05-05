# Import the library
import MDAnalysis as mda
from collections import OrderedDict
import subprocess
import glob
from MDAnalysis.analysis import align
import argparse

import os
import sys
import py3Dmol
import nglview as nv
import shutil
import gromacs
import gromacs.run


class MDrunner_d(gromacs.run.MDrunner):
    mdrun = "/opt/tcbsys/gromacs/2021.5/gmx_d/AVX2_256/bin/gmx_d mdrun"


# Create the parser
parser = argparse.ArgumentParser()
# Add an argument

parser.add_argument('--system', type=str, required=True)
parser.add_argument('--seed', type=str, required=True)
parser.add_argument(
    '--working_dir',
    type=str,
    required=False,
    default=os.getcwd())
parser.add_argument('--name', type=str, required=False, default='system')
parser.add_argument(
    '--starting_pdb',
    type=str,
    required=False,
    default='pdb2gmx.pdb')
parser.add_argument(
    '--MembraneType',
    type=str,
    required=False,
    default="POPC:CHL1")
parser.add_argument('--gpu_id', type=int, required=False, default=0)
parser.add_argument('--seed_loc', type=str, required=False, default='SEEDS')


# Parse the argument
args = parser.parse_args()

seed_loc = args.seed_loc
# In[1]:


# Copy from
# https://github.com/pstansfeld/MemProtMD/blob/main/MemProtMD_insane.ipynb


# In[2]:


visualization = False
name = args.name
starting_pdb = args.working_dir + '/../' + seed_loc + '/' + \
    args.system + '/' + args.seed + '/' + args.starting_pdb

Box_Width = 13
Box_Length = 16.6
cube = [Box_Width, Box_Width, Box_Length]

working_dir = args.working_dir + '/../' + seed_loc + \
    '/' + args.system + '/' + args.seed + '/' + name + '/'
os.makedirs(working_dir, exist_ok=True)
shutil.copyfile(starting_pdb, working_dir + '/' + args.starting_pdb)

# In[3]:


MembraneType = args.MembraneType
#MembraneType = "POPC"
if MembraneType == "POPC":
    lipid = '-l POPC:1'
elif MembraneType == "POPE:POPG":
    lipid = '-l POPE:7 -l POPG:3'
elif MembraneType == "POPE:POPG:CARDIOLIPIN":
    lipid = '-l POPE:7 -l POPG:2 -l CARD:1'
elif MembraneType == "POPC:CHL1":
    lipid = '-l POPC:2 -l CHOL:1'


# In[4]:


if visualization:
    with open(starting_pdb) as ifile:
        mol1 = "".join([x for x in ifile])

    mview = py3Dmol.view(width=800, height=400)
    mview.addModel(mol1, 'pdb')
    mview.setStyle({'cartoon': {'color': 'spectrum'}})
    mview.setStyle({'resn': 'DUM'}, {'sphere': {}})
    mview.setBackgroundColor('0xffffff')
    mview.zoomTo()
    mview.show()


# In[5]:


# find TMD region
os.system(
    '/nethome/yzhuang/git_repo/memembed/bin/memembed -q 1 -o ' +
    working_dir +
    '/memembed.pdb ' +
    starting_pdb)


# In[6]:


if visualization:
    mview = py3Dmol.view(width=800, height=400)
    mol1 = open(working_dir + '/memembed.pdb', 'r').read()
    mview.addModel(mol1, 'pdb')
    mview.setStyle({'cartoon': {'color': 'spectrum'}})
    mview.setStyle({'resn': 'DUM'}, {'sphere': {}})
    mview.setBackgroundColor('0xffffff')
    mview.zoomTo()
    mview.show()


# In[7]:


for file in glob.glob(r'./common_files/*'):
    if not os.path.isdir(file):
        shutil.copy(file, working_dir)
    else:
        shutil.copytree(
            file,
            working_dir +
            '/' +
            os.path.basename(file),
            dirs_exist_ok=True)

#shutil.copytree('./common_files', working_dir)

os.chdir(working_dir)
gromacs.make_ndx(
    f='memembed.pdb',
    o='index.ndx',
    input=(
        'del 0',
        'del 1-100',
        'rDUM',
        'q'),
    backup=False)
gromacs.editconf(
    f='memembed.pdb',
    o='centered.pdb',
    n='index.ndx',
    c=True,
    input=(
        0,
        0),
    box=cube,
    backup=False)

u = mda.Universe('centered.pdb')
x = round(u.dimensions[0] / 10)
y = round(u.dimensions[1] / 10)
z = round(u.dimensions[2] / 10)

gromacs.confrms(
    f2='memembed.pdb',
    f1='centered.pdb',
    one=True,
    o='aligned.gro',
    input=(
        3,
        3),
    backup=False)
gromacs.editconf(
    f='aligned.gro',
    o='protein.pdb',
    label='A',
    resnr=1,
    n=True,
    input=(
        0,
        0),
    backup=False)

v = mda.Universe('aligned.gro')
dum = v.select_atoms('resname DUM')

dm = (z / 2) - (round(dum.center_of_mass()[2] / 10))

with open('protein.pdb', 'r') as file:
    filedata = file.read()
filedata = filedata.replace('HSE', 'HIS')
filedata = filedata.replace('HSD', 'HIS')
# some weird atom name in Climber
filedata = filedata.replace('HISD', 'HIS ')

filedata = filedata.replace('MSE', 'MET')
filedata = filedata.replace(' SE ', ' SD ')
with open('protein.pdb', 'w') as file:
    file.write(filedata)


# In[11]:


os.system('/nethome/yzhuang/anaconda3/envs/deeplearning/bin/martinize2 -f protein.pdb -dssp mkdssp -ff martini3001 -x protein-cg.pdb -o protein-cg.top -elastic -ef 500 -eu 1.0 -el 0.5 -ea 0 -ep 0 -merge A -p backbone -maxwarn 100000 -scfix')


# In[12]:


u = mda.Universe('protein-cg.pdb')
u_ref = mda.Universe('prot_chol.pdb')


# In[13]:


align.AlignTraj(
    u_ref,
    u,
    select="name BB",
    weights="mass",
    in_memory=True).run()
print(
    mda.analysis.rms.rmsd(
        u_ref.select_atoms('name BB').positions,
        u.select_atoms('name BB').positions))
u_merge = mda.core.universe.Merge(u.atoms, u_ref.select_atoms('resname CHOL'))
u_merge.atoms.write('protein-cg-chol.pdb')
os.system("sed -e 's/^molecule.*/Protein 1/g' molecule*.itp >  protein-cg.itp")


# In[14]:


os.system(
    'python2 insane3.py ' +
    lipid +
    ' -salt 0.15 -sol W -o CG-system.gro -p topol.top -f protein-cg-chol.pdb -center -x %s -y %s -z %s -dm %s' %
    (x,
     y,
     z,
     dm))


# In[15]:


replacements = {'NA+': 'NA',
                'CL-': 'CL',
                '#include "martini_v3.itp"': '#include "martini_v3.0.0.itp"\n#include "martini_v3.0.0_ions_v1.itp"\n#include "martini_v3.0.0_solvents_v1.itp"\n#include "martini_v3.0.0_phospholipids_v1.itp"\n#include "martini_v3.0_sterols_beta.itp"\n#include "chol.itp"\n',
                'Protein        1\n': 'Protein        1\nCHOL_PORE        5\n'}
lines = []
with open('topol.top') as infile:
    for line in infile:
        for src, target in replacements.items():
            line = line.replace(src, target)
        lines.append(line)
with open('topol.top', 'w') as outfile:
    for line in lines:
        outfile.write(line)


# In[16]:


gromacs.grompp(
    f='em.mdp',
    o='em.tpr',
    c='CG-system.gro',
    maxwarn='-1',
    backup=False,
    v=True)

# gromacs.mdrun(deffnm='em', cpi='em', c='CG-system.pdb',backup=False, ntmpi=1, ntomp=12)
mdrun_em = MDrunner_d(
    deffnm='em',
    cpi='em',
    c='CG-system.pdb',
    backup=False,
    ntmpi=1,
    ntomp=12)
mdrun_em.run()

gromacs.trjconv(
    f='CG-system.pdb',
    o='CG-system.pdb',
    pbc='res',
    s='em.tpr',
    conect=True,
    input='0',
    backup=False)

os.makedirs('MD', exist_ok=True)

u = mda.Universe('CG-system.pdb')
with mda.selections.gromacs.SelectionWriter('index.ndx', mode='w') as ndx:
    ndx.write(u.select_atoms('protein'), name='Protein')
    ndx.write(
        u.select_atoms('not protein and not resname W ION'),
        name='Lipid')
    ndx.write(u.select_atoms('resname W ION'), name='SOL_ION')

#gromacs.make_ndx(f='CG-system.pdb', o='index.ndx', input=('del 0', 'del 1-40', '0|rPOP*','1&!0','!1','del 1','name 1 Lipid','name 2 SOL_ION','q'),backup=False)

gromacs.grompp(
    f='cgmd.mdp',
    o='MD/md',
    c='CG-system.pdb',
    r='CG-system.pdb',
    maxwarn=-1,
    n='index.ndx',
    backup=False)


# In[17]:


os.chdir('MD')

gromacs.mdrun(
    deffnm='md',
    cpi='md',
    backup=False,
    nsteps=400000,
    ntmpi=1,
    ntomp=12,
    gpu_id=str(
        args.gpu_id))

if visualization:
    mview = py3Dmol.view(width=800, height=400)
    mol1 = open(working_dir + 'MD/md.gro', 'r').read()
    mview.addModel(mol1, 'gro')
    mview.setStyle({'cartoon': {'color': 'spectrum'}})
    mview.setStyle({'atom': 'PO4'}, {'sphere': {}})
    mview.setStyle({'atom': 'BB'}, {'sphere': {}})
    mview.setBackgroundColor('0xffffff')
    mview.zoomTo()
    mview.show()

for file in glob.glob(r'#*'):
    os.remove(file)

for file in glob.glob(r'{working_dir}#*'):
    os.remove(file)


# In[18]:


cg2at_command = [
    '/nethome/yzhuang/anaconda3/envs/deeplearning/bin/python',
    '/nethome/yzhuang/git_repo/cg2at/cg2at',
    '-o',
    'align',
    '-w',
    'tip3p',
    '-c',
    'md.gro',
    '-ncpus',
    '12',
    '-gpu_id',
    str(args.gpu_id),
    '-a',
    starting_pdb,
    '-loc',
    'CG2AT',
    '-ff',
    'charmm36-jul2020-updated',
    '-fg',
    'martini_3-0_charmm36',
    '-ter']
print(cg2at_command)
process = subprocess.Popen(cg2at_command,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           universal_newlines=True,
                           bufsize=0,
                           cwd=working_dir + 'MD/')
process.stdin.write("0\n")
process.stdin.write("1\n")
process.stdin.write("0\n")
process.stdin.write("1\n")
process.stdin.write("0\n")
process.stdin.write("1\n")
process.stdin.write("0\n")
process.stdin.write("1\n")
process.stdin.write("0\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")
process.stdin.write("1\n")

process.stdin.close()

for line in process.stdout:
    print(line.strip())
for line in process.stderr:
    print(line.strip())


# In[19]:


# os.system('/nethome/yzhuang/git_repo/cg2at/cg2at -o align -w tip3p -c md.gro -a ' + working_dir + starting_pdb + ' -loc CG2AT -ff charmm36-jul2020-updated -fg martini_3-0_charmm36 -ter')
os.rename(working_dir + 'MD/md.gro', working_dir + name + '-cgmd.gro')
os.rename(working_dir + 'MD/md.tpr', working_dir + name + '-cgmd.tpr')
os.rename(working_dir + 'memembed.pdb', working_dir + name + '-memembed.pdb')
os.chdir(working_dir)
shutil.rmtree('Atomistic', ignore_errors=True)
shutil.copytree(working_dir + 'MD/CG2AT/FINAL/', working_dir + 'Atomistic/')
os.chdir(working_dir + 'Atomistic/')
os.rename(working_dir + 'Atomistic/final_cg2at_aligned.pdb',
          working_dir + 'Atomistic/' + name + '-System.pdb')
os.rename(working_dir + 'Atomistic/topol_final.top',
          working_dir + 'Atomistic/topol.top')
# gromacs.make_ndx(f=working_dir + 'Atomistic/' + name + '-System.pdb', o='index.ndx', input=('del 2-40', 'rSOL|rNA*|rCL*','1|2','0&!3','del 3','name 2 water_and_ions','name 3 Lipid','q'),backup=False)
with open(working_dir + 'Atomistic/topol.top', 'r') as file:
    filedata = file.read()
filedata = filedata.replace('../FINAL/', '')
with open(working_dir + 'Atomistic/topol.top', 'w') as file:
    file.write(filedata)


# In[20]:


if visualization:
    mview = py3Dmol.view(width=800, height=400)
    mol1 = open(working_dir + 'Atomistic/' + name + '-System.pdb', 'r').read()
    mview.addModel(mol1, 'pdb')
    mview.setStyle({'cartoon': {'color': 'spectrum'}})
    mview.setStyle({'atom': 'P'}, {'sphere': {}})
    mview.setBackgroundColor('0xffffff')
    mview.zoomTo()
    mview.show()


# In[21]:


os.makedirs(working_dir + 'FINAL', exist_ok=True)
u = mda.Universe(working_dir + 'Atomistic/' + name + '-System.pdb')
com_popc_z = u.select_atoms('resname POPC').center_of_mass()[2]
memb_water = u.select_atoms(
    f'resname SOL and same residue as (resname SOL and prop z <= {com_popc_z + 19} and prop z >= {com_popc_z - 19})')
# memb_water
# memb_water.groupby('resnames')
u.select_atoms(
    'not group memb_water',
    memb_water=memb_water).write(
        working_dir +
        'FINAL/' +
        name +
    '-prot_memb.pdb')
os.chdir(working_dir + 'FINAL/')
u = mda.Universe(working_dir + 'FINAL/' + name + '-prot_memb.pdb')
#view = nv.show_mdanalysis(u)
#view.camera = 'orthographic'
#view.add_trajectory(u.select_atoms('resname POPC CHOL'))


# In[14]:


# view


# In[22]:


for file in glob.glob(working_dir + r'/at/*'):
    if not os.path.isdir(file):
        shutil.copy(file, working_dir + 'FINAL/')
    else:
        shutil.copytree(
            file,
            working_dir +
            'FINAL/charmm36.ff',
            dirs_exist_ok=True)
shutil.copy(working_dir + 'Atomistic/topol.top', './topol_init.top')
mol_sec = False
mol_dict = OrderedDict()
with open(working_dir + 'FINAL/topol_init.top', 'r') as file:
    for line in file.readlines():
        if line == '\n':
            mol_sec = False
        if mol_sec:
            mol_dict[line.split()[0]] = line.split()[1]
        if line.find('; Compound') != -1:
            mol_sec = True

mol_dict['TIP3P'] = str(eval(mol_dict['TIP3P']) - memb_water.n_atoms // 3)

other_mols = ['POPC', 'CHOL', 'NA', 'CL', 'TIP3P']
cg2at_dic = {'POPC': 'POPC',
             'CHOL': 'CHL1',
             'NA': 'SOD',
             'CL': 'CLA',
             'TIP3P': 'TIP3'}

with open(working_dir + 'FINAL/topol_test.top', 'a') as file:
    for mol in mol_dict:
        if mol in other_mols:
            file.write(cg2at_dic[mol] + '   ' + mol_dict[mol] + '\n')


# In[38]:


u_aligned = mda.Universe(starting_pdb)
align.AlignTraj(
    u_aligned,
    u,
    select="name CA",
    weights="mass",
    in_memory=True).run()
print(
    mda.analysis.rms.rmsd(
        u_aligned.select_atoms('name CA').positions,
        u.select_atoms('name CA').positions))
u_merge = mda.core.universe.Merge(
    u_aligned.atoms,
    u.select_atoms('not protein'))
u_merge.dimensions = u.dimensions
u_merge.atoms.write(name + '-test.pdb')


# In[44]:


gromacs.grompp(
    f='em.mdp',
    o='em_test.tpr',
    p='topol_test.top',
    c=name +
    '-test.pdb',
    r=name +
    '-test.pdb',
    maxwarn='-1',
    backup=False,
    v=True)
u_tpr = mda.Universe('em_test.tpr')

# In[60]:


u = mda.Universe(name + '-test.pdb')
n_charge = round(u_tpr.atoms.total_charge())
if n_charge > 0:
    remove_sod = u.select_atoms('resname NA')[:n_charge]
    u.select_atoms(
        'not group remove_sod',
        remove_sod=remove_sod).write(
        name +
        '-final.pdb')
    mol_dict['NA'] = str(eval(mol_dict['NA']) - n_charge)
elif n_charge < 0:
    remove_cla = u.select_atoms('resname CL')[:abs(n_charge)]
    u.select_atoms(
        'not group remove_cla',
        remove_cla=remove_cla).write(
        name +
        '-final.pdb')
    mol_dict['CL'] = str(eval(mol_dict['CL']) - n_charge)

with open(working_dir + 'FINAL/topol.top', 'a') as file:
    for mol in mol_dict:
        if mol in other_mols:
            file.write(cg2at_dic[mol] + '   ' + mol_dict[mol] + '\n')


# In[64]:


gromacs.grompp(
    f='em.mdp',
    o='em.tpr',
    p='topol.top',
    c=name +
    '-final.pdb',
    r=name +
    '-final.pdb',
    maxwarn='-1',
    backup=False,
    v=True)


# In[66]:


#gromacs.mdrun_d(deffnm='em', cpi='em', c='start.pdb',backup=False, )
mdrun_em_aa = MDrunner_d(
    deffnm='em',
    cpi='em',
    c='start.pdb',
    backup=False,
    ntmpi=1,
    ntomp=12)
mdrun_em_aa.run()


# In[65]:


u = mda.Universe('start.pdb')
with mda.selections.gromacs.SelectionWriter('index.ndx', mode='w') as ndx:
    ndx.write(u.atoms, name='SYSTEM')
    ndx.write(u.select_atoms('protein'), name='PROT')
    ndx.write(
        u.select_atoms('not protein and not resname SOD CLA TIP3'),
        name='MEMB')
    ndx.write(u.select_atoms('resname SOD CLA TIP3'), name='SOL_ION')
