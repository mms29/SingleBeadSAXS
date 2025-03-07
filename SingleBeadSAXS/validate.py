
# import matplotlib
# matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import pyrosetta as pr
import numpy as np
import pandas as pd
from SingleBeadSAXS.rotamer_library import  restype_1to3 
from SingleBeadSAXS.formfactor import  calculate_distogram
import tqdm
import os 
import numpy as np

import numpy as np
from Bio import PDB

import os

def debye(dgram, q, ff, mask=None, eps=1e-6):
    # distance * q
    dq = dgram[..., None] * q[..., None, None, :]

    # ff in matrix form
    ffmat = (ff[..., None, :, :] * ff[..., :, None, :])

    # Output matrix
    F = np.zeros_like(ffmat)

    indices_zeros = dq<eps
    indices_nonzeros = ~indices_zeros
    F[indices_nonzeros] = ffmat[indices_nonzeros] * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
    F[indices_zeros] = ffmat[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)

    if mask is None: 
        return np.sum(F, axis=(-2,-3))
    else:
        return np.sum(F*mask[...,None], axis=(-2,-3))
    
def debye_mem(dgram, q, ff, eps=1e-6):
    F = np.zeros_like(ff)
    N = dgram.shape[0]

    for i in tqdm.tqdm(range(N)):
        dq = dgram[i][...,None] * q[..., None, :]
        indices_zeros = dq<eps
        indices_nonzeros = ~indices_zeros

        ff2 = ff * ff[i][None]

        F[indices_nonzeros] += ff2[indices_nonzeros]  * np.sin(dq [indices_nonzeros]) / dq[indices_nonzeros]
        F[indices_zeros] += ff2[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)

    return np.sum(F, axis=(-2))
    

def compute_center_of_mass(residue, atom_masses):
    """Compute the center of mass for a residue given atomic masses."""
    total_mass = 0
    com = np.zeros(3)  # Initialize center of mass

    for atom in residue:
        if atom.element != "H":  # Ignore hydrogen atoms
            mass = atom_masses.get(atom.element, 12.0)  # Default to carbon mass if unknown
            com += atom.coord * mass
            total_mass += mass

    return com / total_mass if total_mass > 0 else com  # Avoid division by zero

def get_ca_coordinates_and_residues(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ca_coords = []
    cm_coords = []
    residues = []

    atom_masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065}

    for model in structure:  # Usually, there's only one model
        for chain in model:  # Iterate over chains
            for residue in chain:  # Iterate over residues
                if residue.id[0] == " ":
                    if "CA" in residue:  # Check if C-alpha exists
                        cm_coords.append(compute_center_of_mass(residue, atom_masses))  # Extract coordinates
                        ca_coords.append(residue["CA"].coord)  # Extract coordinates
                        residues.append(residue.resname)  # Extract residue name

    # Convert list of coordinates to a NumPy array
    ca_coords = np.array(ca_coords)
    cm_coords = np.array(cm_coords)

    return ca_coords,cm_coords, residues


def profile_singlebead(pdb_file, max_q = 0.5, memory_efficient=False, center_of_mass=True, ff_data_path = "/home/vuillemr/Workspace/SAXS/outputs/single_beads_ff.csv"):

    ca_coords,cm_coords, residues = get_ca_coordinates_and_residues(pdb_file)

    ff_data = pd.read_csv(ff_data_path)
    q_singlebead = ff_data["q"].to_numpy()
    indices = (q_singlebead<=max_q) * (np.arange(len(q_singlebead))%5 == 0)
    q_singlebead= q_singlebead[indices].astype(np.float32)
    default_res="PRO"
    ff = np.array([
            ff_data[n+"_AA_uncor"][indices] if n in restype_1to3 else 
            ff_data[default_res+"_AA_uncor"][indices] 
            for n in residues]).astype(np.float32)
    
    if center_of_mass:
        dgram = calculate_distogram(cm_coords)
    else:
        dgram = calculate_distogram(ca_coords)

    if memory_efficient:
        I_singlebead = debye_mem(dgram, q_singlebead, ff)
    else:
        I_singlebead = debye(dgram, q_singlebead, ff)

    return pd.DataFrame({"q" : q_singlebead, "I" : I_singlebead})


def profile_pepsi(pdb_file):
    pepsi_out_file = pdb_file+".txt"
    os.system("%s %s -o %s"%(PEPSI_SAXS_PATH, pdb_file, pepsi_out_file))
    pepsi_curve = pd.read_csv(pepsi_out_file, delim_whitespace=True, comment='#', names=["q", "I", "Iat", "Iev","Ihs","AatAev" ,"AatAhs","AevAhs"] )
    pepsi_log = pepsi_out_file[:-4]+".log"
    with open( pepsi_log, "r") as f:
        for l in f:
            if "Intensity scaling" in l:
                pepsi_scaling = float(l.split()[-1])

    for i in list(pepsi_curve.keys()):
        if i !="q":
            pepsi_curve[i]/=pepsi_scaling

    os.system("rm -f %s"%pepsi_log)
    os.system("rm -f %s"%pepsi_out_file)
    return pepsi_curve
    
def profile_foxs(pdb_file):
    os.system("%s foxs %s"%(FOXS_ENV_COMMAND, pdb_file))
    outfile = pdb_file + ".dat"
    foxs_curve = pd.read_csv(outfile, delim_whitespace=True, comment='#', names= ["q"  ,  "I"   , "Ierr"])

    os.system("rm -f %s"%outfile)
    return foxs_curve
    

PEPSI_SAXS_PATH = "/home/vuillemr/Workspace/SAXS/Pepsi-SAXS"
FOXS_ENV_COMMAND = """
    eval "$(/home/vuillemr/miniconda3/condabin/conda shell.bash hook)" &&
    conda activate imp &&
"""

pdb_file= "/home/vuillemr/Workspace/SAXS/validate/6z6u.pdb"

pepsi_curve = profile_pepsi(pdb_file)
foxs_curve = profile_foxs(pdb_file)
singlebead_curve = profile_singlebead(pdb_file, memory_efficient=True,center_of_mass=True)

kwargs = {"ls" : "-", "marker" : None, "linewidth":1.0}

fig, ax = plt.subplots(1,1, layout="constrained", figsize=(6,4))
ax.plot(pepsi_curve.q,pepsi_curve.I, label="Pepsi-SAXS", **kwargs)
ax.plot(foxs_curve.q,foxs_curve.I, label="FoXS", **kwargs)
ax.plot(singlebead_curve.q,singlebead_curve.I, label="SingleBead", **kwargs)
ax.set_yscale("log")
ax.set_xlabel("q (1/$\AA$)", fontsize=14)
ax.set_ylabel("I(q)", fontsize=14)
ax.legend()
fig.show()









