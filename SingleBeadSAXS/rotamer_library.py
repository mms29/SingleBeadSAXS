
import sys
import matplotlib.pyplot as plt
import tqdm
import collections
import os
import pyrosetta as pr
import numpy as np
import pandas as pd

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restypes3 = [v for v in restype_1to3.values()]

def load_rotamor_library(libpath, libname = "ExtendedOpt1-5"):
    # Loads the rotamor library
    amino_acids = [
        "arg",
        "asp",
        "asn",
        "cys",
        "glu",
        "gln",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
    db = {}

    columns = collections.OrderedDict()
    columns["T"] = str
    columns["Phi"] = np.int64
    columns["Psi"] = np.int64
    columns["Count"] = np.int64
    columns["r1"] = np.int64
    columns["r2"] = np.int64
    columns["r3"] = np.int64
    columns["r4"] = np.int64
    columns["Probabil"] = np.float64
    columns["chi1Val"] = np.float64
    columns["chi2Val"] = np.float64
    columns["chi3Val"] = np.float64
    columns["chi4Val"] = np.float64
    columns["chi1Sig"] = np.float64
    columns["chi2Sig"] = np.float64
    columns["chi3Sig"] = np.float64
    columns["chi4Sig"] = np.float64

    for amino_acid in amino_acids:
        db[amino_acid] = pd.read_csv(
            os.path.join(libpath, f"{libname}/{amino_acid}.bbdep.rotamers.lib"),
            names=list(columns.keys()),
            dtype=columns,
            comment="#",
            delim_whitespace=True,
            engine="c",
        )

    return db

def names_from_pose(pose, element_or_name=True):
    res= pose.residue(1)
    atom_names = []
    num_atoms = pose.total_atoms()
    for i in range(1, num_atoms + 1):
        if element_or_name:
            atom_names.append(res.atom_type(i).element())
        else:
            atom_names.append(res.atom_name(i))
    return atom_names

def coords_from_pose(pose):
    res= pose.residue(1)

    # Create an empty list to store the coordinates
    coordinates = []
    num_atoms = pose.total_atoms()

    # Loop through all atoms and get the Cartesian coordinates
    for i in range(1, num_atoms + 1):  # Atom indices in PyRosetta are 1-based
        atom = res.atom(i)
        # Get the Cartesian coordinates of the atom
        coord = atom.xyz()  # This gives a PyRosetta XYZ object
        coordinates.append([coord.x, coord.y, coord.z])  # Convert to list

    # Convert the list to a NumPy array
    return np.array(coordinates)

def all_atom_coordinates_from_restype(restype, db):

    residx = restypes.index(restype)
    restype3 = restype_1to3[restype]
    num_chi = int(sum(chi_angles_mask[residx]))
    pose = pr.pose_from_sequence(restype)
    names = names_from_pose(pose, element_or_name=False)
    elements = names_from_pose(pose, element_or_name=True)

    # all residues with side_chains
    if restype3.lower() in db:

        db_res = db[restype3.lower()]
        backbone_confs = db_res[['Phi', 'Psi']].drop_duplicates().to_numpy()
        nconf = backbone_confs.shape[0]
        probs = np.array(db_res["Probabil"])/nconf
    # backbone only residues
    else: 
        backbone_confs = db["pro"][['Phi', 'Psi']].drop_duplicates().to_numpy() # get the backbone conf of another residue, PRO by default
        nconf = backbone_confs.shape[0]
        db_res = pd.DataFrame()
        db_res["Phi"] = backbone_confs[:,0]
        db_res["Psi"] = backbone_confs[:,1]
        probs=np.ones(nconf)*(1/nconf)

    all_coordinates=[]
    for rotamer  in tqdm.tqdm(db_res.itertuples()):
        pose.set_phi(1, rotamer.Phi)
        pose.set_psi(1, rotamer.Psi)
        for i in range(num_chi):
            pose.set_chi(1+i, 1, rotamer.__getattribute__("chi%iVal"%(i+1)))

        coords = coords_from_pose(pose)
        all_coordinates.append(coords)

    all_coordinates = np.array(all_coordinates)

    return all_coordinates, names, elements, probs


