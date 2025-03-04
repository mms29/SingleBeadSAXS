# mapping_aas.py: This file contains the mapping of amino acids to their corresponding groups.
# Author: Isabel Vinterbladh
### Class Group defines the types and a corresponding integer value ###
class Group:
    class Type:
        C = 0
        O = 1
        N = 2
        S = 3
        CH = 4
        CH2 = 5
        CH3 = 6
        Csp2H = 7
        CaromH = 8
        OalcH = 9
        OacidH = 10
        NIV = 11
        NH = 12
        NH2 = 13
        NHIV = 14
        NH2IV = 15
        NH3IV = 16
        SH = 17
        H = 18
        H_lab = 19
        H_nonlab = 20
        H2O = 21
        NH3 = 22
        MG = 23
        IR = 24
        FE = 25
        CU = 26
        ZN = 27
        CA = 28
        CL = 29
        P = 30
        SE = 31
        SEH = 32
        Ocarbox = 33
        O_ = 34
        N_guanidinium = 35
        N_guanidinium_1 = 36
        I = 37
        D = 38
        D2O = 39
        Bulk = 40
        UNK = 41
        MN = 42
        MO = 43
        BR = 44
        RB = 45
        AR = 46
        K = 47
        XE = 48
        NI = 49
        CO = 50
        KR = 51
        NA = 52
        HG = 53
        Ag = 54
        Au = 55
        U = 56
        F = 57
        Al = 58
        Si = 59
        Cr = 60
        Sr = 61
        Pd = 62
        Cs = 63
        Ba = 64
        Pt = 65
        Ga = 66
        Ge = 67
        Cd = 68
        Zero = 69
        COUNT = 70

    def __init__(self, t=None):
        self.t_ = t

    def __call__(self):
        return self.t_
 
### Class Mapping defines the mapping of atomic groups/atoms to their corresponding type in Class Group ###   
class Mapping:
    def __init__(self):
        self.mapping = {}

    def update(self, new_mapping):
        self.mapping.update(new_mapping)

### Defines Function which updates the mapping between groups/atoms etc. ###
def mapping_amino(params):
    pdbAtomMap = Mapping()
    atomMap = Mapping()
    pdbMap = Mapping()

    if not params.noRes:
        pdbAtomMap.update({
            "OXT": "Ocarbox",
            "OT1": "Ocarbox",
            "OT2": "Ocarbox",
            "OCT1": "Ocarbox",
            "OCT2": "Ocarbox",
            "OT": "Ocarbox",
            "O1": "Ocarbox",
        })
        
    else:
        pdbAtomMap.update({
            "OXT": "O_",
            "OT1": "O_",
            "OT2": "O",
            "OCT1": "O_",
            "OCT2": "O",
            "OT": "O_",
            "O1": "O_",
            "O2": "O"
        })

    pdbAtomMap.update({
        "C": "C",
        "O": "O",
        "N": "NH",
        "CA": "CH"
    })
    atomMap.update({
        "C": "C",
        "O": "O",
        "N": "N",
        "S": "S",
        "P": "P",
        "MG": "MG",
        "IR": "IR",
        "FE": "FE",
        "CU": "CU",
        "ZN": "ZN",
        "CA": "CA",
        "CR": "Cr",
        "CL": "CL",
        "SE": "SE",
        "H": "H",
        "I": "I",
        "IOD": "I",
        "D": "D"
    })
    pdbMap.update({
        "ALA.CB": "CH3",
        "ARG.CG": "CH2",
        "ARG.CD": "CH2",
        "ARG.CZ": "C",
        "ARG.CB": "CH2",
        "ASN.CB": "CH2",
        "ASN.CG": "C",
        "ASN.OD1": "O",
        "ASN.ND2": "NH2",
        "ASP.CB": "CH2",
        "ASP.CG": "C",
        "CYS.CB": "CH2",
        "CYS.SG": "SH",
        "SEC.CB": "CH2",
        "SEC.SE": "SEH",
        "GLU.CB": "CH2",
        "GLU.CG": "CH2",
        "GLU.CD": "C",
        "GLN.CB": "CH2",
        "GLN.CG": "CH2",
        "GLN.CD": "C",
        "GLN.OE1": "O",
        "GLN.NE2": "NH2",
        "GLY.CA": "CH2",
        "HIS.CB": "CH2",
        "HIS.CG": "C",
        "HIS.ND1": "N",
        "HIS.CD2": "Csp2H",
        "HIS.CE1": "Csp2H",
        "HIS.NE2": "NH",
        "ILE.CB": "CH",
        "ILE.CG1": "CH2",
        "ILE.CG2": "CH3",
        "ILE.CD1": "CH3",
        "ILE.CD": "CH3",
        "LEU.CB": "CH2",
        "LEU.CG": "CH",
        "LEU.CD1": "CH3",
        "LEU.CD2": "CH3",
        "LYS.CB": "CH2",
        "LYS.CG": "CH2",
        "LYS.CD": "CH2",
        "LYS.CE": "CH2",
        "LYS.NZ": "NH3IV",
        "MET.CB": "CH2",
        "MET.CG": "CH2",
        "MET.SD": "S",
        "MET.S": "S",
        "MET.CE": "CH3",
        "MSE.CB": "CH2",
        "MSE.CG": "CH2",
        "MSE.SE": "SE",
        "MSE.CE": "CH3",
        "PHE.CB": "CH2",
        "PHE.CG": "C",
        "PHE.CD1": "CaromH",
        "PHE.CD2": "CaromH",
        "PHE.CE1": "CaromH",
        "PHE.CE2": "CaromH",
        "PHE.CZ": "CaromH",
        "PRO.CB": "CH2",
        "PRO.CG": "CH2",
        "PRO.CD": "CH2",
        "SER.CB": "CH2",
        "SER.OG": "OalcH",
        "THR.CB": "CH",
        "THR.OG1": "OalcH",
        "THR.CG2": "CH3",
        "TRP.CB": "CH2",
        "TRP.CG": "C",
        "TRP.CD1": "CH",
        "TRP.CD2": "C",
        "TRP.NE1": "NH",
        "TRP.CE2": "C",
        "TRP.CE3": "CaromH",
        "TRP.CZ2": "CaromH",
        "TRP.CZ3": "CaromH",
        "TRP.CH2": "CaromH",
        "TYR.CB": "CH2",
        "TYR.CG": "C",
        "TYR.CD1": "CaromH",
        "TYR.CE1": "CaromH",
        "TYR.CE2": "CaromH",
        "TYR.CD2": "CaromH",
        "TYR.CZ": "C",
        "TYR.OH": "OalcH",
        "VAL.CB": "CH",
        "VAL.CG1": "CH3",
        "VAL.CG2": "CH3"
    })
    
    if not params.noRes:
        pdbMap.update({
            "ARG.NE": "N_guan_1",
            "ARG.NH1": "N_guan",
            "ARG.NH2": "N_guan",
            "ASP.OD1": "Ocarbox",
            "ASP.OD2": "Ocarbox",
            "GLU.OE1": "Ocarbox",
            "GLU.OE2": "Ocarbox"
        })
        
    else:
        pdbMap.update({
            "ARG.NE": "NHIV",
            "ARG.NH1": "NH2",
            "ARG.NH2": "NH2",
            "ASP.OD1": "O_",
            "ASP.OD2": "O",
            "GLU.OE1": "O_",
            "GLU.OE2": "O"
        })
        
    return pdbAtomMap, atomMap, pdbMap

### Class Params defines the parameters for the mapping of amino acids ###
class Params:
    def __init__(self, noRes):
        self.noRes = noRes

# Create an instance of Params with noRes set to False
params = Params(noRes=False)

# Call the init_mapping_amino function with the params object
pdb_atom_map, atom_map, pdb_map = mapping_amino(params)

# # Print the mappings
# print("PDB Atom Map:", pdb_atom_map.mapping)
# print("Atom Map:", atom_map.mapping)
# print("PDB Map:", pdb_map.mapping)


def map_pyrosetta_atom_names(atom_names, restype3):
    atom_names_mapped = []
    for k in [v.strip() for v in atom_names]:
        if restype3 + "." + k in pdb_map.mapping:
            atom_names_mapped.append(pdb_map.mapping[restype3 + "." + k])
        elif k in pdb_atom_map.mapping:
            atom_names_mapped.append(pdb_atom_map.mapping[k])
        elif k in atom_map.mapping:
            atom_names_mapped.append(atom_map.mapping[k])
        else: 
            # Missing ones are the oxygens (hopefully), there are removed 
            if not "H" in k: # this is crude
                print("Warning : missing atom : %s"%k)
                # raise RuntimeError("We should only have hydrogen atom missing, this should be investigated")
            atom_names_mapped.append(None) # I append nothing here just to remove it later
    return atom_names_mapped


def map_pyrosetta_martini_names(atom_names, restype3, martini_mapping):
    martini_mapped = []
    for k in [v.strip() for v in atom_names]:
        if k not in martini_mapping[restype3]:
            martini_mapped.append(None)
        else:
            martini_mapped.append(martini_mapping[restype3][k])
    return martini_mapped

import os
from SingleBeadSAXS.rotamer_library import restypes3

# Read the file of Martini mapping for one residue
def read_mapping_martini(file):
    mapping = {}
    with open(file, "r") as f:
        line = f.readline().strip()
        while(not line.startswith("[ atoms ]")):
            line = f.readline().strip()

        for l in f:
            l = l.strip()
            if l.startswith("["):
                break
            if l.startswith(";"):
                continue
            spl = l.split()
            if ";" in spl:
                spl = spl[:spl.index(";")]
            if len(spl)<3:
                continue
            mapping[spl[1]] = spl[2:]
    return mapping

# map each residue to martini beads
def get_res_map_martini(
        mapping_path,
        charmm_or_gromo = False,
        remove_hydrogens=False
):

    if not charmm_or_gromo:
        fname =    "%s.gromos.map"
    else:
        fname =    "%s.charmm36.map"

    mapping = {}
    for res in restypes3:
        file = os.path.join(mapping_path, fname%(res.lower()))
        mapping[res] =  read_mapping_martini(file)

    if remove_hydrogens:
        mapping =  {k:{i:v2 for i,v2 in v.items() if not i.startswith("H")} for k, v in mapping.items()}
    return mapping

