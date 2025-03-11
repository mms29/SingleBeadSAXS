import pyrosetta as pr
import numpy as np
from rotamer_library import load_rotamor_library, all_atom_coordinates_from_restype, restype_1to3 
from mapping_aas import map_pyrosetta_atom_names
from formfactor import cSAXSparameters, calculate_distogram, getAAFormFactor_fast, poly6d_fixed
import tqdm
import csv

pr.init()

libpath = '/Users/isabelvinterbladh/Documents/Github/saxs-python/dunbrack-rotamer/original'
db = load_rotamor_library(libpath)

def getAll_FF(qvals, atom_names_mapped_non_hydrogen, all_coordinates_non_hydrogen):
    saxs_params = cSAXSparameters()
    FF = np.array([saxs_params.computeFormFactors(atom_names_mapped_non_hydrogen, q) for q in qvals]).T
    #for atoms in all_coordinates_non_hydrogen:
     #   all_rotamers = []
     #   for i, _q in enumerate(qvals):
            
    amino_FF = saxs_params.getAAFormFactor(qvals, FF, all_coordinates_non_hydrogen) 
    return amino_FF #np.array(total, dtype=complex)

with open('poly_coeffs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Restype', 'Coefficient Index', 'Value'])

    for restype in tqdm.tqdm(restype_1to3.keys()):
        restype3 = restype_1to3[restype]
        print("\n > Calculating bead form factor for %s ..."%restype3)
        
        all_coordinates, atom_names,elements, probs = all_atom_coordinates_from_restype(restype, db)
        n_rotamer = len(all_coordinates)
        print("Number of rotamers found in Dunbrack lib: %i"%n_rotamer)
        print(atom_names, restype3)
        
        atom_names_mapped = map_pyrosetta_atom_names(atom_names, restype3)
        # let's remove the hydrogrens from the coordinates
        non_hydrogen_indices = [i for i, n in enumerate(atom_names_mapped) if n is not None]
        all_coordinates_non_hydrogen = all_coordinates[:,non_hydrogen_indices]
        atom_names_mapped_non_hydrogen = [n for i, n in enumerate(atom_names_mapped) if i in non_hydrogen_indices]
        qfits = np.linspace(0.75, 2, 100)
        qfits = np.insert(qfits, 0, 0.0)
        #print((np.sqrt(np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1))[...,None]*qfits))
        
        
        total = getAll_FF(qfits, atom_names_mapped_non_hydrogen, all_coordinates_non_hydrogen)
        #print(total.shape)
        
        V = np.sum((total)*probs[...,None], axis=0)
        #F0 = getAll_FF([0.0], atom_names_mapped_non_hydrogen, all_coordinates_non_hydrogen)
        #qfits = np.insert(qfits, 0, 0.0)
        #V = np.insert(V, 0, np.sum(F0))
        
        poly_coeffs = poly6d_fixed(qfits, V)
        # Save poly_coeffs to a CSV fil
        coeffs = poly_coeffs.keywords['coeffs']
        for i, coeff in enumerate(coeffs):
                writer.writerow([restype3, i, coeff])
    