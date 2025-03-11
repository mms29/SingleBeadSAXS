
import matplotlib.pyplot as plt
import pyrosetta as pr
import numpy as np
import pandas as pd
from rotamer_library import load_rotamor_library, all_atom_coordinates_from_restype, restype_1to3 
from mapping_aas import map_pyrosetta_atom_names, get_res_map_martini, map_pyrosetta_martini_names
from formfactor import cSAXSparameters, calculate_distogram, getAAFormFactor_fast, poly6d_fixed
import tqdm
import matplotlib.cm as cm
import os 
import csv

output_ff_file = "poly_ff.csv"
output_pngs_dir = "plot_ff"

libpath = '/Users/isabelvinterbladh/Documents/Github/saxs-python/dunbrack-rotamer/original'
db = load_rotamor_library(libpath)

pr.init()
# I define q between 0 and 2, taking 0.001 steps
max_q = 2.0
no_q = 2001
q = np.linspace(0.0,max_q,no_q)



outdf = pd.DataFrame()
outdf["q"] = q
qfits = np.linspace(0.75, 2, 100)
qfits = np.insert(qfits, 0, 0.0)

# Loop over the residues
for restype in restype_1to3.keys():

    restype3 = restype_1to3[restype]
    print("\n > Calculating bead form factor for %s ..."%restype3)



    all_coordinates, atom_names, elements, probs = all_atom_coordinates_from_restype(restype, db)

    atom_names_mapped = map_pyrosetta_atom_names(atom_names, restype3)

        # let's remove the hydrogrens from the coordinates
    non_hydrogen_indices = [i for i, n in enumerate(atom_names_mapped) if n is not None]
    all_coordinates_non_hydrogen = all_coordinates[:,non_hydrogen_indices]
    atom_names_mapped_non_hydrogen = [n for i, n in enumerate(atom_names_mapped) if i in non_hydrogen_indices]

        # I only use the parameters, so no need for the arguments
    saxs_params = cSAXSparameters()
    ff  = np.array([saxs_params.computeFormFactors(atom_names_mapped_non_hydrogen, _q) for _q in qfits]).T
    #    ffDummy =  np.array([saxs_params.getDummyAtomsFactorCorr0(_q, i) for _q in q])
        # ff_excl = ff -ffDummy
    ff_excl = ff #-ffDummy


        
            
        #dgram = calculate_distogram(all)
    F = saxs_params.getAAFormFactor(qfits, ff_excl, all_coordinates_non_hydrogen)


        # Average ponderated by the probabilty of each rotamer
    F_avg = np.sum(F * probs[...,None], axis=-2)

        #Polynomial fit
    F_polyfit = np.zeros_like(F_avg)
    q_range = (q==0) + (q>0.75)

    polynomial = poly6d_fixed(x_data = qfits, y_data= F_avg)
    F_polyfit = polynomial(x=q)

            # Output to csv file
    outdf[restype3 +"__cor"] = F_polyfit
    #outdf[restype3 +"__uncor"] = F_avg
    outdf.to_csv(output_ff_file, index=False, float_format="%.6f")
    
    #fig.savefig("test.png"%(output_pngs_dir, restype3), dpi=300)



    
