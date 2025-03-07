
# import matplotlib
# matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import pyrosetta as pr
import numpy as np
import pandas as pd
from SingleBeadSAXS.rotamer_library import load_rotamor_library, all_atom_coordinates_from_restype, restype_1to3 
from SingleBeadSAXS.mapping_aas import map_pyrosetta_atom_names, get_res_map_martini, map_pyrosetta_martini_names
from SingleBeadSAXS.formfactor import cSAXSparameters, calculate_distogram, getAAFormFactor_fast, poly6d_fixed
import tqdm
import matplotlib.cm as cm
import os 
import argparse

def main(args):
    output_ff_file = args.output_prefix+ "_ff.csv"
    output_pngs_dir = args.output_prefix+ "_ff"
    libpath = args.libpath
    calculate_martini = args.martini_beads

    if calculate_martini:
        martini_mapping_path = args.martini_mapping_path
        if not martini_mapping_path:
            raise
        martini_mapping = get_res_map_martini(martini_mapping_path, remove_hydrogens = True)

    print("Starting ...")
    print("Saving csv file to %s "%output_ff_file)
    print("Saving png file to %s "%output_pngs_dir)

    pr.init()
    # I define q between 0 and 2, taking 0.001 steps
    max_q = 2.0
    no_q = 2001
    q = np.linspace(0.0,max_q,no_q)

    # Read the rotamer library
    db = load_rotamor_library(libpath)

    # if resume option is True, we read the output csv and continue. 
    if args.resume:
        outdf = pd.read_csv(output_ff_file)
    #Otherwise we initialize the output dataframe
    else:
        outdf = pd.DataFrame()
        outdf["q"] = q

    # Loop over the residues
    for restype in restype_1to3.keys():

        restype3 = restype_1to3[restype]
        print("\n > Calculating bead form factor for %s ..."%restype3)

        # if the residue is already calculated, skip ...
        if args.resume:
            if restype3 in [i.strip().split("_")[0] for i in list(outdf.keys())]:
                continue


        all_coordinates, atom_names,elements, probs = all_atom_coordinates_from_restype(restype, db)
        n_rotamer = len(all_coordinates)

        print("Number of rotamers found in Dunbrack lib: %i"%n_rotamer)

        atom_names_mapped = map_pyrosetta_atom_names(atom_names, restype3)

        ## MARTINI mapping
        if calculate_martini:
            martini_mapped = map_pyrosetta_martini_names(atom_names, restype3, martini_mapping)
            beads = list(set(sum([i for i in martini_mapped if i is not None], [])))
            n_beads = len(beads)
            beads_idx = [[i for i,m in enumerate(martini_mapped) if m is not None and b in m ] for b in beads]
        else: # single bead
            n_beads = 1
            beads = ["AA"]

        # let's remove the hydrogrens from the coordinates
        non_hydrogen_indices = [i for i, n in enumerate(atom_names_mapped) if n is not None]
        all_coordinates_non_hydrogen = all_coordinates[:,non_hydrogen_indices]
        atom_names_mapped_non_hydrogen = [n for i, n in enumerate(atom_names_mapped) if i in non_hydrogen_indices]

        # I only use the parameters, so no need for the arguments
        saxs_params = cSAXSparameters(None, None, None, None)

        # Here I get ff for all q and for all atoms
        ff  = np.array([[saxs_params.getFormFactor(_q, i) for _q in q]
                for i in atom_names_mapped_non_hydrogen])

        ffDummy =  np.array([[saxs_params.getDummyAtomsFactorCorr0(_q, i) for _q in q]
                for i in atom_names_mapped_non_hydrogen])

        ff_excl = ff -ffDummy

        # Output form factor
        F = np.zeros_like(all_coordinates_non_hydrogen, shape=(n_beads, n_rotamer, no_q))

        # Loop over the library
        for i, crd in tqdm.tqdm(enumerate(all_coordinates_non_hydrogen), total=n_rotamer):
            if calculate_martini:
                for bi, b in enumerate(beads):
                    dgram = calculate_distogram(crd[beads_idx[bi]])
                    F[bi, i] = getAAFormFactor_fast(dgram, q, ff_excl[beads_idx[bi]])
            else:
                dgram = calculate_distogram(crd)
                F[0, i] = getAAFormFactor_fast(dgram, q, ff_excl)


        # Average ponderated by the probabilty of each rotamer
        F_avg = np.sum(F * probs[...,None], axis=-2)

        #Polynomial fit
        F_polyfit = np.zeros_like(F_avg)
        q_range = (q==0) + (q>0.75)
        for b in range(n_beads):
            polynomial = poly6d_fixed(x_data = q[q_range], y_data= F_avg[b,q_range])
            F_polyfit[b] = polynomial(x=q)

            # Output to csv file
            outdf[restype3 +"_"+ beads[b]+"_cor"] = F_polyfit[b]
            outdf[restype3 +"_"+ beads[b]+"_uncor"] = F_avg[b]
        outdf.to_csv(output_ff_file, index=False, float_format="%.6f")

        # Plots ...
        fig, ax = plt.subplots(1,1, figsize=(5,4), layout="constrained")
        cmap = cm.get_cmap('tab10') 

        # Plots ...
        fig, ax = plt.subplots(1,1, figsize=(5,4), layout="constrained")
        for b in range(F_avg.shape[0]):
            ax.plot(q ,F_avg[b], label="%s uncorrected"%beads[b], ls="-", c=cmap(b))
            ax.plot(q ,F_polyfit[b], label="%s corrected"%beads[b], ls="--", c=cmap(b))
        ax.legend()
        ax.set_title("Bead Form factor for %s"%restype3)
        ax.set_xlabel("q ", fontsize=14)
        ax.set_ylabel("F(q)", fontsize=14)

        fig.savefig("%s_%s.png"%(output_pngs_dir, restype3), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "output_prefix", type=str, help="")
    parser.add_argument( "--libpath", type=str,help="", default="../saxs-python/dunbrack-rotamer/original/")
    parser.add_argument( "--martini_beads", action='store_true')
    parser.add_argument( "--resume", action='store_true')
    parser.add_argument( "--martini_mapping_path", type=str,help="", default="./martini_mapping")
    args = parser.parse_args()
    main(args)
    
