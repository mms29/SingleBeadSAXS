# debyecalc.py
# Author: Isabel Vinterbladh
# Date: June 2024
# Description: Calculate the scattering intensity I(q) using the Debye formula for a protein structure.

import voro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict



class DebyeCalculator:
    def __init__(self, q_min=0.0, q_max=0.5, num_q_points=101):
        self.q_min = q_min
        self.q_max = q_max
        self.num_q_points = num_q_points    

    def calculate_Iq(self, structure_file, solvent_file, poly_ff, Explicit_dummy=False):
        """ Calculate the scattering intensity I(q) using the Debye formula.
        Args:
            structure_file (str): Path to the protein structure file.
            solvent_file (str): Path to the solvent structure file.
            poly_ff (np.array): Polynomial coefficients for form factors.
            Explicit_dummy (bool): Whether to use amino acid volumes for explicit solvent particles.
        Returns:
            q_values (np.array): Array of q values.
            Iq_values (np.array): Corresponding I(q) values.
        """
        # Load the protein structure
        amino_code, structure = self.load_structure(structure_file)
        # Load solvent structure and reduce points
        water_struct, water_weights = self.water_points(solvent_file, box_size=3)
        # Generate q values
        q_values = np.linspace(self.q_min, self.q_max, self.num_q_points)
        # If using amino acid volumes from voronota, adjust form factors
        if Explicit_dummy:
             # Calculate amino acid volumes
            aa_volumes = self.amino_acid_volume(structure, amino_code)
            aa_FormFactors = np.array([self.getfitted_ff(aa, q_values, poly_ff) - self.overallexpansion_factor(q_values, aa_volumes[i])*self.dummyFactor(q_values, aa_volumes[i]) for i, aa in enumerate(amino_code)])
        else:
            aa_FormFactors = np.array([self.getfitted_ff(aa, q_values, poly_ff) for aa in amino_code])
        # Get water form factors
        water_FormFactors = np.array([water_weights[w]*self.getfitted_ff('H20', q_values, poly_ff) for w in range(len(water_struct))])
        # Calculate I(q) using the Debye formula
        Iq_values = self.debye_formula(structure, q_values, aa_FormFactors, water_struct, water_FormFactors)
        return q_values, Iq_values

    def plot_Iq(self, q_values, Iq_values, label='Debye Equation'):
        plt.figure(figsize=(8, 6))
        plt.plot(q_values, Iq_values, label=label)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('q (1/Å)')
        plt.ylabel('I(q)')
        plt.title('Scattering Intensity I(q)')
        plt.legend()
        plt.show()

    def load_structure(self, file_path):
        """ Load the protein structure from a file.
        Args:
            file_path (str): Path to the structure file.
        Returns:
            amino_code (list): List of amino acid codes.
            df_struct (np.array): Array of amino acid coordinates.
        """
        # Load the protein structure from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data =np.array([line.split() for line in lines[2:]])
        amino_code = data[:, 0]
        df_struct = data[:, 1:].astype(float)
        return amino_code, df_struct
    
    def getfitted_ff(amino_acid, q, poly_ff):
        """ Get the fitted form factor for a given amino acid
            q is the scattering vector
        Args:
            amino_acid (str): Amino acid
            q (np.array): scattering vector
            poly_ff (np.array): polynomial coefficients for each amino acid
        returns:
            result (np.array): fitted form factor
        """
        result = np.zeros(len(q))
        poly = poly_ff[poly_ff[:,0]==amino_acid,2]
        result = poly[0]+poly[1]*q+poly[2]*q**2+poly[3]*q**3+poly[4]*q**4+poly[5]*q**5+poly[6]*q**6
        return result

    def water_points(self, solvent_file, box_size=3):
        """ Loading hydration shell points and make water layer continuous.
        Reduce the number of solvent points by averaging points within boxes of given size.
        This makes the hydration shell uniformly distributed/dense.
        Args:
            solvent_file (str): Path to the solvent structure file.
            box_size (float): Size of the boxes to average points within.
        Returns:
            new_points (np.array): Reduced set of solvent points and their coordinates.
            new_weights (np.array): Corresponding weights for the solvent points.
        """
        solvent = pd.read_csv(solvent_file, sep="\t", header=None)
        solvent.columns = solvent.columns.astype(float)
        solvent_np = np.vstack([solvent.columns.values, solvent.to_numpy()])
        water_struct = solvent_np[:, :3].astype(float)
        sol_weights = solvent_np[:,-1].astype(float)
        
        # Dictionary: key = (i,j,k) box index, value = list of point indices
        boxes = defaultdict(list)

        for idx, (x, y, z) in enumerate(water_struct):
            i = int(np.floor(x / box_size))
            j = int(np.floor(y / box_size))
            k = int(np.floor(z / box_size))
            boxes[(i, j, k)].append(idx)
        # --- one representative point per box ---
        new_points = []
        new_weights = []
        for cell, idx_list in boxes.items():
            if len(idx_list) > 1:
                # average coordinates if box has multiple points
                new_points.append(np.mean(water_struct[idx_list], axis=0))
                new_weights.append(np.mean(sol_weights[idx_list]))
            else:
                # take the single point unchanged
                new_points.append(water_struct[idx_list[0]])
                new_weights.append(sol_weights[idx_list[0]])

        return np.array(new_points).astype(float), np.array(new_weights).astype(float)
    
    def amino_acid_volume(self, df_struct, amino_code):
        """ Calculate the volume of each amino acid using Voronoi tessellation.
        Args:
            df_struct (np.array): Array of amino acid coordinates.
            amino_code (list): List of amino acid codes.
        Returns:
            volumes (np.array): Array of volumes for each amino acid.
        """
        balls = []
        # Dictionary containing the radius of respective amino acids
        amino_acid_radii = {
                "ALA": 3.2,
                "ARG": 5.6,
                "ASN": 4.04,
                "ASP": 4.04,
                "CYS": 3.65,
                "GLU": 4.63,
                "GLN": 4.64,
                "GLY": 1.72,
                "HIS": 4.73,
                "ILE": 3.94,
                "LEU": 4.24,
                "LYS": 5.02,
                "MET": 4.47,
                "PHE": 4.99,
                "PRO": 3.61,
                "SER": 3.39,
                "THR": 3.56,
                "TRP": 5.38,
                "TYR": 5.36,
                "VAL": 3.55,
        }
        for i, aa in enumerate(amino_code):
            balls.append(voro.Ball(df_struct[i][0], df_struct[i][1], df_struct[i][2], amino_acid_radii[aa]))
        rt = voro.RadicalTessellation(balls, probe=1.4)
        cells=list(rt.cells)
        return np.array([cell.volume for cell in cells])

    def dummyFactor(q, V):
        """ Dummy factor for testing purposes. """
        q2 = q*q/(4*np.pi**2)
        return 0.334 * V * np.exp(-q2 * pow(V, 2/3))

    def overallexpansion_factor(q, V):
        q2 = q*q /(4*np.pi)
        
        r0 = (3*V/(4*np.pi))**(1/3)
        rm = 4.188
        #if r0 <1.04*rm and r0 >0.96*rm:
        #    print("Using calculated expansion factor")
        #    rbest = r0
        #else:
        #    print("Using max expansion factor")
        #    rbest = 0.5*(1.04*rm + 0.96*rm)
        
        return (r0/rm)**3 * np.exp(-q2 * pow(4*np.pi/3, 2/3)*(r0**2-rm**2))

    def debye_mem(dgram, q, ff, eps=1e-6):
        """ Calculate I(q) using the Debye formula for a single type of scatterer.
        Args:
            dgram (np.array): Distance matrix between scatterers.
            q (float): Scattering vector.
            ff (np.array): Form factors for the scatterers.
            eps (float): Small value to avoid division by zero.
        Returns:
            Iq (float): Scattering intensity I(q).
        """
        ff2 = (ff[None] * ff[:, None])
        if q == 0:
                return np.sum(ff2)
        else:
            Iq = np.zeros_like(dgram)
            dq = dgram * q
            indices_zeros = dq<eps
            indices_nonzeros = ~indices_zeros
            
            Iq[indices_nonzeros] = ff2[indices_nonzeros]  * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
            Iq[indices_zeros] = ff2[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)
            return np.sum(Iq, axis=(0,1))

    def debye_hs(dgram, q, waterff, aminoff, eps=1e-6):
        """ Calculate I(q) using the Debye formula for two types of scatterers (e.g., protein and solvent).
        Args:
            dgram (np.array): Distance matrix between scatterers.
            q (float): Scattering vector.
            waterff (np.array): Form factors for the solvent scatterers.
            aminoff (np.array): Form factors for the protein scatterers.
            eps (float): Small value to avoid division by zero.
        Returns:
            Iq (float): Scattering intensity I(q).
        """
        # Calculate all combinations of form factors between water and amino acids
        ff2 = waterff[:, None] * aminoff[None] #all water form factors times all amino acid form factor
        if q == 0:
                return np.sum(ff2)
        else:
            Iq = np.zeros_like(dgram)
            dq = dgram * q
            indices_zeros = dq<eps
            indices_nonzeros = ~indices_zeros

            Iq[indices_nonzeros] = ff2[indices_nonzeros]  * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
            Iq[indices_zeros] = ff2[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)
            return np.sum(Iq, axis=(0,1))
        
    def calculate_distogram(coords):
        dgram = np.sqrt(np.sum(
            (coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1
        ))
        return dgram

    def calculate_dist2(water_coords, aa_coords):
        dgram = np.sqrt(np.sum(
            (water_coords[..., None, :] - aa_coords[..., None, :, :]) ** 2, axis=-1
        ))
        return dgram


    def debye_formula(self, structure, q_values, aa_FormFactors, water_struct, water_FormFactors):
        """ Calculate I(q) using the Debye formula.
        Args:
            structure (np.array): Coordinates of the structure.
            q_values (np.array): Scattering vector values.
            aa_FormFactors (np.array): Form factors for the amino acid scatterers.
            water_struct (np.array): Coordinates of the solvent.
            water_FormFactors (np.array): Form factors for the solvent scatterers.
        Returns:
            np.array: Scattering intensity I(q) for each q value where 
            I(q) = I_aa(q) + I_solv(q) + 2*I_cross(q). (atomistic, excluded solvent and hydration shell contributions)
        """
        # Calculate I(q) using the Debye formula
        Iq_values = np.zeros(len(q_values))
        for i, q in enumerate(q_values):
            Iq_values[i] = self.debye_mem(self.calculate_distogram(structure), q, aa_FormFactors[:,i]) 
            + self.debye_mem(self.calculate_distogram(water_struct), q, water_FormFactors[:,i])
            + 2 * self.debye_hs(self.calculate_dist2(water_struct, structure), q, water_FormFactors[:,i], aa_FormFactors[:,i])
        return Iq_values


