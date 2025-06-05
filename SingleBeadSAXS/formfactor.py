# formfactor.py
"""
Author: Isabel Vinterbladh
This module contains the `cSAXSparameters` class, which is used to calculate the form factors of atoms and amino acids in protein structures. 
The calculations are based on scattering vector magnitudes and predefined atomic parameters. The module also includes utility functions for distance calculations and fast form factor computations.
Classes:
cSAXSparameters:
    - A class for calculating atomic and amino acid form factors.
    - Contains methods for initializing atomic parameters, computing form factors, and handling dummy atom corrections.
Functions:
----------
calculate_distogram(coords):
    - Calculates the pairwise Euclidean distance matrix (distogram) for a set of coordinates.
getAAFormFactor_fast(dgram, q, ff, eps=1e-6):
    - Computes the form factor using a fast implementation of equations from the Martini paper.
poly6d_fixed(x_data, y_data):
    - Fits a fixed 6th-degree polynomial to the given data points with a constraint that the first derivative at x=0 is zero.
------
- The `cSAXSparameters` class includes methods for handling special cases, such as Arginine residues, and provides corrections for dummy atoms and hydration shells.
- The module uses mathematical models and optimization techniques to compute accurate form factors for protein structures.
"""
import numpy as np
from scipy.optimize import minimize
from functools import partial
#import voronotalt_python as voronota


class cSAXSparameters:
    """
    This class is used to calculate the form factors of the atoms and then amino acids in the protein structure.
    The class contains the following methods:
    1. __init__: Initializes the class with the given parameters.
    2. paramsFormFactors: Initializes the parameters for the form factors of the atoms.
    3. getFormFactor: Calculates the form factor of an atom given the q value and the atom type.
    4. getGroupFormFactor: Calculates the form factor of a group of atoms given the q value and the group type.
    5. getGroupFormFactor2: Calculates the form factor of a group of atoms given the q value and the group type. - Special case for Arginine.
    6. getDummyAtomsFactor: Calculates the form factor of the dummy atoms given the q value and the atom type. 
    7. getDummyAtomsFactorCorr0: Calculates the form factor of the dummy atoms given the q value and the atom type.
    8. getDummyAtomsFactorFraser: Calculates the form factor of the dummy atoms given the q value and the atom type.
    9. getDummyAtomsFactorSvergun: Calculates the form factor of the dummy atoms given the q value and the atom type.
    10. getHydrationShell: Calculates the hydration shell of the protein structure.
    11. computeFormFactors: Computes the form factors of the atoms in the protein structure. 
    12. getAAFormFactor: Calculates the form factor of the amino acids in the protein structure. ref. Single bead approximation 2.1.4
    """
    # Mean electron density of the solvent
    MEAN_ELECTRON_DENSITY = 0.334

    def __init__(self):
        self.fj = {}
        self.paramsFormFactors()

    def paramsFormFactors(self):
        """
        Initializes the `fj` dictionary with predefined parameters for various form factors.
        This method defines a nested class `cSASParams` to encapsulate the parameters for each form factor.
        The `fj` dictionary is populated with instances of `cSASParams`, each representing a specific form factor
        with its associated parameters.
        Attributes:
            fj (dict): A dictionary where keys are atom or atom group names (e.g., 'N', 'C', 'O') and values are instances
                       of `cSASParams` containing the parameters for the respective atom.
        Nested Class:
            cSASParams:
                Represents form factor parameters for an atom.
                Attributes:
                    a1, a2, a3, a4, a5 (float): Coefficients for the five-Gaussian equation used for the form factor calculation.
                    b1, b2, b3, b4, b5 (float): Exponential coefficients for five-Gaussian equation used for the form factor calculation.
                    c (float): Constant parameter in the five-Gaussian equation.
                    h (int): Hydrogen count associated with the atom/atom group.
                    rh (float): Distanc between main atom and hydrogen (optional, default is 0, when no hydrogens are included).
                    r (float): Radius of the atom or group.
                    dsv (float): Dry solvent volume.
                    name (str): Name of the atom.
        Notes:
            - The `fourPi` constant is used to calculate the volume of atoms based on their radius.
            - Some form factors include specific adjustments or scaling factors (e.g., 'NIV', 'NHIV').
            - The method supports the calculation of a wide range of form factors, including atoms, functional groups, and ions.
        Example:
            After calling this method, the `fj` dictionary will contain entries like:
            `fj['N']`, `fj['C']`, `fj['O']`, etc., each initialized with their respective parameters.
        """
        
        class cSASParams:
            
            def __init__(self, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c, h, r, dsv, name, rh=0):
                self.a1 = a1
                self.a2 = a2
                self.a3 = a3
                self.a4 = a4
                self.a5 = a5
                self.b1 = b1
                self.b2 = b2
                self.b3 = b3
                self.b4 = b4
                self.b5 = b5
                self.c = c
                self.h = h
                self.rh = rh
                self.r = r
                self.dsv = dsv
                self.name = name
                
        fourPi = 4 * np.pi  # Used to calculate the volume of atoms based on their radius
        self.fj['N'] = cSASParams(11.893780, 3.277479, 1.858092, 0.858927, 0.912985, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490, 0, 0.84, 2.49, "N")
        self.fj['C'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 0, 1.577, 16.43, "C")
        self.fj['O'] = cSASParams(2.960427, 2.508818, 0.637853, 0.722838, 1.142756, 14.182259, 5.936858, 0.112726, 34.958481, 0.390240, 0.027014, 0, 1.3, 9.203, "O")
        self.fj['S'] = cSASParams(6.372157, 5.154568, 1.473732, 1.635073, 1.209372, 1.514347, 22.092527, 0.061373, 55.445175, 0.646925, 0.154722, 0, 1.68, 19.86, "S")
        self.fj['SE'] = cSASParams(6.372157, 5.154568, 1.473732, 1.635073, 1.209372, 1.514347, 22.092527, 0.061373, 55.445175, 0.646925, 0.154722, 0, 1.68, 19.86, "SE")
        self.fj['H'] = cSASParams(0.493002, 0.322912, 0.140191, 0.040810, 0, 10.5109, 26.1257, 3.14236, 6.14236, 57.7997, 0.003038, 0, pow(3./(4*np.pi)*5.116, 1./3), 5.116, "H")
        self.fj['CH'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 1, 1.73, 21.69, "CH", 1.099)
        self.fj['CH2'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 2, 1.85, 26.52, "CH2", 1.092)
        self.fj['CH3'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 3, 1.97, 32.03, "CH3", 1.059)
        self.fj['Csp2H'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 1, 1.73, 21.69, "Csp2H", 1.077)
        self.fj['CaromH'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 1, 1.73, 21.69, "CaromH", 1.083)
        self.fj['OalcH'] = cSASParams(2.960427, 2.508818, 0.637853, 0.722838, 1.142756, 14.182259, 5.936858, 0.112726, 34.958481, 0.390240, 0.027014, 1, 1.5, 14.14, "OalcH", 0.967)
        self.fj['OacidH'] = cSASParams(2.960427, 2.508818, 0.637853, 0.722838, 1.142756, 14.182259, 5.936858, 0.112726, 34.958481, 0.390240, 0.027014, 1, 1.5, 14.14, "OacidH", 1.015)
        self.fj['O_'] = cSASParams(3.106934, 3.235142, 1.148886, 0.783981, 0.676953, 19.86808, 6.960252, 0.170043, 65.693512, 0.63, 0.046136, 0, 1.3, 9.203, "O_")
        self.fj['Ocarbox'] = cSASParams(0.688944, 2.929687, 0.416472, 2.606983, 1.319232, 29.3192, 6.572228, 64.951658, 16.267799, 0.45564, 0.537548, 0, 1.3, 9.203, "Ocarbox")
        self.fj['NH'] = cSASParams(11.893780, 3.277479, 1.858092, 0.858927, 0.912985, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490, 1, 1.009, 7.606, "NH", 1.009)
        self.fj['NH2'] = cSASParams(11.893780, 3.277479, 1.858092, 0.858927, 0.912985, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490, 2, 1.45, 12.77, "NH2", 1.009)
        self.fj['NIV'] = cSASParams(11.893780 * 6.0/7.0, 3.277479 * 6.0/7.0, 1.858092 * 6.0/7.0, 0.858927 * 6.0/7.0, 0.912985 * 6.0/7.0, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490 * 6.0/7.0, 0, 0.76 * 1.033, 2.49, "NIV", 0.76 * 1.033)
        self.fj['NHIV'] = cSASParams(11.893780 * 6.0/7.0, 3.277479 * 6.0/7.0, 1.858092 * 6.0/7.0, 0.858927 * 6.0/7.0, 0.912985 * 6.0/7.0, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490 * 6.0/7.0, 1, 1.22, 7.606, "NHIV", 0.76 * 1.033)
        self.fj['NH2IV'] = cSASParams(11.893780 * 6.0/7.0, 3.277479 * 6.0/7.0, 1.858092 * 6.0/7.0, 0.858927 * 6.0/7.0, 0.912985 * 6.0/7.0, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490 * 6.0/7.0, 2, 1.45, 12.77, "NH2IV", 0.76 * 1.033)
        self.fj['NH3IV'] = cSASParams(11.893780 * 6.0/7.0, 3.277479 * 6.0/7.0, 1.858092 * 6.0/7.0, 0.858927 * 6.0/7.0, 0.912985 * 6.0/7.0, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490 * 6.0/7.0, 3, 1.62, 17.81, "NH3IV", 0.76 * 1.033)
        self.fj['N_guan'] = cSASParams(1.792216, 0.724464, 2.347044, 1.90302, 1.313042, 10.83006, 6.846763, 29.579607, 10.800018, 0.720448, 0.583312, 0, 1.54, 15.29, "N_guan")
        self.fj['N_guan_1'] = cSASParams(3.630164, 0.22831, 1.869734, 0.17055, 1.440894, 10.267139, 25.118086, 30.241288, 3.412776, 0.486644, 0.323504, 0, 1.45, 12.77, "N_guan_1")
        self.fj['SH'] = cSASParams(6.372157, 5.154568, 1.473732, 1.635073, 1.209372, 1.514347, 22.092527, 0.061373, 55.445175, 0.646925, 0.154722, 1, 1.43, 24.84, "SH", 1.43)
        self.fj['H2O'] = cSASParams(2.960427, 2.508818, 0.637853, 0.722838, 1.142756, 14.182259, 5.936858, 0.112726, 34.958481, 0.390240, 0.027014, 2, pow(3./(4*np.pi)*30.0, 1./3), 30.0, "H2O", 0.9584)
        self.fj['UNK'] = cSASParams(2.657506, 1.078079, 1.490909, -4.241070, 0.713791, 14.780758, 0.776775, 42.086842, -0.000294, 0.239535, 4.297983, 0, 1.577, 16.43, "UNK")
        self.fj['NH3'] = cSASParams(11.893780, 3.277479, 1.858092, 0.858927, 0.912985, 0.000158, 10.232723, 30.344690, 0.656065, 0.217287, -11.80490, 3, 1.62, 17.81, "NH3", 0.76 * 1.033)
        self.fj['MG'] = cSASParams(4.708971, 1.194814, 1.558157, 1.170413, 3.239403, 4.875207, 108.506081, 0.111516, 48.292408, 1.928171, 0.126842, 0, 1.6, fourPi*1.6*1.6*1.6/3.0, "MG")
        self.fj['CU'] = cSASParams(14.014192, 4.784577, 5.056806, 1.457971, 6.932996, 3.73828, 0.003744, 13.034982, 72.554794, 0.265666, -3.254477, 0, 1.28, fourPi*1.28*1.28*1.28/3.0, "CU")
        self.fj['FE'] = cSASParams(12.311098, 1.876623, 3.066177, 2.070451, 6.975185, 5.009415, 0.014461, 18.74304, 82.767876, 0.346506, -0.304931, 0, 1.24, fourPi*1.24*1.24*1.24/3.0, "FE")
        self.fj['CL'] = cSASParams(1.446071, 6.870609, 6.151801, 1.750347, 0.634168, 0.052357, 1.193165, 18.343416, 46.398396, 0.401005, 0.146773, 0, 1.75, fourPi*1.75*1.75*1.75/3.0, "CL")
        self.fj['ZN'] = cSASParams(14.741002, 6.907748, 4.642337, 2.191766, 38.424042, 3.388232, 0.243315, 11.903689, 63.31213, 0.000397, -36.915829, 0, 1.33, fourPi*1.33*1.33*1.33/3.0, "ZN")
        self.fj['IR'] = cSASParams(32.004436, 1.975454, 17.070105, 15.939454, 5.990003, 1.353767, 81.014175, 0.128093, 7.661196, 26.659403, 4.018893, 0, 1.58, fourPi*1.58*1.58*1.58/3.0, "IR")
        self.fj['Cr'] = cSASParams(11.007069, 1.555477, 2.985293, 1.347855, 7.034779, 6.366281, 0.023987, 23.244839, 105.774498, 0.429369, 0.06551, 0, 1.85, fourPi*1.85*1.85*1.85/3.0, "Cr")
        self.fj['P'] = cSASParams(1.950541, 4.14693, 1.49456, 1.522042, 5.729711, 0.908139, 27.044952, 0.07128, 67.520187, 1.981173, 0.155233, 0, 1.11, fourPi*1.11*1.11*1.11/3.0, "P")
        self.fj['CA'] = cSASParams(8.593655, 1.477324, 1.436254, 1.182839, 7.113258, 10.460644, 0.041891, 81.390381, 169.847839, 0.688098, 0.196255, 0, 1.97, fourPi*1.97*1.97*1.97/3.0, "CA")
        self.fj['I'] = cSASParams(19.884502, 6.736593, 8.110516, 1.170953, 17.548716, 4.628591, 0.027754, 31.849096, 84.406387, 0.463550, -0.448811, 0, 1.98, fourPi*1.98*1.98*1.98/3.0, "I")


    def getFormFactor(self, q, F): # F is the atom type as a string
        """
        Calculate the form factor for a given atom type and scattering vector magnitude.
        Parameters:
            q (float): The magnitude of the scattering vector.
            F (str): The atom type as a string.
        Returns:
            float: The calculated form factor for the specified atom type.
        Raises:
            ValueError: If the provided atom type `F` is not valid.
        Notes:
            - The function uses pre-defined parameters for each atom type stored in `self.fj`.
            - Special cases for "N_guan_1" (NE of Arginine) and "N_guan" (NH of Arginine) are handled
              separately using `getGroupFormFactor2`.
            - For atoms with `h == 0`, the form factor is calculated using a sum of exponential terms
              based on atom-specific parameters.
            - For other atoms, the form factor is calculated using `getGroupFormFactor`.
        """
        
        if F not in self.fj:
            raise ValueError(f"Invalid atom type '{F}'. Please provide a valid atom type.")
        
        q2 = q * q / (16 * np.pi * np.pi)   
        atom = self.fj[F] # finding which params to use for the atom type
        if atom.name == "N_guan_1":  # NE of Arginine
            return self.getGroupFormFactor2(q, 2.0 / 3.0, self.fj['NHIV'], 1.0 / 3.0, self.fj['NIV'])
        if atom.name == "N_guan":  # NH of Arginine
            return self.getGroupFormFactor2(q, 2.0 / 3.0, self.fj['NH2'], 1.0 / 3.0, self.fj['NH'])

        if atom.h == 0:
            return atom.c + atom.a1 * np.exp(-q2 * atom.b1) + atom.a2 * np.exp(-q2 * atom.b2) + atom.a3 * np.exp(-q2 * atom.b3) + atom.a4 * np.exp(-q2 * atom.b4) + atom.a5 * np.exp(-q2 * atom.b5)  #- atom.dsv*0.334*np.exp(-np.pi*q2*atom.dsv**(2/3))
        else:
            return self.getGroupFormFactor(q, atom)

    def getGroupFormFactor(self, q, F):
        """
        Calculate the group form factor for a given scattering vector magnitude `q` and form factor parameters `F`.
        This function computes the form factor `f` based on the input parameters, including the scattering vector magnitude `q`,
        and the form factor coefficients `F`. It uses mathematical expressions involving exponential decay and trigonometric functions
        to calculate the result.
        Args:
            q (float): The magnitude of the scattering vector.
            F (object): An object containing form factor parameters. It is expected to have the following attributes:
                - c (float): Constant term in the form factor calculation.
                - a1, a2, a3, a4, a5 (float): Coefficients for exponential terms.
                - b1, b2, b3, b4, b5 (float): Exponential decay factors.
                - h (float): Height parameter for the form factor.
                - rh (float): Radius parameter for the form factor.
        Returns:
            float: The calculated group form factor `f`.
        Notes:
            - If `q` is zero, a small epsilon value is added to avoid division by zero in the calculation.
            - The function uses attributes from `self.fj['H']` for intermediate calculations, which should be defined in the class.
        """
        
        q2 = q * q / (16 * np.pi * np.pi)

        G = self.fj['H']
        fh = G.c + G.a1 * np.exp(-q2 * G.b1) + G.a2 * np.exp(-q2 * G.b2) + G.a3 * np.exp(-q2 * G.b3) + G.a4 * np.exp(-q2 * G.b4) + G.a5 * np.exp(-q2 * G.b5)

        fc = F.c + F.a1 * np.exp(-q2 * F.b1) + F.a2 * np.exp(-q2 * F.b2) + F.a3 * np.exp(-q2 * F.b3) + F.a4 * np.exp(-q2 * F.b4) + F.a5 * np.exp(-q2 * F.b5)
        f = 0
        if q == 0:
            epsilon = 1e-6  # Small value to avoid division by zero
            f = np.sqrt(fc * fc + F.h * F.h * fh * fh + 2 * fc * F.h * fh * np.sin(q * F.rh) / (q * F.rh + epsilon))
        else:
            f = np.sqrt(fc * fc + F.h * F.h * fh * fh + 2 * fc * F.h * fh * np.sin(q * F.rh) / (q * F.rh))
            #f = fc + F.h * fh * np.sin(q * F.rh) / (q * F.rh)
        return f

    def getGroupFormFactor2(self, q, c1, F1, c2, F2):
        """
        Calculate the combined form factor for two groups based on their individual form factors.
        This method computes the weighted sum of the form factors for two groups, F1 and F2, 
        using their respective coefficients c1 and c2. The form factor for each group is 
        determined either by a predefined formula (if `h == 0`) or by calling the 
        `getGroupFormFactor` method.
        Args:
            q (float): The scattering vector magnitude.
            c1 (float): Coefficient for the first group's form factor.
            F1 (object): An object representing the first group, containing attributes 
                         `h`, `c`, `a1` to `a5`, and `b1` to `b5`.
            c2 (float): Coefficient for the second group's form factor.
            F2 (object): An object representing the second group, containing attributes 
                         `h`, `c`, `a1` to `a5`, and `b1` to `b5`.
        Returns:
            float: The combined form factor for the two groups.
        """
        
        q2 = q * q / (16 * np.pi * np.pi)

        if F1.h == 0:
            f1 = F1.c + F1.a1 * np.exp(-q2 * F1.b1) + F1.a2 * np.exp(-q2 * F1.b2) + F1.a3 * np.exp(-q2 * F1.b3) + F1.a4 * np.exp(-q2 * F1.b4) + F1.a5 * np.exp(-q2 * F1.b5)
        else:
            f1 = self.getGroupFormFactor(q, F1)

        if F2.h == 0:
            f2 = F2.c + F2.a1 * np.exp(-q2 * F2.b1) + F2.a2 * np.exp(-q2 * F2.b2) + F2.a3 * np.exp(-q2 * F2.b3) + F2.a4 * np.exp(-q2 * F2.b4) + F2.a5 * np.exp(-q2 * F2.b5)
        else:
            f2 = self.getGroupFormFactor(q, F2)

        return c1 * f1 + c2 * f2  #- F1.dsv*0.334*np.exp(-np.pi*q2*F1.dsv**(2/3)) * F2.dsv*0.334*np.exp(-np.pi*q2*F2.dsv**(2/3))
    
    def getDummyAtomsFactor(self, q, G):
        """
        Calculate the correction factor for dummy atoms contribution to the form factor.
        This method computes the correction factor for the excluded solvent contribution 
        to the form factor, based on the dummy atoms model (nr0) used in Pepsi-SAXS.
        Args:
            q (float): The scattering vector magnitude.
            G (object): An object representing the dummy atom group, containing radius `r`.
        Returns:
            float: The correction factor for the dummy atoms contribution.
        Notes:
            - `fj[G]` is used to retrieve the specific dummy atom group properties.
            - `MEAN_ELECTRON_DENSITY` represents the average electron density.
            - The volume of the dummy atom is calculated using the formula for the volume 
              of a sphere: V = (4/3) * π * r^3.
            - The exponential term accounts for the scattering attenuation based on the 
              scattering vector and the dummy atom volume.
        """
        G = self.fj[G]
        p = self.MEAN_ELECTRON_DENSITY
        V = (4 * np.pi * G.r**3) / 3.0
        q2 = q * q
        return p * V * np.exp(-q2 * pow(V, 2.0/3.0) / (4 * np.pi))
    
    
    def getDummyAtomsFactorCorr0(self, q, G):
        """
        Calculate the correction factor for dummy atoms contribution to the form factor.
        This method computes the correction factor for the excluded solvent contribution 
        of dummy atoms according to the Pepsi-SAXS model (version nr1). Unlike version nr0, 
        this implementation uses the tabulated volume of the dummy atom.
        Args:
            q (float): The scattering vector magnitude.
            G (object): An object representing the dummy atom, containing its tabulated 
                        volume (`dsv`) and other properties.
        Returns:
            float: The correction factor for the dummy atoms contribution to the form factor.
        Notes:
            - The correction factor is calculated using the formula:
              `p * V * exp(-s2 * π * (V^(2/3)))`
              where:
              - `p` is a the mean electron density of the solvent (0.334),
              - `V` is the tabulated volume of the dummy atom,
              - `s2` is derived from the scattering vector magnitude `q`.
        """ 
        G = self.fj[G]
        p = self.MEAN_ELECTRON_DENSITY  # mean electron density of the solvent
        V = G.dsv
        s2 = q * q / (4 * np.pi * np.pi)
        factor = s2 * np.pi * pow(V, 2.0 / 3.0)
        return p * V * np.exp(-factor)
    
    def getDummyAtomsFactorFraser(self, q, G):
        """
        Calculate the correction factor for dummy atoms contribution to the form factor 
        based on Fraser et al. 1978 J. Appl. Cryst. 11, 693-694.

        This method computes the excluded solvent contribution to the form factor 
        using the mean electron density of the solvent and the radius of the dummy atoms.

        Args:
            q (float): The scattering vector magnitude.
            G (object): An object representing the dummy atom, containing its radius `r`.

        Returns:
            float: The correction factor for the dummy atoms contribution to the form factor.
        """
        G = self.fj[G]
        p = self.MEAN_ELECTRON_DENSITY  # mean electron density of the solvent
        V = pow(np.pi, 3/2) * G.r**3
        q2 = q * q / (4 * np.pi * np.pi)
        return p * V * np.exp(-q2 * np.pi * pow(V, 2.0/3.0))
    
    def getDummyAtomsFactorSvergun(self, q, G):
        """
        Calculate the correction factor for dummy atoms contribution to the form factor 
        based on the method described by Svergun et al. (1995, J. Appl. Cryst. 28, 768-773).

        This function computes the excluded solvent correction factor for dummy atoms, 
        incorporating an additional exponential factor to account for the average radius 
        of the dummy atom, referred to as the overall expansion factor in the referenced paper.

        Args:
            q (float): Scattering vector magnitude.
            G (object): An object containing properties of the dummy atom group, including:
                - dsv: Excluded solvent volume.
                - r: Radius of the dummy atom.

        Returns:
            float: The computed correction factor for the dummy atoms contribution to the form factor.
        """ 
        G = self.fj[G]
        V = G.dsv
        r0 = G.r
        s2 = q * q / (4 * np.pi * np.pi)
        factor = s2 * np.pi * pow(V, 2.0 / 3.0)
        rm= 1.62 # radius of average dummy atom
        expG = (r0/rm)**2 * np.exp(- s2 * np.pi * pow(4*np.pi/3, 3/2) * (pow(r0, 2.0) - pow(rm,2.0)))
        return expG * V * np.exp(-factor)
    
    def getHydrationShell(self, q, tessellation, limit=30):
        """
        Calculate the hydration shell form factor.

        This method computes the hydration shell form factor based on the 
        provided scattering vector `q`, tessellation data, and a limit for 
        solvent-accessible surface area (SASA). The hydration shell form factor 
        is calculated using the form factor of water molecules and the SASA 
        of the tessellation cells.

        Args:
            q (numpy.ndarray): Scattering vector values.
            tessellation (object): Tessellation object containing cell data.
            limit (int, optional): Threshold for solvent-accessible surface area 
                (SASA). Cells with SASA below this limit are excluded. Defaults to 30.
                
        Returns:
            numpy.ndarray: Hydration shell form factor values.
        """
        # Calculate the hydration shell form factor
        H = self.fj['H2O']  # Get the form factor for water
        cells = tessellation.cells
        H20_FF = self.getFormFactor(q, 'H2O')
        #sasa = np.array([aa.sas_area for aai, aa in enumerate(cells)])  #get sasa for amino acid
        sasa = np.array([aa.sas_area for aa in cells]) 
        sasa[sasa <= limit] = 0
        HS = sasa * H20_FF * np.sin(q * H.r) / (q * H.r)
        # Set values under the limit to zero
        return HS #- self.getDummyAtomsFactor(q, 'H2O')
    
    def getHydrationShell1(self, q, tessellation, amino_acids):
        #tessellation.compute(with_net=False)
        H = self.fj['H2O']
        H20_FF = self.getFormFactor(q, 'H2O')
       #sasa = np.array([tessellation.available_area(aai) for aai, aa in enumerate(amino_acids)])  #get sasa for amino acid
        #HS = sasa * H20_FF * np.sin(q * H.r) / (q * H.r)
        return H20_FF
    
    def getHydrationShell2(self, q, dgram):
        H = self.fj['H2O']
        H20_FF = self.getFormFactor(q, 'H2O')
        #print(f"Form factor: {H20_FF}, {q}")
        Hff_matrix = H20_FF #0.08259629432541811
        #print(f"Form factor matrix: {Hff_matrix}")
        if q == 0:
            return Hff_matrix*len(dgram)
        else:
            # create the qr matrix
            dq = dgram * q
            dq = dq[dq !=0]
            #print(f"Distance matrix: {dq}")
            # calculate the hydration shell form factors
            HS_full = np.array([Hff_matrix * np.sin(di) / (di) for di in dq])
            #print(HS_full, sum(HS_full))
            return np.sum(HS_full)
    
        
        
    def computeFormFactors(self, params, q):
        """
        Compute the form factors for a set of atoms based on the given parameters and q-values.
        This method calculates the form factors for atoms by subtracting the contribution
        of dummy atoms from the raw form factor values. The resulting form factors exclude
        the displaced solvent contribution.
        Args:
            params (list or array-like): A list of parameters representing the atoms.
            q (float or array-like): The q-values (momentum transfer) for which the form factors
                are computed.
        Returns:
            numpy.ndarray: An array of computed form factors for the atoms, with the dummy
            atoms contribution removed.
        """
        atomffs = np.zeros(len(params)) # form factors for the atoms
        for nr, it in enumerate(params):
            atomffs[nr] =self.getFormFactor(q, it) - self.getDummyAtomsFactorFraser(q, it) # self.getDummyAtomsFactor(q, it) # remove the dummy atoms contribution
        return atomffs # returning the exclusion form factors - have removed the displaced solvent contribution

    
    def getAAFormFactor(self, qval, atomffs, coords):
        """
        Calculate the atomic form factor for a given set of atomic coordinates and form factors.
        Parameters:
        -----------
        qval : float
            The scattering vector magnitude (q). If qval is 0, the form factor is calculated 
            as the sum of the atomic form factors.
        atomffs : numpy.ndarray
            Array of atomic form factors for each atom.
        coords : numpy.ndarray
            Array of atomic coordinates with shape (N, 3), where N is the number of atoms.
        Returns:
        --------
        numpy.ndarray
            The calculated atomic form factor for the given q value and atomic configuration.
            If qval is 0, returns a single scalar value. Otherwise, returns an array of form 
            factors for each atom.
        """
        if qval == 0:
            return np.sum(atomffs) # for q = 0, the form factor is the sum of the form factors of the atoms
        else:
            #calculate the distance matrix
            dgram = np.sqrt(np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1))
            #create the qr matrix
            dq = dgram * qval
            #calculate the form factor matrix
            aff_matrix = (atomffs[None] * atomffs[:, None])
            #create the amino acid form factors matrix
            aff_matrix = np.tile(aff_matrix, (len(coords), 1, 1))
            Factors = np.zeros_like(aff_matrix)
            # calculate the aa form factors
            diag = dq < 1e-6
            Factors[diag] = aff_matrix[diag] * (1 - 1/6 * (dq[diag])**2)
            Factors[~diag] = aff_matrix[~diag] * (np.sin(dq[~diag]) / dq[~diag])
            Factors = np.sqrt(np.sum(Factors, axis=(1,2)))
            return Factors


# Calculate distances 
def calculate_distogram(coords):
    """
    Calculate the pairwise Euclidean distance matrix (distogram) for a set of coordinates.
    Parameters:
        coords (numpy.ndarray): A NumPy array of shape (..., N, D), where N is the number of points 
                                and D is the dimensionality of each point. The ellipsis (...) allows 
                                for additional leading dimensions.
    Returns:
        numpy.ndarray: A NumPy array of shape (..., N, N) containing the pairwise Euclidean distances 
                       between all points in the input coordinates.
    """
    
    dgram = np.sqrt(np.sum(
        (coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1
    ))
    return dgram


# calculate form factor with vector operations
def getAAFormFactor_fast(dgram, q, ff, eps=1e-6):
    """
    Computes the form factor using a fast implementation of equations from the Martini paper.
    This function calculates the form factor based on interatomic distances, scattering 
    vector magnitudes, and atomic form factors. It uses a second-order Taylor expansion 
    for cases where the product of distance and scattering vector magnitude is close to zero.
    Args:
        dgram (numpy.ndarray): Distance matrix representing interatomic distances.
        q (numpy.ndarray): Scattering vector magnitudes.
        ff (numpy.ndarray): Atomic form factors.
        eps (float, optional): Small threshold value to determine near-zero distances. 
                               Default is 1e-6.
    Returns:
        numpy.ndarray: Computed form factor values.
    Notes:
        - Implements Eq.10 and Eq.11 from the Martini paper.
        - Eq.10 handles the computation of the form factor matrix using sine functions 
          and Taylor expansion for near-zero values.
        - Eq.11 adjusts the form factor for cases where the scattering vector magnitude is zero.
    """
    # distance * q
    dq = dgram[..., None] * q

    # ff in matrix form
    ffmat = (ff[None] * ff[:, None])

    # Output matrix
    F = np.zeros_like(ffmat)

    # Eq.10 Find the indices where dq ==0 and use second order taylor expansion for them
    indices_zeros = dq<eps
    indices_nonzeros = ~indices_zeros
    F[indices_nonzeros] = ffmat[indices_nonzeros] * np.sin(dq[indices_nonzeros]) / dq[indices_nonzeros]
    F[indices_zeros] = ffmat[indices_zeros] * (1 - (1/6) * (dq[indices_zeros])**2)

    # Eq.10 Get average and sqrt
    F = np.sqrt(np.sum(F, axis=(-2,-3)))

    # Eq.11
    F[..., q==0.0] = np.sum(ff[...,q==0.0])

    return F



def poly6d_fixed(x_data, y_data):
    """
    Fits a fixed 6th-degree polynomial to the given data points with a constraint 
    that the first derivative at x=0 is zero (coeffs[1] = 0).
    Parameters:
        x_data (array-like): The x-coordinates of the data points.
        y_data (array-like): The y-coordinates of the data points.
    Returns:
        function: A partial function representing the best-fit polynomial, 
                  where the coefficients are optimized to minimize the least squares error.
    Notes:
        - The optimization is performed using constrained minimization, ensuring that 
          the first derivative at x=0 is zero by fixing the second coefficient (coeffs[1]).
        - The initial guess for the coefficients is generated randomly.
    """
    def poly_func(coeffs, x):
        return sum(c * x**i for i, c in enumerate(coeffs))

    # Define the objective function (least squares error)
    def objective(coeffs):
        return np.sum((poly_func(coeffs, x_data) - y_data) ** 2)

    # Initial guess (7 coefficients for a 6th-degree polynomial)
    initial_guess = np.random.randn(7)

    # Constraint: a_1 = coeffs[1] = 0 (fix derivative at x=0)
    constraints = {'type': 'eq', 'fun': lambda coeffs: coeffs[1]}  # Forces coeffs[1] = 0

    # Solve using constrained optimization
    result = minimize(objective, initial_guess, constraints=constraints)

    # Extract best-fit coefficients
    best_fit_coeffs = result.x

    return partial(poly_func, coeffs=best_fit_coeffs)