# formfactor.py
# Author: Isabel Vinterbladh
# This file contains the class cSaXSparameters which is used to calculate the form factors of the atoms and then amino acids in the protein structure.
import numpy as np
import cmath
from scipy.optimize import minimize
from functools import partial
import tqdm
### Class cSaXSparameters ###
# This class is used to calculate the form factors of the atoms and then amino acids in the protein structure.
# The class contains the following methods:
# 1. __init__: Initializes the class with the given parameters.
# 2. paramsFormFactors: Initializes the parameters for the form factors of the atoms.
# 3. getFormFactor: Calculates the form factor of an atom given the q value and the atom type.
# 4. getGroupFormFactor: Calculates the form factor of a group of atoms given the q value and the group type.
# 5. getGroupFormFactor2: Calculates the form factor of a group of atoms given the q value and the group type. - Special case for Arginine.
# 6. getDummyAtomsFactor: Calculates the form factor of the dummy atoms given the q value and the atom type. 
# 7. getDummyAtomsFactorCorr0: Calculates the form factor of the dummy atoms given the q value and the atom type.
# 8. computeFormFactors: Computes the form factors of the atoms in the protein structure. 
# 9. getAAFormFactor: Calculates the form factor of the amino acids in the protein structure. ref. Single bead approximation 2.1.4
# 10. getAAFormFactor212: Calculates the form factor of the amino acids in the protein structure. ref. Spherical glob approximation 2.1.2 
# 11. getAAFormFactorDummy: Calculates the form factor of the dummy atoms in the protein structure.
# 12. getr_kl: Calculates the position vector of an atom in the structure relative to the center of mass.

class cSAXSparameters:
    def __init__(self):
        self.fj = {}
        self.paramsFormFactors()

    def paramsFormFactors(self):
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
                
        fourPi = 4 * np.pi
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
        q2 = q * q / (4 * np.pi * np.pi)   
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
        q2 = q * q / (4 * np.pi * np.pi)

        G = self.fj['H']
        fh = G.c + G.a1 * np.exp(-q2 * G.b1) + G.a2 * np.exp(-q2 * G.b2) + G.a3 * np.exp(-q2 * G.b3) + G.a4 * np.exp(-q2 * G.b4) + G.a5 * np.exp(-q2 * G.b5)

        fc = F.c + F.a1 * np.exp(-q2 * F.b1) + F.a2 * np.exp(-q2 * F.b2) + F.a3 * np.exp(-q2 * F.b3) + F.a4 * np.exp(-q2 * F.b4) + F.a5 * np.exp(-q2 * F.b5)
        f = 0
        if q == 0:
            f = fc + F.h * fh
        else:
            #f = np.sqrt(fc * fc + F.h * F.h * fh * fh + 2 * fc * F.h * fh * np.sin(q * F.rh) / (q * F.rh))
            f = fc + F.h * fh * np.sin(q * F.rh) / (q * F.rh)
        return f

    def getGroupFormFactor2(self, q, c1, F1, c2, F2):
        q2 = q * q / (4 * np.pi * np.pi)

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
        G = self.fj[G]
        p = 0.334
        V = (4 * np.pi * G.r**3) / 3.0
        q2 = q * q
        return p * V * np.exp(-q2 * pow(V, 2.0/3.0) / (4 * np.pi))

    def getDummyAtomsFactorCorr0(self, q, G):
        G = self.fj[G]
        p = 0.334
        V = G.dsv
        q2 = q * q / (4 * np.pi * np.pi)
        factor = q2 * np.pi * pow(V, 2.0 / 3.0)
        return p * V * np.exp(-factor)

    def computeFormFactors(self, params, q):
        atomffs = np.zeros(len(params)) # form factors for the atoms
        for nr, it in enumerate(params):
            atomffs[nr] = self.getFormFactor(q, it) #- self.getDummyAtomsFactorCorr0(q, it) # remove the dummy atoms contribution
        return atomffs # returning the exclusion form factors - have removed the displaced solvent contribution

    
    def getAAFormFactor(self, qval, atomffs, params, coords):
        Factors = np.zeros_like(atomffs)
        dgram = np.sqrt(np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1))
        print(dgram.shape)
        aff_matrix = (atomffs[None] * atomffs[:, None])
        ffmat = (atomffs[..., None, :, :] * atomffs[..., :, None, :])
        print("FF:", aff_matrix.shape, ffmat.shape, "atomffs:", atomffs.shape)
        print("Factors:", Factors.shape)
        #for itr, qval in enumerate(q):
        
        
        for i in tqdm.tqdm(range(dgram.shape[0])):
            dq = dgram[i][...,None] * qval[...,None,:]
            print("dq", dq.shape, qval[...,None,:].shape)
            FF = atomffs * atomffs[i][None]
            zeros = dq < 1e-6
            print(zeros.shape, Factors.shape)
            Factors[zeros] += FF[zeros] * (1)
            Factors[~zeros] += aff_matrix[~zeros] * (np.sin(dq[~zeros]) / dq[~zeros])
            
        
        #Factors[...,qval==0] = np.sum(atomffs[...,qval==0]) # for q = 0
        return np.sqrt(np.sum(Factors, axis=(-2)))


# Calculate distances 
def calculate_distogram(coords):
    dgram = np.sqrt(np.sum(
        (coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1
    ))
    return dgram


# calculate form factor with vector operations
def getAAFormFactor_fast(dgram, q, ff, eps=1e-6):
    # implements Eq.10 and Eq.11

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
    # Define polynomial function (degree 6)
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