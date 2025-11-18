import numpy as np
import scipy as sp
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
import copy

class GFMethod(object):
    """
    This class is a versatile method to compute GF frequencies.

    TODO: Insert standard uncertainties of amu_elmass and HARTREE_WAVENUM
    """

    def __init__(self, G, F, zmat, ted, options, symtext = None, cma=None, sym_sort=[]):
        self.G = G
        self.F = F
        self.zmat = zmat
        self.ted = ted
        self.options = options
        self.symtext = symtext
        self.cma = cma
        self.sym_sort = sym_sort
        self.AMU_ELMASS = 5.48579909065 * (10 ** (-4))
        self.HARTREE_WAVENUM = 219474.6313708

    def run(self):
        # Construct the orthogonalizer
        self.G_O = fractional_matrix_power(self.G, 0.5)

        # Symmetrize F, diagonalize, then backtransform the eigenvectors.
        self.F_O = np.dot(np.dot(self.G_O, self.F), self.G_O)
        # self.F_O[np.abs(self.F_O) < 1.0e-8] = 0
        # self.F_O[np.abs(self.F_O) < self.options.tol] = 0
   
        if self.options.molsym_symmetry:
            #allows the block diagonalization of the GF matrix so normal modes stay
            #in the assumed ordering for the level A displacements (Within their symmetry blocks)
            self.block_GF()
        else:
            # self.eig_v, self.L_p = sp.linalg.eigh(self.F_O)
            self.eig_v, self.L_p = LA.eigh(self.F_O)
        
        self.L_p[np.abs(self.L_p) < self.options.tol] = 0
        self.L = np.dot(self.G_O, self.L_p)
        self.L = np.real(self.L)
        # L = np.absolute(self.L)
        L_p = np.real(self.L_p)
        self.L_p = L_p

        self.L[np.abs(self.L) < self.options.tol] = 0
        # del_tol = 1.0e-3
        # del_tol = 1.0e-2
        # self.L = self.L.T
        # for row in self.L:
           # abs_row = np.abs(row)
           # row[abs_row < np.max(abs_row)*del_tol] = 0
        # self.L = self.L.T
        # Compute the frequencies by the square root of the eigenvalues.
        self.freq = np.sqrt(self.eig_v, dtype=complex)
        # Filter for imaginary modes.
        for i in range(len(self.freq)):
            if np.real(self.freq[i]) > 0.0:
                self.freq[i] = np.real(self.freq[i])
            else:
                self.freq[i] = -np.imag(self.freq[i])

        # This seems to fix the constant warning I get about casting out imaginary values from the complex numbers,
        # however I will need to test it on transition states to see if it captures the imaginary frequencies.
        self.freq = self.freq.astype(float)

        # Convert from Hartrees to wavenumbers.
        self.freq *= self.HARTREE_WAVENUM
        if self.options.molsym_symmetry:
            for i in range(len(self.freq)):
                print(
                    "frequency #"
                    + "{:3d}".format(i + 1)
                    + ": "
                    + "{:10.2f}".format(self.freq[i])
                    + " {}".format(self.irrep_degen[i])
                    + " x {}".format(self.irrep_labels[i])
                )
        else:
            for i in range(len(self.freq)):
                print(
                    "frequency #"
                    + "{:3d}".format(i + 1)
                    + ": "
                    + "{:10.2f}".format(self.freq[i])
                )
        # Compute and then print the TED.
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format("Total Energy Distribution (TED)"))
        print("////////////////////////////////////////////")
        if self.options.molsym_symmetry:
            self.ted.run(self.L, self.freq, self.symtext, rect_print=False)
        else:
            self.ted.run(self.L, self.freq, self.symtext, rect_print=False)

    def block_GF(self):
        self.eigvals, self.eigvecs = [], []
        offset_h = 0
        self.block_fo = []
        self.irrep_labels = []
        self.irrep_degen = []
        for hi, h in enumerate(self.symtext.salcblocks):
            
            #Extract the block from the supermatrix
            fo = self.extract(self.F_O, offset_h, h.shape[0])
            self.block_fo.append(fo)
            eig_v_h, L_p_h = LA.eigh(fo)
            self.eigvals.append(eig_v_h)
            self.eigvecs.append(L_p_h)
            for i in range(len(eig_v_h)):
                self.irrep_labels.append(self.symtext.irreps[hi].symbol)
                self.irrep_degen.append(self.symtext.irreps[hi].d)

            # UPDATE OFFSET
            offset_h += h.shape[0]
        self.block_eigvals = copy.deepcopy(self.eigvals)
        self.eigvals = [item for sublist in self.eigvals for item in sublist]
        self.eig_v = np.asarray(self.eigvals)
        self.block_eigvecs = copy.deepcopy(self.eigvecs)
        self.L_p = block_diag(*self.eigvecs)
  
     #function to extract bd matrices from supermatrix
    def extract(self, A, offset, size):
        return A[offset:offset+size, offset:offset+size]
