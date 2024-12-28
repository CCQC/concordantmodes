import numpy as np
import os
import scipy
from scipy import stats


class Algorithm(object):
    """
    The purpose of this class is to return a list of indices by which the force constants of the CMA method
    will be computed. These indices will be determined by user input or by a scoring function which takes into
    consideration the overlap of the normal coordinates and the difference in force constants for a particular normal
    mode.
    """

    def __init__(self, eigs, level_A, options, proj_irreps):
        self.eigs = eigs
        self.level_A = level_A
        self.options = options
        self.proj_irreps = proj_irreps

    def run(self):
        if self.options.symmetry:
            if self.level_A or self.options.deriv_level_init:
                self.loop_symmetry_diagonal()
            else:
                self.loop_symmetry()
        else:
            self.loop()
    def loop_symmetry_diagonal(self):
        self.indices = []
        self.indices_by_irrep = []
        offset = 0
        for h, irrep in enumerate(self.proj_irreps):
            irrep_indices = []
            if type(irrep) is list:
                degen_list = []
                for irrepl in irrep:
                    irrep_indices = []
                    for i in range(offset, irrepl + offset):
                        irrep_indices.append([i,i])
                    degen_list.append(irrep_indices)
                    offset += irrepl
                self.indices_by_irrep.append(degen_list)
            else:
                for i in range(offset, irrep + offset):
                    irrep_indices.append([i,i])
                self.indices_by_irrep.append(irrep_indices)
                offset += irrep
        for i, irrep_ind in enumerate(self.indices_by_irrep):
            if type(self.proj_irreps[i]) is list:
                self.indices.append(irrep_ind[0])
            else:
                self.indices.append(irrep_ind)
        self.indices = [item for sublist in self.indices for item in sublist]


    def loop_symmetry(self):
        self.indices = []
        self.indices_by_irrep = []
        offset = 0
        for h, irrep in enumerate(self.proj_irreps):
            irrep_indices = []
            if type(irrep) is list:
                degen_list = []
                for irrepl in irrep:
                    irrep_indices = []
                    for i in range(offset, irrepl + offset):
                        for j in range(i, irrepl + offset):
                            irrep_indices.append([i,j])
                    degen_list.append(irrep_indices)
                    offset += irrepl
                self.indices_by_irrep.append(degen_list)
            else:
                for i in range(offset, irrep + offset):
                    for j in range(i, irrep + offset):
                        irrep_indices.append([i,j])
                self.indices_by_irrep.append(irrep_indices)
                offset += self.proj_irreps[h]
        for i, irrep_ind in enumerate(self.indices_by_irrep):
            if type(self.proj_irreps[i]) is list:
                self.indices.append(irrep_ind[0])
            else:
                self.indices.append(irrep_ind)
        self.indices = [item for sublist in self.indices for item in sublist]

    def loop(self):
        print("algorithm loop, no symmetry please")
        if self.level_A:
            addem = 1
        else:
            addem = self.eigs
        self.indices = []
        #print(f"num modes {self.eigs}")
        for i in range(self.eigs):
            for j in range(i, i + addem):
                if j > self.eigs - 1:
                    break
                else:
                    if i == j:
                        self.indices.append([i, j])
                    elif i != j:
                        self.indices.append([i, j])
        self.indices_by_irrep = None
        self.degens = None