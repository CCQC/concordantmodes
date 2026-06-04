import numpy as np
from concordantmodes.s_vectors import SVectors
from concordantmodes.symmetry import Symmetry
from concordantmodes.zmat import Zmat


class SymHelper:
    """
    This class handles sym_sort generation and
    helper functions for natural internal coordinates
    via projection operator in MolSym
    """

    def __init__(self, options, proj=None, extra_indices=np.array([])):
        self.options = options
        self.proj = proj
        self.extra_indices = extra_indices

    def run(self):
        # Parse the output to get all pertinent ZMAT info
        self.zmat_obj = Zmat(self.options)
        self.zmat_obj.run()
        if self.options.geom_check:
            raise RuntimeError

        # Do we want to use symmetry? Default is False
        self.symm_obj = Symmetry(self.zmat_obj, self.options, self.proj)
        if self.options.molsym_symmetry:
            self.symm_obj.run()
            print("The symtext")
            print(self.symm_obj.symtext)
            self.symm_obj.molsym_salcs_ic()
            print(self.symm_obj.sym_sort)
            self.sym_sort = self.symm_obj.sym_sort
            print("The sym sort")
            print(self.sym_sort)
