import numpy as np
from concordantmodes.s_vectors import SVectors
from concordantmodes.symmetry import Symmetry
from concordantmodes.zmat import Zmat
import logging
from concordantmodes.errors import CMAError
from concordantmodes.log import debug_print, info_print

logger = logging.getLogger(__name__)


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
            raise CMAError("geom_check is set: stopping after reading the zmat file.")

        # Do we want to use symmetry? Default is False
        self.symm_obj = Symmetry(self.zmat_obj, self.options, self.proj)
        if self.options.molsym_symmetry:
            self.symm_obj.run()
            debug_print(logger, "The symtext")
            debug_print(logger, self.symm_obj.symtext)
            self.symm_obj.molsym_salcs_ic()
            debug_print(logger, self.symm_obj.sym_sort)
            self.sym_sort = self.symm_obj.sym_sort
            debug_print(logger, "The sym sort")
            debug_print(logger, self.sym_sort)
