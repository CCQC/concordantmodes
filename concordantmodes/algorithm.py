import numpy as np
import os
import scipy
from scipy import stats


class Algorithm:
    """
    Generate force-constant displacement indices for the Concordant Modes
    Algorithm (CMA).

    This class determines which coordinate pairs `(i, j)` will be evaluated
    during finite-difference force-constant calculations. The generated index
    list depends on:

    - The CMA level being performed (Level A or Level B).
    - Whether force constants are computed from energies or gradients.
    - Whether molecular symmetry is enabled.
    - The irreducible-representation (irrep) structure of the projected
      vibrational coordinate space.

    For Level B calculations without gradient-based Hessian construction,
    off-diagonal force constants may be included. For Level A calculations
    (or when gradient derivatives are used), only diagonal force constants
    are generated. When symmetry is enabled, indices are restricted to
    coordinate pairs belonging to the same irreducible representation,
    reducing the number of required electronic structure calculations.

    Parameters
    ----------
    num_deg_free : int
        Number of vibrational degrees of freedom in the projected coordinate
        space.
    cma_level : {"A", "B"}
        CMA stage for which displacement indices are generated.

        - ``"B"``: Lower-level reference Hessian calculation.
        - ``"A"``: Higher-level CMA correction calculation.

    options : object
        User options containing symmetry settings and derivative-level
        controls. The following attributes are used:

        - ``molsym_symmetry`` : bool
        - ``deriv_level_b`` : bool

    proj_irreps : list, optional
        Symmetry decomposition of the projected coordinate space. Each
        element specifies the number of coordinates belonging to an
        irreducible representation. Degenerate irreducible
        representations may be represented as nested lists.

    Attributes
    ----------
    indices : list[list[int]]
        Flat list of coordinate index pairs to be displaced and evaluated.

    indices_by_irrep : list or None
        Index pairs grouped by irreducible representation when symmetry
        is enabled. Otherwise ``None``.

    degens : list or None
        Placeholder for degeneracy information. Currently set to ``None``
        when symmetry grouping is not used.

    Notes
    -----
    Four index-generation strategies are available:

    ``loop()``
        Generate all upper-triangular coordinate pairs without symmetry.

    ``loop_diagonal()``
        Generate only diagonal coordinate pairs without symmetry.

    ``loop_symmetry()``
        Generate upper-triangular coordinate pairs within each irreducible
        representation.

    ``loop_symmetry_diagonal()``
        Generate only diagonal coordinate pairs within each irreducible
        representation.

    The :meth:`run` method automatically selects the appropriate strategy
    based on the CMA level, derivative level, and symmetry settings.
    """

    def __init__(self, num_deg_free, cma_level, options, proj_irreps=None):
        self.num_deg_free = num_deg_free
        self.cma_level = cma_level
        self.options = options
        self.proj_irreps = proj_irreps

    def run(self):
        """
        Generate displacement indices for the current CMA calculation.

        Selects and executes the appropriate index-generation algorithm based
        on the symmetry settings, CMA level, and derivative level specified
        in the options object.

        Selection logic
        ---------------
        Symmetry enabled
            * Level B, energy finite differences:
              :meth:`loop_symmetry`
            * Level A or gradient finite differences:
              :meth:`loop_symmetry_diagonal`

        Symmetry disabled
            * Level B, energy finite differences:
              :meth:`loop`
            * Level A or gradient finite differences:
              :meth:`loop_diagonal`

        Returns
        -------
        None

        Notes
        -----
        Results are stored in the instance attributes ``indices`` and
        ``indices_by_irrep``.
        """
        if self.options.molsym_symmetry:
            if self.cma_level == "A" or self.options.deriv_level_b:
                self.loop_symmetry_diagonal()
            else:
                self.loop_symmetry()
        else:
            if self.cma_level == "A" or self.options.deriv_level_b:
                self.loop_diagonal()
            else:
                self.loop()

    # Indices for high level A looping or deriv_level_b == 1
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
                        irrep_indices.append([i, i])
                    degen_list.append(irrep_indices)
                    offset += irrepl
                self.indices_by_irrep.append(degen_list)
            else:
                for i in range(offset, irrep + offset):
                    irrep_indices.append([i, i])
                self.indices_by_irrep.append(irrep_indices)
                offset += irrep
        for i, irrep_ind in enumerate(self.indices_by_irrep):
            if type(self.proj_irreps[i]) is list:
                self.indices.append(irrep_ind[0])
            else:
                self.indices.append(irrep_ind)
        self.indices = [item for sublist in self.indices for item in sublist]

    # Indices for level B where deriv_level_b == 0
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
                            irrep_indices.append([i, j])
                    degen_list.append(irrep_indices)
                    offset += irrepl
                self.indices_by_irrep.append(degen_list)
            else:
                for i in range(offset, irrep + offset):
                    for j in range(i, irrep + offset):
                        irrep_indices.append([i, j])
                self.indices_by_irrep.append(irrep_indices)
                offset += self.proj_irreps[h]
        for i, irrep_ind in enumerate(self.indices_by_irrep):
            if type(self.proj_irreps[i]) is list:
                self.indices.append(irrep_ind[0])
            else:
                self.indices.append(irrep_ind)
        self.indices = [item for sublist in self.indices for item in sublist]

    def loop_diagonal(self):
        self.indices = []
        for i in range(self.num_deg_free):
            self.indices.append([i, i])
        self.indices_by_irrep = None
        self.degens = None

    # Generates level B indices where no symmetry is being used
    def loop(self):
        if self.cma_level == "A":
            addem = 1
        else:
            addem = self.num_deg_free
        self.indices = []
        for i in range(self.num_deg_free):
            for j in range(i, i + addem):
                if j > self.num_deg_free - 1:
                    break
                else:
                    self.indices.append([i, j])
        self.indices_by_irrep = None
        self.degens = None
