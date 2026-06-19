import numpy as np
from numpy import linalg as LA


class TED:
    """
    Compute and print the Total Energy Distribution (TED) for vibrational modes.

    The TED quantifies the contribution of each internal coordinate to each
    normal mode and is expressed as a percentage. Contributions are computed
    from the normal-mode eigenvector matrix and its pseudoinverse according
    to the standard TED formalism:

        TED_ij = L_ij * (L^-1)_ji * 100

    where ``L`` is the normal-mode transformation matrix. The resulting TED
    matrix provides a chemically intuitive decomposition of vibrational
    motions into stretches, bends, torsions, out-of-plane bends, and linear
    bending coordinates.

    Parameters
    ----------
    proj : ndarray
        Projection matrix used to transform eigenvectors from a reduced or
        symmetry-adapted coordinate basis back into the full internal
        coordinate representation.
    zmat : Zmat
        Molecular coordinate object containing internal-coordinate
        definitions and indexing information used for TED labeling.
    options : Options
        Runtime options controlling output formatting and coordinate
        handling.

    Attributes
    ----------
    proj : ndarray
        Projection matrix relating projected coordinates to the full
        internal-coordinate basis.
    zmat : Zmat
        Molecular coordinate representation.
    options : Options
        User-defined runtime options.
    symtext : molsym.Symtext or None
        Molecular symmetry information used to annotate TED tables with
        irreducible representation labels.
    TED : ndarray
        Total Energy Distribution matrix expressed as percentages.

    Notes
    -----
    The TED matrix is computed from the projected normal-mode eigenvectors
    and their Moore-Penrose pseudoinverse. Each column corresponds to a
    vibrational normal mode and each row corresponds to an internal
    coordinate.

    When ``rect_print=True``, the eigenvectors are first transformed back
    into the full internal-coordinate basis using the projection matrix
    before TED analysis. This is typically used when redundant or
    symmetry-adapted coordinates have been employed during the vibrational
    calculation.

    Coordinate labels are automatically generated from the Z-matrix
    coordinate definitions using the following conventions:

    - B  : Bond stretches
    - A  : Bond angles
    - D  : Dihedral/torsional angles
    - O  : Out-of-plane bends
    - L  : Linear bends
    - Lx : Linear bend x-components
    - Ly : Linear bend y-components

    If molecular symmetry information is available, TED tables include
    point-group and irreducible-representation assignments for each
    vibrational mode.

    Examples
    --------
    >>> ted = TED(proj, zmat, options)
    >>> ted.run(L_matrix, frequencies)

    Print TED using symmetry-adapted normal modes:

    >>> ted.run(L_matrix, frequencies, symtext=symtext)

    The resulting table reports the percentage contribution of each
    internal coordinate to every vibrational mode.
    """

    def __init__(self, proj, zmat, options):
        self.proj = proj
        self.zmat = zmat
        self.options = options

    def run(self, eigs, freq, symtext=None, rect_print=True):
        self.symtext = symtext

        proj_eigs = np.dot(self.proj, eigs) if rect_print else eigs
        proj_eigs_inv = LA.pinv(proj_eigs)

        print("The eigenvectors (check these for phase):")
        print(proj_eigs)

        self.TED = np.multiply(proj_eigs, proj_eigs_inv.T) * 100
        self.table_print(freq, self.TED, rect_print)

    def table_print(self, freq, TED, rect_print):
        if len(freq) != TED.shape[1]:
            print(
                "Something has gone terribly wrong. Your Total Energy "
                "distribution should have the same number of columns as "
                "frequencies, but it does not."
            )
            print("TED columns: " + str(TED.shape[1]))
            print("# frequencies: " + str(len(freq)))
            raise RuntimeError

        div = 15
        chunks = [freq[i : i + div] for i in range(0, len(freq), div)]

        for idx, chunk in enumerate(chunks):
            print(self.sub_table(chunk, TED, idx, div, rect_print))

    def _coord_label(self, i):
        """Return formatted coordinate label prefix (exact same formatting)."""
        z = self.zmat

        bounds = [
            (len(z.bond_indices), "B", " STRE: "),
            (len(z.angle_indices), "A", " BEND: "),
            (len(z.torsion_indices), "D", " TORS: "),
            (len(z.oop_indices), "O", "  OOP: "),
            (len(z.lin_indices), "L", "  LIN: "),
            (len(z.linx_indices), "Lx", " LINX: "),
            (len(z.liny_indices), "Ly", " LINY: "),
        ]

        offset = 0
        for size, label, suffix in bounds:
            if i < offset + size:
                k = i - offset
                return (
                    "{:15s}".format(" ") + "{:4s}".format(label + str(k + 1)) + suffix
                )
            offset += size

        return "{:15s}".format(" ")

    def sub_table(self, freq, TED, int_div, div, rect_print):
        n = len(freq)

        table_output = "{:>26s}".format("frequency #: ")
        for l in range(n):
            table_output += "{:8d}".format(l + int_div * div + 1)
        table_output += "\n"

        table_output += "--------------------------" + "--------" * n + "\n"

        irrep_labels = []
        if self.symtext is not None:
            table_output += "Point Group " + str(self.symtext.pg) + "           "
            for ir, h in enumerate(self.symtext.salcblocks):
                irrep_labels.extend([str(self.symtext.irreps[ir].symbol)] * h.shape[0])
            table_output += "\n"

        table_output += "{:>26s}".format("frequency: ")
        for val in freq:
            table_output += " " + "{:7.1f}".format(val)
        table_output += "\n"

        if self.symtext is not None:
            table_output += "{:>26s}".format("Irreps: ")
            for l in range(n):
                table_output += " " + "{:>7}".format(irrep_labels[l])
            table_output += "\n"

        table_output += "--------------------------" + "--------" * n + "\n"

        if rect_print:
            for i in range(len(TED)):
                for j in range(n):
                    if j == 0:
                        table_output += self._coord_label(i)
                    table_output += "{:8.1f}".format(TED[i][j + div * int_div])
                table_output += "\n"
        else:
            for i in range(len(TED)):
                for j in range(n):
                    if j == 0:
                        table_output += (
                            "{:>21s}".format("Mode #")
                            + "{:>3s}".format(str(i + 1))
                            + "  "
                        )
                    table_output += "{:8.1f}".format(TED[i][j + div * int_div])
                table_output += "\n"

        return table_output
