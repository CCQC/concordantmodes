import numpy as np
from numpy import linalg as LA


class TED:
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
