import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power


class TED(object):
    """
    This class will be used to print out total energy distributions
    from internal coordinate eigenvalue matrices. If redundant
    coordinates are utilized, projection matrices will be stored here.
    """

    def __init__(self, proj, zmat):
        self.proj = proj
        self.zmat = zmat

    def run(self, eigs, freq, rect_print=True):

        proj_eigs = eigs
        if len(np.shape(self.proj)) > 2 and np.shape(self.proj)[0] > 1:
            proj_buff = self.proj[0]
            for i in range(len(self.proj) - 1):
                proj_buff = np.append(proj_buff, self.proj[i + 1], axis=1)
            print(proj_buff)
            raise RuntimeError
        if rect_print:
            proj_eigs = np.dot(self.proj, proj_eigs)
        proj_eigs_inv = LA.pinv(proj_eigs)
        self.TED = np.multiply(proj_eigs, proj_eigs_inv.T) * 100
        self.ted_breakdown = self.TED
        self.table_print(freq, self.TED, rect_print)

    def table_print(self, freq, TED, rect_print):
        if len(freq) != len(TED[0]):
            print(
                "Something has gone terribly wrong. Your Total Energy \
                    distribution should have the same number of columns as \
                    frequencies, but it does not."
            )
            print("TED columns: " + str(len(TED[0])))
            print("# frequencies: " + str(len(freq)))
            raise RuntimeError
        div = 15
        a = len(freq) // div
        for i in range(a):
            print(
                self.sub_table(freq[i * div : (i + 1) * div], TED, i, div, rect_print)
            )
        if len(freq) % div:
            print(self.sub_table(freq[a * div :], TED, a, div, rect_print))

    def sub_table(self, freq, TED, int_div, div, rect_print):
        table_output = "{:>26s}".format("frequency #: ")
        n = len(freq)
        for l in range(n):
            table_output += "{:8d}".format(l + int_div * div + 1)
        table_output += "\n"
        table_output += "--------------------------"
        for l in range(n):
            table_output += "--------"
        table_output += "\n"
        table_output += "{:>26s}".format("frequency: ")
        for l in range(n):
            table_output += " " + "{:7.1f}".format(freq[l])
        table_output += "\n"
        table_output += "--------------------------"
        for l in range(n):
            table_output += "--------"
        table_output += "\n"
        if rect_print:
            for i in range(len(TED)):
                for j in range(len(freq)):
                    if i < len(self.zmat.bond_indices) and j == 0:
                        table_output += (
                            "{:10s}".format(" ")
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.bond_indices[i][0]) - 1
                                    ]
                                )
                                + str(self.zmat.bond_indices[i][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.bond_indices[i][1]) - 1
                                    ]
                                )
                                + str(self.zmat.bond_indices[i][1])
                            )
                            + " STRE: "
                        )
                    elif (
                        i < len(self.zmat.bond_indices) + len(self.zmat.angle_indices)
                        and j == 0
                    ):
                        k = i - len(self.zmat.bond_indices)
                        table_output += (
                            "{:5s}".format(" ")
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.angle_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.angle_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.angle_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.angle_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.angle_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.angle_indices[k][2])
                            )
                            + " BEND: "
                        )
                    elif (
                        i
                        < len(self.zmat.bond_indices)
                        + len(self.zmat.angle_indices)
                        + len(self.zmat.torsion_indices)
                        and j == 0
                    ):
                        k = (
                            i
                            - len(self.zmat.bond_indices)
                            - len(self.zmat.angle_indices)
                        )
                        table_output += (
                            "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.torsion_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.torsion_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.torsion_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.torsion_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.torsion_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.torsion_indices[k][2])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.torsion_indices[k][3]) - 1
                                    ]
                                )
                                + str(self.zmat.torsion_indices[k][3])
                            )
                            + " TORS: "
                        )
                    elif (
                        i
                        < len(self.zmat.bond_indices)
                        + len(self.zmat.angle_indices)
                        + len(self.zmat.torsion_indices)
                        + len(self.zmat.oop_indices)
                        and j == 0
                    ):
                        k = (
                            i
                            - len(self.zmat.bond_indices)
                            - len(self.zmat.angle_indices)
                            - len(self.zmat.torsion_indices)
                        )
                        table_output += (
                            "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.oop_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.oop_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.oop_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.oop_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.oop_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.oop_indices[k][2])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.oop_indices[k][3]) - 1
                                    ]
                                )
                                + str(self.zmat.oop_indices[k][3])
                            )
                            + "  OOP: "
                        )
                    elif (
                        i
                        < len(self.zmat.bond_indices)
                        + len(self.zmat.angle_indices)
                        + len(self.zmat.torsion_indices)
                        + len(self.zmat.oop_indices)
                        + len(self.zmat.lin_indices)
                        and j == 0
                    ):
                        k = (
                            i
                            - len(self.zmat.bond_indices)
                            - len(self.zmat.angle_indices)
                            - len(self.zmat.torsion_indices)
                            - len(self.zmat.oop_indices)
                        )
                        table_output += (
                            "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.lin_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.lin_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.lin_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.lin_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.lin_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.lin_indices[k][2])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.lin_indices[k][3]) - 1
                                    ]
                                )
                                + str(self.zmat.lin_indices[k][3])
                            )
                            + "  LIN: "
                        )
                    elif (
                        i
                        < len(self.zmat.bond_indices)
                        + len(self.zmat.angle_indices)
                        + len(self.zmat.torsion_indices)
                        + len(self.zmat.oop_indices)
                        + len(self.zmat.lin_indices)
                        + len(self.zmat.linx_indices)
                        and j == 0
                    ):
                        k = (
                            i
                            - len(self.zmat.bond_indices)
                            - len(self.zmat.angle_indices)
                            - len(self.zmat.torsion_indices)
                            - len(self.zmat.oop_indices)
                            - len(self.zmat.lin_indices)
                        )
                        table_output += (
                            "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.linx_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.linx_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.linx_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.linx_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.linx_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.linx_indices[k][2])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.linx_indices[k][3]) - 1
                                    ]
                                )
                                + str(self.zmat.linx_indices[k][3])
                            )
                            + " LINX: "
                        )
                    elif (
                        i
                        < len(self.zmat.bond_indices)
                        + len(self.zmat.angle_indices)
                        + len(self.zmat.torsion_indices)
                        + len(self.zmat.oop_indices)
                        + len(self.zmat.lin_indices)
                        + len(self.zmat.linx_indices)
                        + len(self.zmat.liny_indices)
                        and j == 0
                    ):
                        k = (
                            i
                            - len(self.zmat.bond_indices)
                            - len(self.zmat.angle_indices)
                            - len(self.zmat.torsion_indices)
                            - len(self.zmat.oop_indices)
                            - len(self.zmat.lin_indices)
                            - len(self.zmat.linx_indices)
                        )
                        table_output += (
                            "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.liny_indices[k][0]) - 1
                                    ]
                                )
                                + str(self.zmat.liny_indices[k][0])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.liny_indices[k][1]) - 1
                                    ]
                                )
                                + str(self.zmat.liny_indices[k][1])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.liny_indices[k][2]) - 1
                                    ]
                                )
                                + str(self.zmat.liny_indices[k][2])
                            )
                            + " "
                            + "{:4s}".format(
                                str(
                                    self.zmat.atom_list[
                                        int(self.zmat.liny_indices[k][3]) - 1
                                    ]
                                )
                                + str(self.zmat.liny_indices[k][3])
                            )
                            + " LINY: "
                        )
                    table_output += "{:8.1f}".format(TED[i][j + div * int_div])
                table_output += "\n"
        else:
            for i in range(len(TED)):
                for j in range(len(freq)):
                    if not j:
                        table_output += (
                            "{:>21s}".format("Mode #")
                            + "{:>3s}".format(str(i + 1))
                            + "  "
                        )
                    table_output += "{:8.1f}".format(TED[i][j + div * int_div])
                table_output += "\n"

        return table_output
