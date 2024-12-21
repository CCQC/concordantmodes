from concordantmodes.molden_template import molden_template
import numpy as np
import json
import os
import qcelemental as qcel
import re
import shutil


class MoldenWriter(object):
    def __init__(
        self,
        zmat,
        disps,
        freq,
        # prog_name,
    ):
        # self.prog_name = prog_name
        self.zmat = zmat
        self.disps = disps
        self.freq = freq

    def run(self):
        geometry = ""
        atom_num = np.array([])
        for i in self.zmat.atom_list:
            atom_num = np.append(atom_num, int(qcel.periodictable.to_atomic_number(i)))

        for i in range(len(self.zmat.cartesians_final)):
            geometry += "{:2<}{:>9}{:>5}{:-20.10f}{:-20.10f}{:-20.10f}\n".format(
                self.zmat.atom_list[i],
                str(i + 1),
                str(int(atom_num[i])),
                self.zmat.cartesians_final[i][0],
                self.zmat.cartesians_final[i][1],
                self.zmat.cartesians_final[i][2],
            )

        frequencies = ""
        for i in self.freq:
            frequencies += "{:-20.10f}\n".format(i)

        fr_geom = ""

        for i in range(len(self.zmat.cartesians_final)):
            fr_geom += "{:<5}{:-20.10f}{:-20.10f}{:-20.10f}\n".format(
                self.zmat.atom_list[i],
                self.zmat.cartesians_final[i][0],
                self.zmat.cartesians_final[i][1],
                self.zmat.cartesians_final[i][2],
            )

        normal_modes = ""
        for i in range(len(self.disps.p_disp)):
            normal_modes += "  vibration{:>23}\n".format(str(i + 1))
            disp = self.disps.p_disp[i, i] - self.zmat.cartesians_final
            disp *= 30
            for j in range(len(disp)):
                normal_modes += "{:-20.10f}{:-20.10f}{:-20.10f}\n".format(
                    disp[j][0], disp[j][1], disp[j][2]
                )

        molden_template = """[Molden Format]\n[ATOMS] AU\n{geometry}[Molden Format]\n [FREQ]\n{frequencies} [FR-COORD]\n{fr_geom} [FR-NORM-COORD]\n{normal_modes}"""
        inout = {
            "geometry": geometry,
            "frequencies": frequencies,
            "fr_geom": fr_geom,
            "normal_modes": normal_modes,
        }
        molden_template = molden_template.format(**inout)

        # print(molden_template)
        with open("MOLDEN", "w+") as file:
            file.write(molden_template)
        pass
