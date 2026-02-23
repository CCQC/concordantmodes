import fileinput
import os
import re
import shutil
import copy
import numpy as np


class DirectoryTree(object):
    def __init__(
        self,
        prog_name,
        zmat,
        ref_geom,
        cma_level,
        p_disp,
        m_disp,
        options,
        indices,
        symm_obj,
        template,
        dir_name,
        deriv_level=0,
    ):
        self.prog_name = prog_name
        self.zmat = zmat
        self.ref_geom = ref_geom
        self.cma_level = cma_level
        self.p_disp = p_disp
        self.m_disp = m_disp
        self.options = options
        self.indices = indices
        self.symm_obj = symm_obj
        self.template = template
        self.dir_name = dir_name
        self.deriv_level = deriv_level
        if cma_level == "A":
            self.insertion_index = self.options.cart_insert_a
        elif cma_level == "B":
            self.insertion_index = self.options.cart_insert_b
        elif cma_level == "C":
            self.insertion_index = self.options.cart_insert_c
        else:
            print("Please specify A, B, or C for your cma_level")
            raise RuntimeError

    def run(self):

        root = os.getcwd()

        prog_name = self.prog_name

        n_atoms = len(self.zmat.atom_list)

        prog_list = ["molpro", "psi4", "cfour", "orca"]

        if prog_name in prog_list:
            with open(self.template, "r") as file:
                data = file.readlines()
        else:
            print("Specified program not supported: " + prog_name)
            raise RuntimeError

        self.init = False
        self.genbas = False
        self.ecp = False
        self.sub = False
        print("Giraffe")
        print(root)
        if os.path.exists(root + "/initden.dat"):
            self.init = True
        if os.path.exists(root + "/GENBAS"):
            self.genbas = True
        if os.path.exists(root + "/ECPDATA"):
            self.ecp = True
        if os.path.exists(root + "/sub_script.sh"):
            print("We made it here.")
            self.sub = True
        # raise RuntimeError

        data_buff = data.copy()
        if os.path.exists(os.getcwd() + "/old" + self.dir_name):
            shutil.rmtree("old" + self.dir_name, ignore_errors=True)
        if os.path.exists(os.getcwd() + "/" + self.dir_name):
            shutil.copytree(
                os.getcwd() + "/" + self.dir_name, os.getcwd() + "/old" + self.dir_name
            )
            shutil.rmtree(os.getcwd() + "/" + self.dir_name)
        inp = ""
        if self.prog_name == "cfour":
            inp = "ZMAT"
        else:
            inp = "input.dat"
        if os.path.exists(os.getcwd() + "/" + self.dir_name):
            shutil.move(self.dir_name, "old" + self.dir_name)
        os.mkdir(self.dir_name)
        os.chdir("./" + self.dir_name)
        if not self.deriv_level:
            self.make_input(
                copy.deepcopy(data),
                self.ref_geom,
                n_atoms,
                self.zmat.atom_list,
                self.insertion_index,
                inp,
                "1",
            )

            # Not including the reference input, this function generates the directories for the displacement
            # jobs and copies in the input file data. Following this, these jobs are ready to be submitted to the queue.

            p_disp = self.p_disp
            m_disp = self.m_disp
            indices = self.indices
            direc = 2

            if self.symm_obj.symtext is not None and self.options.exploit_pm_symm:
                if self.options.only_TSIR:
                    indices = self.symm_obj.indices_by_irrep[0]
                else:
                    indices = (
                        deepcopy.copy(self.symm_obj.indices_by_irrep)
                        .flatten()
                        .reshape((-1, 2))
                    )

            Sum = 1
            h = 0
            for index in indices:
                i, j = index[0], index[1]
                self.make_input(
                    copy.deepcopy(data),
                    p_disp[i, j],
                    n_atoms,
                    self.zmat.atom_list,
                    self.insertion_index,
                    inp,
                    direc,
                )
                if h != 0:
                    direc += 1
                else:
                    self.make_input(
                        copy.deepcopy(data),
                        m_disp[i, j],
                        n_atoms,
                        self.zmat.atom_list,
                        self.insertion_index,
                        inp,
                        direc + 1,
                    )

                    direc += 2
                if (
                    self.symm_obj.symtext is not None
                    and self.options.exploit_pm_symm
                    and not self.options.only_TSIR
                    and Sum > len(self.symm_obj.indices_by_irrep[0])
                ):
                    h += 1
                Sum += 1

        elif self.deriv_level == 1:
            direc = 1
            for index in self.indices:
                self.make_input(
                    copy.deepcopy(data),
                    self.p_disp[index[0]],
                    n_atoms,
                    self.zmat.atom_list,
                    self.insertion_index,
                    inp,
                    direc,
                )

                self.make_input(
                    copy.deepcopy(data),
                    self.m_disp[index[0]],
                    n_atoms,
                    self.zmat.atom_list,
                    self.insertion_index,
                    inp,
                    direc + 1,
                )

                direc += 2
        else:
            print(
                "Only energy and gradient derivatives are supported. Check your deriv_level keyword."
            )
            raise RuntimeError

    def make_input(self, data, dispp, n_at, at, index, inp, direc, copy_files=False):
        os.mkdir(str(direc))
        os.chdir("./" + str(direc))
        data_buff = copy.deepcopy(data)
        space = " "
        if self.prog_name == "cfour":
            space = ""
        if index == -1:
            print(
                "The user needs to specify a value for the \
                   cart_insert_a, cart_insert_b, or cart_insert_c keyword."
            )
            raise RuntimeError
        else:
            for i in range(n_at):
                data.insert(
                    index + i,
                    space
                    + at[i]
                    + "{:16.10f}".format(dispp[i][0])
                    + "{:16.10f}".format(dispp[i][1])
                    + "{:16.10f}".format(dispp[i][2])
                    + "\n",
                )

            with open(inp, "w") as file:
                file.writelines(data)
            data_final = copy.deepcopy(data)
            data = copy.deepcopy(data_buff)
            if self.options.cluster.lower() == "custom":
                copy_files = True
            if copy_files:
                if self.init:
                    shutil.copy("../../initden.dat", ".")
                if self.genbas:
                    shutil.copy("../../GENBAS", ".")
                if self.ecp:
                    shutil.copy("../../ECPDATA", ".")
                if self.sub:
                    shutil.copy("../../sub_script.sh", ".")
                    print("we also made it here")
            os.chdir("..")

        return data_final
