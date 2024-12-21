import fileinput
import os
import re
import shutil
import numpy as np


class DirectoryTree(object):
    def __init__(
        self,
        prog_name,
        zmat,
        disps,
        insertion_index,
        p_disp,
        m_disp,
        options,
        indices,
        template,
        dir_name,
        deriv_level=0,
        anharm=False,
    ):
        self.prog_name = prog_name
        self.zmat = zmat
        self.insertion_index = insertion_index
        self.dir_name = dir_name
        self.disps = disps  # This should be the 'TransfDisp' object
        self.p_disp = p_disp
        self.m_disp = m_disp
        self.options = options
        self.template = template
        self.indices = indices
        self.deriv_level = deriv_level
        self.anharm = anharm

    def run(self):
        disp_dict = {}
        disp_dict["disp"] = []
        disp_dict["geom"] = []

        root = os.getcwd()

        prog_name = self.prog_name

        n_atoms = len(self.zmat.atom_list)

        if prog_name == "molpro" or prog_name == "psi4" or prog_name == "cfour":
            with open(self.template, "r") as file:
                data = file.readlines()
            if self.options.pert_off_diag:
                with open("template_od.dat") as file:
                    data_od = file.readlines()
        else:
            print("Specified program not supported: " + prog_name)
            raise RuntimeError

        init = False
        genbas = False
        ecp = False
        if os.path.exists(root + "/initden.dat"):
            init = True
        if os.path.exists(root + "/GENBAS"):
            genbas = True
        if os.path.exists(root + "/ECPDATA"):
            ecp = True
        data_buff = data.copy()
        if self.options.pert_off_diag:
            data_od_buff = data_od.copy()
        if os.path.exists(os.getcwd() + "/old" + self.dir_name):
            shutil.rmtree("old" + self.dir_name, ignore_errors=True)
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
            if self.anharm:
                for i in range(len(self.p_disp)):
                    p_data = self.make_input(
                        data,
                        self.p_disp[i],
                        str(n_atoms),
                        self.zmat.atom_list,
                        self.insertion_index,
                    )

                    os.mkdir(str(2 * i + 1))
                    os.chdir("./" + str(2 * i + 1))
                    with open(inp, "w") as file:
                        file.writelines(p_data)
                    data = data_buff.copy()
                    if init:
                        shutil.copy("../../initden.dat", ".")
                    if genbas:
                        shutil.copy("../../GENBAS", ".")
                    if ecp:
                        shutil.copy("../../ECPDATA", ".")
                    os.chdir("..")

                    m_data = self.make_input(
                        data,
                        self.m_disp[i],
                        str(n_atoms),
                        self.zmat.atom_list,
                        self.insertion_index,
                    )
                    os.mkdir(str(2 * i + 2))
                    os.chdir("./" + str(2 * i + 2))
                    with open(inp, "w") as file:
                        file.writelines(m_data)
                    data = data_buff.copy()
                    if init:
                        shutil.copy("../../initden.dat", ".")
                    if genbas:
                        shutil.copy("../../GENBAS", ".")
                    if ecp:
                        shutil.copy("../../ECPDATA", ".")
                    os.chdir("..")
                    # direc += 2
                # pass
            else:
                os.mkdir("1")
                os.chdir("./1")
                data = self.make_input(
                    data,
                    self.disps.disp_cart["ref"],
                    str(n_atoms),
                    self.zmat.atom_list,
                    self.insertion_index,
                )

                with open(inp, "w") as file:
                    file.writelines(data)
                data = data_buff.copy()
                if init:
                    shutil.copy("../../initden.dat", ".")
                if genbas:
                    shutil.copy("../../GENBAS", ".")
                if ecp:
                    shutil.copy("../../ECPDATA", ".")
                os.chdir("..")

                # Not including the reference input, this function generates the directories for the displacement
                # jobs and copies in the input file data. Following this, these jobs are ready to be submitted to the queue.

                p_disp = self.p_disp
                m_disp = self.m_disp
                # a = p_disp.shape[0]
                indices = self.indices
                direc = 2

                # This loop makes calls to the Make_Input function for storing the cartesian displacements as a variable,
                # upon which it is written to a standard input file after the create_directory function creates a directory
                # within the Disps directory.
                if self.options.pert_off_diag:
                    for index in indices:
                        i, j = index[0], index[1]
                        if i == j:
                            p_data = self.make_input(
                                data,
                                p_disp[i, j],
                                str(n_atoms),
                                self.zmat.atom_list,
                                self.insertion_index,
                            )
                            data = data_buff.copy()
                        else:
                            p_data = self.make_input(
                                data_od,
                                p_disp[i, j],
                                str(n_atoms),
                                self.zmat.atom_list,
                                self.insertion_index,
                            )
                            data_od = data_od_buff.copy()

                        os.mkdir(str(direc))
                        os.chdir("./" + str(direc))
                        with open(inp, "w") as file:
                            file.writelines(p_data)
                        if init:
                            shutil.copy("../../initden.dat", ".")
                        if genbas:
                            shutil.copy("../../GENBAS", ".")
                        if ecp:
                            shutil.copy("../../ECPDATA", ".")
                        os.chdir("..")
                        if i == j:
                            m_data = self.make_input(
                                data,
                                m_disp[i, j],
                                str(n_atoms),
                                self.zmat.atom_list,
                                self.insertion_index,
                            )
                            data = data_buff.copy()
                        else:
                            m_data = self.make_input(
                                data_od,
                                m_disp[i, j],
                                str(n_atoms),
                                self.zmat.atom_list,
                                self.insertion_index,
                            )
                            data_od = data_od_buff.copy()
                        os.mkdir(str(direc + 1))
                        os.chdir("./" + str(direc + 1))
                        with open(inp, "w") as file:
                            file.writelines(m_data)
                        if init:
                            shutil.copy("../../initden.dat", ".")
                        if genbas:
                            shutil.copy("../../GENBAS", ".")
                        if ecp:
                            shutil.copy("../../ECPDATA", ".")
                        os.chdir("..")
                        direc += 2
                else:
                    for index in indices:
                        i, j = index[0], index[1]
                        p_data = self.make_input(
                            data,
                            p_disp[i, j],
                            str(n_atoms),
                            self.zmat.atom_list,
                            self.insertion_index,
                        )

                        os.mkdir(str(direc))
                        os.chdir("./" + str(direc))
                        with open(inp, "w") as file:
                            file.writelines(p_data)
                        data = data_buff.copy()
                        if init:
                            shutil.copy("../../initden.dat", ".")
                        if genbas:
                            shutil.copy("../../GENBAS", ".")
                        if ecp:
                            shutil.copy("../../ECPDATA", ".")
                        os.chdir("..")

                        m_data = self.make_input(
                            data,
                            m_disp[i, j],
                            str(n_atoms),
                            self.zmat.atom_list,
                            self.insertion_index,
                        )
                        os.mkdir(str(direc + 1))
                        os.chdir("./" + str(direc + 1))
                        with open(inp, "w") as file:
                            file.writelines(m_data)
                        data = data_buff.copy()
                        if init:
                            shutil.copy("../../initden.dat", ".")
                        if genbas:
                            shutil.copy("../../GENBAS", ".")
                        if ecp:
                            shutil.copy("../../ECPDATA", ".")
                        os.chdir("..")
                        direc += 2
        elif self.deriv_level == 1:
            direc = 1
            for index in self.indices:
                p_data = self.make_input(
                    data,
                    self.p_disp[index],
                    str(n_atoms),
                    self.zmat.atom_list,
                    self.insertion_index,
                )

                os.mkdir(str(direc))
                os.chdir("./" + str(direc))
                with open(inp, "w") as file:
                    file.writelines(p_data)
                data = data_buff.copy()
                if init:
                    shutil.copy("../../initden.dat", ".")
                if genbas:
                    shutil.copy("../../GENBAS", ".")
                if ecp:
                    shutil.copy("../../ECPDATA", ".")
                os.chdir("..")

                m_data = self.make_input(
                    data,
                    self.m_disp[index],
                    str(n_atoms),
                    self.zmat.atom_list,
                    self.insertion_index,
                )
                os.mkdir(str(direc + 1))
                os.chdir("./" + str(direc + 1))
                with open(inp, "w") as file:
                    file.writelines(m_data)
                data = data_buff.copy()
                if init:
                    shutil.copy("../../initden.dat", ".")
                if genbas:
                    shutil.copy("../../GENBAS", ".")
                if ecp:
                    shutil.copy("../../ECPDATA", ".")
                os.chdir("..")

                direc += 2
        else:
            print(
                "Only energy and gradient derivatives are supported. Check your deriv_level keyword."
            )
            raise RuntimeError

    def make_input(self, data, dispp, n_at, at, index):
        space = " "
        if self.prog_name == "cfour":
            space = ""
        if index == -1:
            print(
                "The user needs to specify a different value for the \
                   cart_insert keyword."
            )
            raise RuntimeError
        else:
            for i in range(int(n_at)):
                data.insert(
                    index + i,
                    space
                    + at[i]
                    + "{:16.10f}".format(dispp[i][0])
                    + "{:16.10f}".format(dispp[i][1])
                    + "{:16.10f}".format(dispp[i][2])
                    + "\n",
                )

        return data
