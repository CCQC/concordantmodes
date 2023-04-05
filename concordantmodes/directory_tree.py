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
    ):
        # self.dispSym = dispSym
        self.prog_name = prog_name
        self.zmat = zmat
        self.insertion_index = insertion_index
        self.dir_name = dir_name
        self.disps = disps  # This should be the 'TransfDisp' object
        # nate
        self.p_disp = p_disp
        self.m_disp = m_disp
        self.options = options
        self.template = template
        self.indices = indices
        self.deriv_level = deriv_level

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
                # print(data)
        else:
            print("Specified program not supported: " + prog_name)
            raise RuntimeError

        init = False
        genbas = False
        if os.path.exists(root + "/initden.dat"):
            init = True
        if os.path.exists(root + "/GENBAS"):
            genbas = True
        data_buff = data.copy()
        # print(os.getcwd())
        if os.path.exists(os.getcwd() + "/old" + self.dir_name):
            shutil.rmtree("old" + self.dir_name, ignore_errors=True)
        # if os.path.exists(os.getcwd() + "/oldDisps"):
        # shutil.rmtree("oldDisps", ignore_errors=True)
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
            os.chdir("..")

            # Not including the reference input, this function generates the directories for the displacement
            # jobs and copies in the input file data. Following this, these jobs are ready to be submitted to the queue.

            p_disp = self.p_disp
            m_disp = self.m_disp
            a = p_disp.shape[0]
            indices = self.indices
            direc = 2

            # This loop makes calls to the Make_Input function for storing the cartesian displacements as a variable,
            # upon which it is written to a standard input file after the create_directory function creates a directory
            # within the Disps directory.

            # for index in indices:
            #    i,j = index[0], index[1]
            #    p_data = self.Make_Input(data,p_disp[i,j],str(n_atoms),self.zmat.atom_list,self.insertionIndex)

            #    self.create_directory(direc,p_data)
            #    m_data = self.Make_Input(data,m_disp[i,j],str(n_atoms),self.zmat.atom_list,self.insertionIndex)

            #    self.create_directory(direc+1,m_data)
            #    direc += 2

            for index in indices:
                i, j = index[0], index[1]
                # print(p_disp[i,j])
                # raise RuntimeError
                p_data = self.make_input(
                    data,
                    p_disp[i, j],
                    str(n_atoms),
                    self.zmat.atom_list,
                    self.insertion_index,
                )

                # create_directory(direc,p_data)
                os.mkdir(str(direc))
                os.chdir("./" + str(direc))
                with open(inp, "w") as file:
                    file.writelines(p_data)
                data = data_buff.copy()
                if init:
                    shutil.copy("../../initden.dat", ".")
                if genbas:
                    shutil.copy("../../GENBAS", ".")
                os.chdir("..")

                # Don't generate a displacement if full symmetry is present
                # if self.disps.disp_sym[i] and self.disps.disp_sym[j]:
                # direc -= 1
                # else:
                # m_data = self.make_input(
                # data,
                # m_disp[i, j],
                # str(n_atoms),
                # self.zmat.atom_list,
                # self.insertion_index,
                # )
                # os.mkdir(str(direc + 1))
                # os.chdir("./" + str(direc + 1))
                # with open(inp, "w") as file:
                # file.writelines(m_data)
                # data = data_buff.copy()
                # if init:
                # shutil.copy("../../initden.dat", ".")
                # if genbas:
                # shutil.copy("../../GENBAS", ".")
                # os.chdir("..")

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
                os.chdir("..")
                # create_directory(direc+1,m_data)
                # data = data_buff.copy()
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

                # create_directory(direc,p_data)
                os.mkdir(str(direc))
                os.chdir("./" + str(direc))
                with open(inp, "w") as file:
                    file.writelines(p_data)
                data = data_buff.copy()
                if init:
                    shutil.copy("../../initden.dat", ".")
                if genbas:
                    shutil.copy("../../GENBAS", ".")
                os.chdir("..")

                # if self.disps.disp_sym[i] and self.disps.disp_sym[j]:
                # direc -= 1
                # else:
                # m_data = self.make_input(
                # data,
                # self.m_disp[index],
                # str(n_atoms),
                # self.zmat.atom_list,
                # self.insertion_index,
                # )
                # os.mkdir(str(direc + 1))
                # os.chdir("./" + str(direc + 1))
                # with open(inp, "w") as file:
                # file.writelines(m_data)
                # data = data_buff.copy()
                # if init:
                # shutil.copy("../../initden.dat", ".")
                # if genbas:
                # shutil.copy("../../GENBAS", ".")
                # os.chdir("..")
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
                os.chdir("..")

                # create_directory(direc+1,m_data)
                # data = data_buff.copy()
                direc += 2
            # raise RuntimeError
        else:
            print(
                "Only energy and gradient derivatives are supported. Check your deriv_level keyword."
            )
            raise RuntimeError
