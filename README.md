# ConcordantModes

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This program utilizes the Concordant Modes theoretical framework to compute highly accurate Hessians with linear scaling. The CMA protocol is detailed in literature (J. Am. Chem. Soc., 2022, 144, 51, 23271–23274.; J. Chem. Theory Comput., 2024, 20, 24, 10886–10898.).
As of right now, the user must be on a cluster with either a Sun Grid Engine or Slurm queueing system and have access to CCQC programs to use the built in computation submission procedure. A feature to allow for custom submit scripts to be utilized is in the works.
The following procedure may be used to enact the CMA-0A procedure using manual submission of computations for any electronic structure program on any machine. To use CMA-0A, the following files must be present in the working directory: templateA.dat, templateB.dat, main.py, and zmat. These files are discussed below, and sample versions are present in the examples directory. When the files are set, then follow this procedure.
1)	Set the "gen_disps_b" keyword to True and the "calc_b" keyword to False, then run the CMA program
2)	Compute the generated initial displacements using your quantum chemistry package of choice.
3)	Set the "gen_disps_b" and "calc_a" keywords to False and the "gen_disps_a" keyword to True, then run the CMA program again.
4)	Compute the generated displacements using your quantum chemistry package of choice.
5)	Set the "gen_disps_a" keyword to False and run the CMA program one more time.


## Installation

User must have at least:
Python version 3.10
Numpy version 1.13.3
qcelemental

Developers must have:
Pytest 7.2.0 (-c condaforge)
Pytest-xdist 3.0.2 (-c condaforge)

A simple way to ensure this program may be run is by creating a conda environment with the following command:

`conda create --name CMA python=3.10 sympy numpy scipy qcelemental`

If you need to install anaconda, consult the [official website](https://www.anaconda.com/products/individual)

Alternatively, after installing the source code, ensure `pip` is installed and run `pip install -e .` from `ConcordantModes/`. All
dependencies will be installed and `concordantmodes` will be made available as a python module.

## Quickstart Guide

A myriad of examples are provided in the "examples" directory.

4 files are necessary for CMA-0A: "zmat", "templateA.dat", "templateB.dat",  "main.py"

zmat: This file contains geometric information about the system of interest. There is a "ZMAT" block and a "cart" block.
The "ZMAT" block contains connectivity information for specifying the internal coordinates of the molecule by which the normal modes will be described. The user may consult the "examples" directory contained herein for more guidance on how to construct the different connectivities.

The "cart" block contains the cartesian structures for the system. If two structures are entered (separated by a --- on a new line), the first structure will be used to construct the starting normal modes from the provided force constants. These normal modes are then placed atop the second structure and the requisite displacements are generated to compute force constants for these modes and the subsequent frequencies (CMA-0B). If only one structure is entered, the same process will take place but all atop a single structure (CMA-0A). The default units are bohr.

templateA/B.dat: This is the template input file for an interfacing quantum chemistry code. The displaced geometries are inserted into this file and single point energies/gradients for each displaced geometry are computed. The energies/gradients are then reaped and a second/first order numerical derivative is performed to calculate force constants from these energies.

main.py: This file imports the Concordant Mode options and program, and then runs the program. A manual detailing each keyword is in the works. In the meantime consult the examples directory for templates of different procedures. To run the program, simply enter "python main.py" or "nohup python -u main.py &" if you would like to retain the output.

