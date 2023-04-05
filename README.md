# ConcordantModes

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This program facilitates the computation of molecular force constants at a desired level of theory in the basis of normal modes from a lower level of theory.
The higher and lower levels of theory are denoted as levels A and B, respectively.
Two proecures for approximate hessians currently exist entitled CMA-0A and CMA-0B.
CMA-0A computes an initial level B Hessian atop the level A stationary point to obtain the normal mode basis for force constant computation.
CMA-0B reads in level B cartesian force constants computed atop the corresponding level B structure to obtain the normal mode basis for force constant computation.
In both cases, only the diagonal force constants are computed to obtain the final, approximate frequencies.

As of right now, the user must be on a cluster with a Sun Grid Engine queueing system to use the built in computation submission procedure.

The following procedure may be used to enact the CMA-0A/B procedure using manual submission of computations for any electronic structure program on any machine.
CMA-0A:
To use CMA-0A, the following files must be present in the working directory: template.dat, templateInit.dat, main.py, and zmat.
These files are discussed below. When the initial files are set, then follow this procedure.
1) Set the "gen_disps_init" keyword to True and the "calc_init" keyword to False, then run the CMA program
2) Compute the generated initial displacements using your quantum chemistry package of choice.
3) Set the "gen_disps_init" and "calc" keywords to False and the "gen_disps" keyword to True, then run the CMA program again.
4) Compute the generated displacements using your quantum chemistry package of choice.
5) Set the "gen_disps" keyword to False and run the CMA program one more time.
An example is available at the following directory concordantmodes/Example/1_H2O/psi4/CMA_0A/manual_submission
CMA-0B:
To use CMA-0B, the following files must be present in the working directory: template.dat, fc.dat, main.py, and zmat. These files are discussed below. When the initial files are set, then follow this procedure.
1) Set the "gen_disps" keyword to True and the "calc" keyword to False, then run the CMA program.
2) Compute the generated displacements using your quantum chemistry package of choice.
3) Set the "gen_disps" keyword to False and run the CMA program one more time.
An example is available at the following directory concordantmodes/Example/1_H2O/psi4/CMA_0B/Redundants/manual_submission

## Installation

User must have at least:
Python version 3.7
Numpy version 1.13.3
qcelemental

Developers must have:
Pytest 7.2.0 (-c condaforge)
Pytest-xdist 3.0.2 (-c condaforge)

A simple way to ensure this program may be run is by creating a conda environment with the following command:

`conda create --name CMA python=3.9 sympy numpy scipy qcelemental`

If you need to install anaconda, consult the [official website](https://www.anaconda.com/products/individual)

Alternatively, after installing the source code, ensure `pip` is installed and run `pip install -e .` from `ConcordantModes/`. All
dependencies will be installed and `concordantmodes` will be made available as a python module.

## Quickstart Guide

A myriad of examples are is provided in the "Example" directory.

4 files are necessary: "zmat", "template.dat", "main.py", "fc.dat"

zmat:
This file contains geometric information about the system of interest. There is a "ZMAT" block and a "cart" block.

The "ZMAT" block contains connectivity information for specifying the internal coordinates of the molecule by which the normal modes will be described.
The user may consult the "Example" directory contained herein for more guidance on how to construct the different connectivities.

The "cart" block contains the cartesian structures for the system. If two structures are entered (separated by a --- on a new line), the first structure will be used to construct the starting normal modes from the provided force constants. These normal modes are then placed atop the second structure and the requisite displacements are generated to compute force constants for these modes and the subsequent frequencies (CMA-0B). If only one structure is entered, the same process will take place but all atop a single structure (CMA-0A). The default units are bohr.

template.dat:
This is the template input file for an interfacing quantum chemistry code. The displaced geometries are inserted to this file and single point energies for each displaced geometry are computed. The energies are then reaped and a second order numerical derivative is performed to calculate force constants from these energies.

templateInit.dat:
This file is only used in the CMA-0A procedure. It serves the same purpose as the template.dat, but it is instead used in the computation of the initial Hessian.

An *experimental* procedure exists to compute the initial Hessian via findif of gradients. User beware as this code is, again, experimantal.

main.py:
This file imports the Concordant Mode options and program, and then runs the program. 
A manual detailing each keyword is in the works.
In the meantime consult the Example directory for templates of different procedures.
To run the program, simply enter "python main.py" or "nohup python -u main.py &" if you would like to retain the output.

fc.dat:
This file is only used in the CMA-0B procedure and it contains the cartesian force constants of the first structure if CMA-0B is performed.
