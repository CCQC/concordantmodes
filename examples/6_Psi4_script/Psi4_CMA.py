import psi4
import concordantmodes
import time
import numpy as np
from numpy import linalg as LA
import os
import re
import sys
import shutil
import subprocess
from concordantmodes.algorithm import Algorithm
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.force_constant import ForceConstant
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.g_read import GrRead
from concordantmodes import masses
from concordantmodes.molden_writer import MoldenWriter
from concordantmodes.options import Options
from concordantmodes.s_vectors import SVectors
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.zmat import Zmat

t1 = time.time()

# First things first. Let's get a Psi4 hessian running.
print(psi4.__version__)

psi4.set_output_file("output.dat", False)
# psi4.core.be_quiet()

# Set psi4 options here
psi4.set_memory("10 GB")
psi4.set_options({"freeze_core": True})
psi4.set_options({"normal_modes_write": True})
psi4.set_options({"e_convergence": 10})
psi4.set_options({"d_convergence": 10})
psi4.set_options({"r_convergence": 8})
psi4.set_options({"cc_type": "conv"})
# psi4.set_options({'scf_type': 'pk'})

mol = psi4.geometry(
    """
     C         0.00000000     0.00000000     1.14423616
     H         0.00000000    -1.76987484     2.24620597
     H         0.00000000     1.76987484     2.24620597
     O        -0.00000000     0.00000000    -1.14151276
        """
)

# psi4 units set as bohr, if not then CMA units will need to be set as angstrom
mol.set_units(psi4.core.GeometryUnits.Bohr)
mol.fix_orientation(True)
mol.fix_com(True)

# Set the CMA variables here:
level_a = "ccsd(t)/cc-pvtz"
level_b = "mp2/cc-pvtz"

options_kwargs = {
    "man_proj": False,
    "coords": "Delocalized",
    "reduced_disp": True,
    "cart_fc_b": True,
    "covalent_radii": True,
    "second_order": True,
    # "reduced_disp"           : 0.002,
    # "molsym_symmetry" : True,
    # "autosalcs" : True,
}

options = Options(**options_kwargs)
print("Disp:")
print(options.disp)

# Change this if man_proj=True
proj = None
# modify this for custom symmetry
sym_sort = []


# string value to access "state" of CMA program. Are we in level A, B, C, etc...
cma_level = "B"

# Read in Psi4 Molecule information into our zmat object.
zmat_obj = Zmat(options)
zmat_obj.divider = False

zmat_obj.cartesians_b = mol.geometry().to_array()
zmat_obj.cartesians_a = mol.geometry().to_array()

atom_list = []
for i in range(len(zmat_obj.cartesians_a)):
    atom_list.append(mol.fsymbol(i))
zmat_obj.atom_list = atom_list

zmat_obj.masses = [masses.get_mass(label) for label in zmat_obj.atom_list]
for i in range(len(zmat_obj.masses)):
    zmat_obj.masses[i] = zmat_obj.masses[i] / zmat_obj.amu_elMass
zmat_obj.mass_weight = np.diag(np.array(zmat_obj.masses).repeat(3))

zmat_obj.zmat_process("")
zmat_obj.zmat_calc()
zmat_obj.zmat_compile()
zmat_obj.zmat_print()


# Do we want to use molsym_symmetry or "manual" symmetry via sym_sort?
symm_obj = Symmetry(zmat_obj, options, proj)
if options.molsym_symmetry:
    symm_obj.run()
else:
    """
    We won't run the symmetry code, but we'll create a dummy object to be passed as an argument.
    #TODO: This is a hacky way to do this, but it's a quick fix for now. Maybe reincorporate symmetry as a s_vector obj?
    """
    symm_obj.dummy_obj()
    symm_obj.symtext = None
    # check if sym_sort object was passed in. If so, intialize sym_sort objects
    if len(sym_sort) > 1:
        symm_obj.create_flat_sym_sort(sym_sort)


# Compute the initial s-vectors
s_vec = SVectors(zmat_obj, options, zmat_obj.variable_dictionary_b)
s_vec.run(
    zmat_obj.cartesians_b,
    True,
    second_order=options.second_order,
)

if options.molsym_symmetry:
    symm_obj.make_proj(s_vec)
    s_vec.proj = copy.deepcopy(symm_obj.salc_proj)

# Define the Total Energy Distribution object
TED_obj = TED(s_vec.proj, zmat_obj, options)

# Print out the percentage composition of the projected coordinates
if options.coords != "ZMAT":
    TED_obj.run(np.eye(len(TED_obj.proj.T)), np.zeros(len(TED_obj.proj.T)))

# Compute G-Matrix
g_mat = GMatrix(zmat_obj, s_vec, options)
g_mat.run()

psi4.energy(level_b)

np.set_printoptions(precision=4, linewidth=240)

grad_e, grad_wfn = psi4.gradient(level_b, molecule=mol, return_wfn=True)
# print(grad_wfn.gradient().to_array().flatten())
cart_grad = grad_wfn.gradient().to_array().flatten()

if options.second_order:
    g_read_obj = GrRead("")
    g_read_obj.cart_grad = cart_grad

freq_e, freq_wfn = psi4.frequency(level_b, molecule=mol, return_wfn=True)
print(freq_wfn.hessian().to_array())
cart_hess = freq_wfn.hessian().to_array()

# raise RuntimeError

# This might not be necessary, just holding onto it for now
options.deriv_level_b = 0
options.deriv_level_a = 0

f_conv_obj = FcConv(
    cart_hess,
    # f_read_obj.fc_mat,
    s_vec,
    zmat_obj,
    "internal",
    False,
    TED_obj,
    options,
)
if options.second_order:
    f_conv_obj.run(grad=g_read_obj.cart_grad)
else:
    f_conv_obj.run()

F = f_conv_obj.F

g_mat.G = np.dot(TED_obj.proj.T, np.dot(g_mat.G, TED_obj.proj))

options.second_order = False
# Numerical trimming
F[np.abs(F) < options.tol] = 0
g_mat.G[np.abs(g_mat.G) < options.tol] = 0

# Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
print("Initial Frequencies:")
b_GF = GFMethod(
    g_mat.G.copy(),
    F.copy(),
    zmat_obj,
    TED_obj,
    options,
    symm_obj.symtext,
    cma="init",
)
b_GF.run()

# raise RuntimeError

ted_b = b_GF.ted.TED
if len(sym_sort):
    irreps_b, flat_sym_freqs = symm_obj.mode_symmetry_sort(
        b_GF.ted.TED, sym_sort, b_GF.freq
    )
    ref_b = np.array(flat_sym_freqs)
    #### this block could probably be moved inside the symmetry.py module?
    flat_sym_modes_b = [x for xs in irreps_b for x in xs]
    print(flat_sym_modes_b)
    del_list = []
    for i in range(len(irreps_b)):
        if len(irreps_b[i]) == 1:
            del_list.append(irreps_b[i][0])
    flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b), del_list2)
    ted_b = ted_b.T
    ted_b = ted_b[flat_sym_modes_b]
    ted_b = ted_b.T
    #### end of block that could probably be moved inside the symmetry.py module?


# Now for the TED check.
G = np.dot(np.dot(LA.inv(b_GF.L), g_mat.G), LA.inv(b_GF.L).T)
G[np.abs(G) < options.tol] = 0
F = np.dot(np.dot(b_GF.L.T, F), b_GF.L)
F[np.abs(F) < options.tol] = 0

print("TED Frequencies:")
TED_GF = GFMethod(
    G,
    F,
    zmat_obj,
    TED_obj,
    options,
    symm_obj.symtext,
    cma=False,
)
TED_GF.run()

initial_fc = TED_GF.eig_v
eigs = len(TED_GF.L)


cma_level = "A"
num_deg_free = s_vec.proj.shape[1]
if options.molsym_symmetry:
    algo = Algorithm(num_deg_free, cma_level, options, symm_obj.proj_irreps)
else:
    algo = Algorithm(num_deg_free, cma_level, options, None)
# Only generate CMA-0 indices for now
algo.run()
if options.molsym_symmetry:
    symm_obj.indices_by_irrep = algo.indices_by_irrep

transf_disp = TransfDisp(
    s_vec, zmat_obj, b_GF.L, True, TED_obj, options, algo.indices, symm_obj=symm_obj
)
transf_disp.run(fc=F)
p_disp = transf_disp.p_disp
m_disp = transf_disp.m_disp

print("Computing Ref Energy:")
sub_t1 = time.time()
ref_en, first_wfn = psi4.energy(level_a, molecule=mol, return_wfn=True)
print(ref_en)
psi4.core.clean()
sub_t2 = time.time()
print("This energy took " + str(sub_t2 - sub_t1) + " seconds to run.")

mol_disp = mol.clone()

p_en_array = []
print("Positive disps ({length:d} total):".format(length=len(p_disp)))
for i in range(len(p_disp)):
    sub_t1 = time.time()
    disp = p_disp[i, i]
    disp_shape = disp.shape
    a, b = disp_shape[0], disp_shape[1]
    psi_disp = psi4.core.Matrix(a, b).from_array(disp)  # initialize psi4.Matrix
    mol_disp.set_geometry(psi_disp)
    e = psi4.energy(level_a, molecule=mol_disp)
    p_en_array.append(e)
    psi4.core.clean()
    print(i)
    print(e)
    sub_t2 = time.time()
    print("This energy took " + str(sub_t2 - sub_t1) + " seconds to run.")

p_en_array = np.diag(p_en_array)
print(p_en_array)

m_en_array = []
print("Negative disps ({length:d} total):".format(length=len(p_disp)))
for i in range(len(m_disp)):
    sub_t1 = time.time()
    disp = m_disp[i, i]
    disp_shape = disp.shape
    a, b = disp_shape[0], disp_shape[1]
    psi_disp = psi4.core.Matrix(a, b).from_array(disp)  # initialize psi4.Matrix
    mol_disp.set_geometry(psi_disp)
    e = psi4.energy(level_a, molecule=mol_disp)
    m_en_array.append(e)
    psi4.core.clean()
    print(i)
    print(e)
    sub_t2 = time.time()
    print("This energy took " + str(sub_t2 - sub_t1) + " seconds to run.")

m_en_array = np.diag(m_en_array)
print(m_en_array)

f_c = ForceConstant(transf_disp, p_en_array, m_en_array, ref_en, options, algo.indices)
f_c.run()
print("Computed Force Constants:")
print(f_c.FC)
F = f_c.FC

g_mat = GMatrix(zmat_obj, s_vec, options)
g_mat.run()
g_mat.G = np.dot(TED_obj.proj.T, np.dot(g_mat.G, TED_obj.proj))

g_mat.G = np.dot(np.dot(transf_disp.eig_inv, g_mat.G), transf_disp.eig_inv.T)
g_mat.G[np.abs(g_mat.G) < options.tol] = 0
G = g_mat.G

# Final GF Matrix run
print("Final Harmonic Frequencies:")
a_GF = GFMethod(
    G,
    F,
    zmat_obj,
    TED_obj,
    options,
    symm_obj.symtext,
)
a_GF.run()


print("////////////////////////////////////////////")
print("//{:^40s}//".format(" Final TED"))
print("////////////////////////////////////////////")
TED_obj.run(np.dot(b_GF.L, a_GF.L), a_GF.freq)

# This code prints out the frequencies in order of energy as well
# as the ZPVE in several different units.
print(
    "Final Harmonic ZPVE in: "
    + "{:6.2f}".format(np.sum(a_GF.freq) / 2)
    + " (cm^-1) "
    + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 349.7550881133)
    + " (kcal mol^-1) "
    + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 219474.6313708)
    + " (hartrees) "
)

F = np.dot(np.dot(transf_disp.eig_inv.T, F), transf_disp.eig_inv)
if options.coords != "ZMAT":
    F = np.dot(TED_obj.proj, np.dot(F, TED_obj.proj.T))
options.second_order = False
cart_conv = FcConv(
    F,
    s_vec,
    zmat_obj,
    "cartesian",
    True,
    TED_obj,
    options,
)
cart_conv.run()

molden = MoldenWriter(zmat_obj, transf_disp, a_GF.freq)
molden.run()

t2 = time.time()
print("This program took " + str(t2 - t1) + " seconds to run.")

# Bonus, run the psi4 TZ frequencies to see how well we did
t1 = time.time()
psi4_e, psi4_wfn = psi4.frequencies("ccsd(t)/cc-pvtz", molecule=mol, return_wfn=True)
t2 = time.time()
print("Psi4 freq - CMA freq:")
print(psi4_wfn.frequencies().to_array() - a_GF.freq)
print("Psi4 took " + str(t2 - t1) + " seconds to run.")
