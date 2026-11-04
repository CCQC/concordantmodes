Theory
======

This page explains what CMA computes and why it works, and how the program
carries it out. The method is described in [CMA2022]_, [CMA2024]_ and
[CMA2026]_ (see :doc:`../citing`); the notation here follows the code.

.. contents:: On this page
   :local:
   :depth: 1

The cost of a harmonic frequency calculation
--------------------------------------------

Harmonic frequencies come from the matrix of second derivatives of the energy
(the force constants, or Hessian). When analytic second derivatives are not
available, as is common for the most accurate methods such as CCSD(T), the
Hessian is built by finite differences of energies. For :math:`n` vibrational
coordinates this needs every diagonal and off-diagonal element, which with
central differences costs

.. math::

   N_\text{full} = 1 + 2n + 2\binom{n}{2} = 1 + n(n+1)

single-point energies (the reference, :math:`\pm` displacements of each
coordinate, and :math:`\pm` double displacements of each pair). The number
grows quadratically with molecular size, on top of the steep scaling of the
method itself (:math:`N^7` for CCSD(T)).

CMA reduces this to

.. math::

   N_\text{CMA-0} = 1 + 2n

high-level energies, which grows only linearly with the number of modes.

.. list-table:: Number of level-A single points (energies)
   :header-rows: 1

   * - Molecule
     - :math:`n`
     - Full Hessian
     - CMA-0A
   * - water
     - 3
     - 13
     - 7
   * - methanol
     - 12
     - 157
     - 25

The GF method
-------------

Vibrations are analysed in internal coordinates :math:`\mathbf{S}` (bond
stretches, angle bends, torsions, ...). To first order they are related to
the Cartesian displacements :math:`\mathbf{x}` by the Wilson
:math:`\mathbf{B}` matrix,

.. math::

   \delta\mathbf{S} = \mathbf{B}\,\delta\mathbf{x},

and the kinetic energy in these coordinates is described by

.. math::
   :label: gmatrix

   \mathbf{G} = \mathbf{B}\,\mathbf{M}^{-1}\mathbf{B}^\mathsf{T},

where :math:`\mathbf{M}` holds the atomic masses. With the force-constant
matrix :math:`\mathbf{F}` in the same coordinates, the harmonic normal modes
:math:`\mathbf{L}` and frequencies :math:`\nu_k` follow from the eigenvalue
problem

.. math::
   :label: gf

   \mathbf{G}\mathbf{F}\,\mathbf{L} = \mathbf{L}\,\boldsymbol{\Lambda},
   \qquad \lambda_k = 4\pi^2 c^2 \nu_k^2 .

The columns of :math:`\mathbf{L}` define the normal coordinates
:math:`\mathbf{Q}` through :math:`\mathbf{S} = \mathbf{L}\mathbf{Q}`. In the
code, ``s_vectors.py`` builds :math:`\mathbf{B}`, ``g_matrix.py`` evaluates
:eq:`gmatrix`, and ``gf_method.py`` solves :eq:`gf`.

Concordant normal coordinates
-----------------------------

CMA uses two levels of theory:

* **level B**, an inexpensive method (for example MP2) whose full Hessian
  :math:`\mathbf{F}_B` is computed;
* **level A**, the target method (for example CCSD(T)).

Solving :eq:`gf` with :math:`\mathbf{F}_B` gives level-B normal modes
:math:`\mathbf{L}_B`. The level-A force constants expressed in these
coordinates are

.. math::
   :label: fcma

   \mathbf{F}_\text{CMA}(A) = \mathbf{L}_B^\mathsf{T}\,\mathbf{F}_A\,\mathbf{L}_B,
   \qquad
   \mathbf{F}_A = (\mathbf{L}_B^{-1})^\mathsf{T}\,\mathbf{F}_\text{CMA}(A)\,\mathbf{L}_B^{-1}.

If levels A and B gave identical normal modes, :math:`\mathbf{F}_\text{CMA}(A)`
would be exactly diagonal. Normal-mode *shapes* are much less sensitive to the
level of theory than frequencies are, so for a reasonable level B the two sets
of modes are nearly the same ("concordant") and the off-diagonal elements of
:math:`\mathbf{F}_\text{CMA}(A)` are small. Neglecting them, only the
:math:`n` diagonal elements are needed, each from two displaced level-A
energies.

The final frequencies are obtained by solving :eq:`gf` once more in the
concordant coordinates, with :math:`\mathbf{F}_\text{CMA}(A)` and the
:math:`\mathbf{G}` matrix transformed into the same basis,

.. math::

   \mathbf{G}_\text{CMA} = \mathbf{L}_B^{-1}\,\mathbf{G}\,(\mathbf{L}_B^{-1})^\mathsf{T},

evaluated at the level-A reference geometry. In practice the program works
with the rows of :math:`\mathbf{L}_B^{-1}` normalized to unit length, so that a
displacement along a concordant coordinate has the same units (bohr or radian)
as one along an internal coordinate.

The CMA variants
----------------

CMA-0A
    Level-B Hessian at the level-A reference geometry; diagonal level-A force
    constants only. This is what the program does when the ``zmat`` file has
    one geometry. For the benchmark molecules of [CMA2024]_, CMA-0A reduced the
    level-A cost by a factor of about 7–10.

CMA-0B
    The level-B Hessian (and normal modes) are taken at the level-B
    equilibrium geometry and used at the level-A reference geometry. This lets
    an existing level-B Hessian be reused (``fc_cart_b.dat``; see
    :doc:`../tutorials/cma0b`), and is selected by giving two geometries in
    the ``zmat`` file. CMA-0A was found to be more accurate than CMA-0B,
    which the authors attribute to the importance of using the level-A
    reference geometry [CMA2024]_.

CMA-1 and CMA-2
    Some off-diagonal elements of :math:`\mathbf{F}_\text{CMA}(A)` are
    computed as well. In CMA-2 they are chosen in advance from a cheap
    third level of theory C (for example Hartree–Fock), whose force constants
    in the level-B concordant coordinates give the coupling diagnostic

    .. math::

       \xi_{ij} = \frac{\left|F_{\text{CMA},ij}(C)\right|}
                       {\sqrt{\left|F_{\text{CMA},ii}(C)\,F_{\text{CMA},jj}(C)\right|}} .

    Pairs with :math:`\xi_{ij}` above a threshold (``xi_tol``; 0.02 is the
    recommended general value) are computed at level A. With MP2/cc-pVTZ as
    level B and :math:`\xi = 0.02`, CMA-2A reproduced CCSD(T)/cc-pVTZ
    frequencies with an average maximum error per molecule of 0.17 cm\
    :sup:`-1` for a 33 % increase in cost over CMA-0A [CMA2024]_. For
    noncovalent complexes CMA-0A gave a mean absolute error of 0.23 cm\
    :sup:`-1`, and CMA-2A 0.08 cm\ :sup:`-1` while adding only 3 % of the
    neglected couplings [CMA2026]_.

    .. note::

       The CMA-2 code path (``off_diag = 2``) is not finished in this version
       of the program and stops with an explanatory error.

Choosing the levels of theory
-----------------------------

The level-B normal modes must be qualitatively right; their frequencies need
not be. [CMA2024]_ recommends MP2/cc-pVTZ as level B for CCSD(T)/cc-pVTZ
targets, and [CMA2026]_ found MP2 with an augmented triple-zeta basis
(h-aug-cc-pVTZ) suitable for weakly bound complexes. Use the same basis
family for both levels where possible, and make sure both reference
geometries are well converged.

How the program computes it
---------------------------

Internal coordinates
~~~~~~~~~~~~~~~~~~~~

The ``coords`` option selects the internal coordinates
(:doc:`../user_guide/zmat`). With ``Delocalized`` (or any redundant set), CMA
removes redundancies with a singular value decomposition of
:math:`\mathbf{B}`,

.. math::

   \mathbf{B} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\mathsf{T},

keeping the columns of :math:`\mathbf{U}` whose singular values exceed
``red_tol``. These columns form a projection :math:`\mathbf{P}`, and the
non-redundant coordinates are :math:`\mathbf{P}^\mathsf{T}\mathbf{S}`. A
projection can also be supplied directly (``man_proj``), for example
symmetry-adapted combinations (:doc:`../tutorials/symmetry`).

Displaced geometries
~~~~~~~~~~~~~~~~~~~~

Each displacement is defined in internal (level B) or concordant (level A)
coordinates with size ``disp_b`` or ``disp_a``. Because the internal
coordinates are non-linear in the Cartesians, the Cartesian geometry is found
iteratively: starting from the reference,

.. math::

   \delta\mathbf{x} = \mathbf{M}^{-1}\mathbf{B}^\mathsf{T}\mathbf{G}^{-1}\,\delta\mathbf{S}

is applied and repeated with the remaining error in :math:`\mathbf{S}` until
it is below ``transf_tol`` (at most ``transf_max_it`` steps).

Finite differences
~~~~~~~~~~~~~~~~~~

With energies :math:`E_0` (reference), :math:`E_{\pm i}` (coordinate
:math:`i` displaced by :math:`\pm\Delta_i`) and :math:`E_{\pm i \pm j}`
(coordinates :math:`i` and :math:`j` displaced together), ``force_constant.py``
uses central differences:

.. math::

   F_{ii} &= \frac{E_{+i} - 2E_0 + E_{-i}}{\Delta_i^2}, \\
   F_{ij} &= \frac{E_{+i+j} - E_{+i} - E_{+j} + 2E_0 - E_{-i} - E_{-j} + E_{-i-j}}{2\Delta_i\Delta_j}, \\
   g_i &= \frac{E_{+i} - E_{-i}}{2\Delta_i}.

With ``deriv_level = 1`` the force constants come instead from central
differences of analytic gradients, which needs :math:`1 + 2n` gradient
calculations for a full Hessian; Cartesian gradients are converted to the
internal coordinates with the pseudoinverse of :math:`\mathbf{B}`.

With ``reduced_disp`` the level-A step sizes are adapted to each mode's
curvature, :math:`\Delta_i = s / |F_{ii}|^{1/4}` with :math:`s` =
``reduced_disp_size``, using the level-B force constants.

Non-stationary geometries
~~~~~~~~~~~~~~~~~~~~~~~~~

At a geometry where the gradient :math:`\mathbf{g}` does not vanish, converting
force constants between internal and Cartesian coordinates needs the second
derivatives of the internal coordinates, :math:`B^{(2)}_{k,ab} =
\partial^2 S_k / \partial x_a \partial x_b`:

.. math::

   F^{(x)}_{ab} = \sum_{kl} B_{ka} F_{kl} B_{lb} + \sum_k g_k\, B^{(2)}_{k,ab}.

``second_order = true`` includes this term. CMA computes
:math:`\mathbf{B}^{(2)}` by finite differences of :math:`\mathbf{B}` and saves
the gradients alongside the force constants (``fc_int_a.grad``) so that
results read back from files are identical.
