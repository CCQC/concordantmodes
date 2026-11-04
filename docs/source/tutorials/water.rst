Water: a first CMA-0A calculation
=================================

This tutorial computes CCSD(T)/cc-pVTZ harmonic frequencies of water using
CCSD(T)/cc-pVDZ normal modes (the starter files written by ``cma init``,
also in ``examples/0_quickstart/molpro``).

Input files
-----------

``zmat`` gives the geometry (in bohr) and the two O–H bonds; with
``coords = "Delocalized"`` CMA builds the H–O–H angle itself:

.. literalinclude:: ../../../examples/0_quickstart/molpro/zmat
   :caption: zmat

``templateB.dat`` and ``templateA.dat`` are Molpro inputs that differ only in
the basis set. CMA replaces ``{natoms}`` and the ``{geometry}`` line for every
displacement:

.. literalinclude:: ../../../examples/0_quickstart/molpro/templateA.dat
   :caption: templateA.dat

``cma.toml`` holds the options; the defaults run the calculations
"manually", i.e. CMA writes the inputs and you run them:

.. literalinclude:: ../../../examples/0_quickstart/molpro/cma.toml
   :caption: cma.toml
   :language: toml

Level B
-------

.. code-block:: console

   $ cma run
   Level B: wrote 13 molpro inputs to DispsB/
   Level B: 0 of 13 calculations finished.
     Not run yet: 13 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13).
   Run input.dat in each of these directories of DispsB/ so that it writes
   output.dat next to it, then run CMA again. ...

Water has three vibrational coordinates, so the full level-B Hessian needs
:math:`1 + 3\cdot 4 = 13` energies: the reference in ``DispsB/1`` and plus/minus
displacements for every diagonal and off-diagonal force constant. Run them,
for example:

.. code-block:: bash

   for d in DispsB/*/; do (cd "$d" && molpro -o output.dat input.dat); done

``cma status`` shows the progress at any time.

Level A
-------

.. code-block:: console

   $ cma run
   Level B: all 13 calculations finished.
   Level B harmonic frequencies (cm^-1):
   ...
   Level A: wrote 7 molpro inputs to DispsA/

Only the three diagonal force constants in the level-B normal coordinates are
needed at level A: :math:`1 + 2\cdot 3 = 7` CCSD(T)/cc-pVTZ energies instead of
13. Run them the same way, then:

.. code-block:: console

   $ cma run
   Level A: all 7 calculations finished.

   Final (CMA) harmonic frequencies (cm^-1):
   frequency #  1:  ...
   ...
   Frequency shift from level B to CMA (cm^-1):
     mode      level B        CMA      shift
   ...

The final Cartesian force constants are in ``force_constants/fc_cart_a.dat``
and the normal modes in ``MOLDEN`` (:doc:`../user_guide/outputs`).

Letting CMA run the jobs
------------------------

On a Slurm cluster add, for example,

.. code-block:: toml

   cluster = "slurm"
   nslots = 4
   memory = "8GB"
   job_setup = "module load Molpro"

and a single ``cma run`` submits both levels as job arrays, waits for them,
and prints the results.
