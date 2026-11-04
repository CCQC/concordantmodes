CMA-0B: reusing an existing level-B Hessian
===========================================

If you already have a Hessian for the cheap method, for example from an
analytic frequency calculation at the level-B equilibrium geometry, CMA can
use it instead of computing level B by finite differences. Only the level-A
calculations remain. This follows ``examples/1_H2O/molpro/CMA0B/ZMAT``.

Two geometries
--------------

The cart block holds both geometries, separated by ``---``: first the level-B
geometry (where the Hessian was computed), then the level-A reference
geometry:

.. literalinclude:: ../../../examples/1_H2O/molpro/CMA0B/ZMAT/zmat
   :caption: zmat

The ZMAT block in this example is a Z-matrix, so ``coords`` stays at its
default ``"ZMAT"``.

The level-B Hessian
-------------------

Put the Cartesian Hessian in ``force_constants/fc_cart_b.dat`` in CFOUR's
``FCMFINAL`` format (:doc:`../reference/files`): a header with the number of
atoms and :math:`3N`, then the :math:`3N\times 3N` matrix in hartree/bohr²,
three numbers per line. Set ``cart_fc_b`` so CMA reads it as Cartesian:

.. code-block:: toml
   :caption: cma.toml

   cart_fc_b = true
   program_a = "molpro"
   cart_insert_a = 9

No ``templateB.dat`` is needed.

Running
-------

.. code-block:: console

   $ cma run
   Level B: reading force constants from force_constants/fc_cart_b.dat
   Level B harmonic frequencies (cm^-1):
   frequency #  1:    1690.30
   frequency #  2:    3821.60
   frequency #  3:    3927.58
   Level A: wrote 7 molpro inputs to DispsA/

The level-B frequencies printed here come from the Hessian shipped with the
example. Run the seven level-A calculations and ``cma run`` again for the
final frequencies.

As discussed in :doc:`../theory/index`, CMA-0A (level-B Hessian at the level-A
geometry) is generally more accurate than CMA-0B; CMA-0B is the choice when a
level-B Hessian is already available or cheap to obtain analytically.
