Using symmetry
==============

Force constants between coordinates of different symmetry species
(irreducible representations) vanish by symmetry, so they need not be
computed. CMA can exploit this in two ways.

Automatic: MolSym
-----------------

With ``molsym_symmetry = true`` CMA uses the MolSym package to determine the
point group and build symmetry-adapted linear combinations (SALCs) of the
internal coordinates, and only computes force constants within each
symmetry block. Install MolSym first (``pip install -e ".[symmetry]"``).

This follows ``examples/3_Ammonia/molpro/CMA0A`` (:math:`C_{3v}` ammonia with
three stretches, three bends and three out-of-plane coordinates):

.. literalinclude:: ../../../examples/3_Ammonia/molpro/CMA0A/zmat
   :caption: zmat

.. code-block:: toml
   :caption: cma.toml

   coords = "Custom"
   molsym_symmetry = true
   program_a = "molpro"
   program_b = "molpro"
   cart_insert_a = 7
   cart_insert_b = 7

.. code-block:: console

   $ cma run
   Level B: wrote 27 molpro inputs to DispsB/

Without symmetry the same input needs 43 level-B calculations. The final
frequencies are labelled with their irreducible representation and
degeneracy.

``subgroup`` runs the analysis in a subgroup of the full point group (for
example ``"C2v"``), and ``exploit_pm_symm`` / ``only_TSIR`` skip further
displacements that are equivalent by symmetry.

Manual: your own symmetry coordinates
-------------------------------------

You can also supply symmetry coordinates yourself, from a Python script
(``examples/2_Methanol/molpro/CMA0A/manual_sym``). The projection matrix
``Proj`` turns the twelve primitive internal coordinates into symmetry
coordinates, and ``sym_sort`` lists which of them belong to each irreducible
representation (here :math:`A'` and :math:`A''` of :math:`C_s` methanol):

.. literalinclude:: ../../../examples/2_Methanol/molpro/CMA0A/manual_sym/main.py
   :caption: main.py
   :lines: 5-

With ``man_proj = True`` CMA uses ``Proj`` instead of building its own
coordinates, and with ``sym_sort`` it only computes force constants between
coordinates of the same irreducible representation.
