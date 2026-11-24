File formats
============

Input files
-----------

``zmat``
    Geometry and internal coordinates; see :doc:`../user_guide/zmat`.

``templateA.dat``, ``templateB.dat``
    Program inputs with ``{geometry}`` and ``{natoms}`` placeholders; see
    :doc:`../user_guide/templates`.

``cma.toml``
    Options in `TOML <https://toml.io>`_ syntax, one ``name = value`` per
    line: strings in quotes, ``true``/``false``, numbers and ``[lists]``.
    Unknown names are errors. ``--set`` on the command line overrides
    individual values.

Displacement directories
------------------------

``DispsB`` and ``DispsA`` contain one numbered directory per calculation.
Directory ``1`` is always the reference geometry; the following pairs are the
plus and minus displacements, in the order CMA lists the force constants.
Each holds the program input (``input.dat``; ``ZMAT`` for CFOUR) and, after
the calculation, ``output.dat`` (and ``output.xml`` for Molpro gradients).
With Slurm job arrays, scheduler logs go to ``Disps<L>/slurm_logs`` and the
job script is ``Disps<L>/sub_script.sh``.

Force constants
---------------

All force-constant files are written to ``force_constants/`` (files in the run
directory from older versions are still read). They use the layout of CFOUR's
``FCMFINAL`` file:

.. code-block:: text

       3    9
           0.0000000213        0.0000000000        0.0000000000
          -0.0000000361        0.0000000000        0.0000000000
   ...

(the first lines of the water Hessian shipped with the CMA-0B example).

The first line holds the number of atoms and the dimension :math:`m` of the
matrix; then follow the :math:`m^2` elements row by row, three per line, in
hartree and bohr (``units = "MdyneAng"`` writes mdyn/Å instead).

``fc_cart_a.dat``, ``fc_cart_b.dat``
    Cartesian force constants, :math:`m = 3N`.

``fc_int_b.dat``
    Level-B force constants in the (non-redundant) internal coordinates.

``fc_int_a.dat``
    Level-A force constants in the level-B concordant normal coordinates
    (diagonal for CMA-0).

Gradients
---------

``fc_int_b.grad``, ``fc_int_a.grad``, ``fc_cart_a.grad``, ``fc_cart_b.grad``
    The gradient in the same coordinates as the matching ``.dat`` file,
    three numbers per line and no header.

MOLDEN
------

``MOLDEN`` contains the level-A geometry (``[ATOMS] AU``), the final
frequencies (``[FREQ]``), the geometry again (``[FR-COORD]``) and one
displacement vector per mode (``[FR-NORM-COORD]``), and can be opened in
Molden, Jmol, Avogadro and similar viewers to animate the modes.
