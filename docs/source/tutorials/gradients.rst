Level B from gradients
======================

When the level-B method has analytic gradients, its Hessian can be built from
finite differences of gradients instead of energies. That needs only
:math:`1 + 2n` calculations instead of :math:`1 + n(n+1)`. This follows
``examples/2_Methanol/molpro/CMA0A/gradients``.

Options
-------

.. code-block:: toml
   :caption: cma.toml

   coords = "Delocalized"
   covalent_radii = true     # find the bonds automatically
   topo_analysis = true
   deriv_level_b = 1         # level B from gradients
   program_a = "molpro"
   program_b = "molpro"
   cart_insert_a = 9
   cart_insert_b = 9

The level-B template must ask for the gradient; in Molpro add ``forces`` after
the method:

.. literalinclude:: ../../../examples/2_Methanol/molpro/CMA0A/gradients/templateB.dat
   :caption: templateB.dat

Reading the gradients
---------------------

For gradients CMA needs the lines that bracket the gradient block in the
output, ``gradient_regex_b = [start, end]``. Presets exist for Molpro (read
from ``output.xml``) and Psi4; for other programs, or a different output
layout, set it yourself. Every line between the two patterns that contains
three decimal numbers is read as one atom's gradient (its last three fields).

Running
-------

.. code-block:: console

   $ cma run
   Level B: wrote 25 molpro inputs to DispsB/

Methanol needs 25 gradient calculations at level B rather than 157 energies.
Level A is unchanged (``deriv_level_a = 0``), so it still needs 25 energies.
