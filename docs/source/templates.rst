Input templates and output parsing
==================================

Templates
---------

``templateB.dat`` and ``templateA.dat`` are ordinary inputs for the program
named by ``program_b`` and ``program_a``. For every displacement CMA copies
the template into ``DispsB/<n>/input.dat`` (``ZMAT`` for CFOUR) and

* replaces the line containing ``{geometry}`` with one line per atom
  (symbol and x, y, z in Bohr), and
* replaces ``{natoms}`` anywhere with the number of atoms.

Make sure the input reads the coordinates in Bohr (``bohr`` in Molpro,
``units bohr`` in Psi4, ``! Bohrs`` in ORCA, ``UNITS=BOHR`` in CFOUR).

Older inputs that set ``cart_insert_a`` / ``cart_insert_b`` (insert the
geometry after a given line number) still work; use one method or the other.
Files named ``GENBAS``, ``ECPDATA``, ``initden.dat`` or ``sub_script.sh`` in
the run directory are copied into every displacement directory.

Reading the results
-------------------

A calculation counts as finished when its output contains the program's
normal-termination line and an energy. The built-in patterns:

===========  ==========================================  ==========================================================
Program      Finished when the output contains            Energy read from
===========  ==========================================  ==========================================================
molpro       ``Molpro calculation terminated``            the final summary line, e.g. ``CCSD(T)/cc-pVDZ energy=``
psi4         ``Psi4 exiting successfully``                a ``CMA_ENERGY`` line CMA appends to each input (see below)
orca         ``ORCA TERMINATED NORMALLY``                 ``FINAL SINGLE POINT ENERGY``
cfour        ``The final electronic energy is``           ``The final electronic energy is``
===========  ==========================================  ==========================================================

**Molpro** prints a ``<method>/<basis> energy=`` line at the end of every
job holding the energy of the *last* method in the input. If your input runs
several methods and the one you want is not last, name it with ``method_a``
/ ``method_b`` (``hf``, ``rhf``, ``uhf``, ``mp2``, ``ccsd``, ``ccsd(t)``).

**Psi4** prints a different line for every method, so CMA appends one line
to each generated input::

   psi4.print_out("\nCMA_ENERGY %.12f\n" % psi4.variable("CURRENT ENERGY"))

``CURRENT ENERGY`` is the energy of the last ``energy()`` call in the input.

**Your own patterns.** ``energy_regex_a`` (a regular expression whose first
group captures the energy) and ``success_regex_a`` override the presets;
the same for level B. If ``energy_regex_*`` is set for Psi4, nothing is
appended to the inputs.

**Gradients** (``deriv_level_* = 1``) are read between two patterns given
by ``gradient_regex_*``; presets exist for Molpro and Psi4.
