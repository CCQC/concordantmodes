The ``cma`` command
===================

``cma`` is installed with the package. It reads the options from
``cma.toml`` in the current directory unless ``--input`` names another file.

.. code-block:: text

   cma [--debug] {init,run,status,options} ...

``--debug``
    Show the full Python traceback when an error occurs (normally only the
    error message is printed).

``cma init``
------------

Write starter input files in the current directory: ``cma.toml``,
``templateA.dat``, ``templateB.dat`` and ``zmat`` (water). Existing files are
never overwritten.

``--program {molpro,psi4,orca,cfour}``
    Program for the two templates (default ``molpro``).

``cma run``
-----------

Run the calculation, or continue it from where the last run stopped
(:doc:`../user_guide/running`).

``-i FILE``, ``--input FILE``
    Options file (default ``cma.toml``).

``--set NAME=VALUE``
    Override one option for this run; may be repeated. ``VALUE`` is read as a
    Python literal when possible (``--set nslots=4``,
    ``--set second_order=True``) and as a string otherwise
    (``--set verbosity=debug``).

Exit status:

== ==========================================================
0  finished: final frequencies computed
3  waiting: calculations still have to run (the message says which)
1  error (the message says what to fix)
== ==========================================================

The waiting status makes it easy to drive CMA from a script, for example a
cron job that calls ``cma run`` until it returns 0.

``cma status``
--------------

Show how far each level has got without running anything, e.g.

.. code-block:: text

   Level B: done (force constants read from force_constants/fc_int_b.dat)
   Level A: 5 of 7 calculations finished; not run: 6, 7

Takes the same ``--input`` and ``--set`` arguments as ``cma run``.

``cma options``
---------------

List options with their defaults and descriptions (the same information as
:doc:`options`).

``NAME``
    Only show options whose name contains ``NAME``, e.g. ``cma options
    regex``.

``--all``
    Also list options that are accepted for old inputs but have no effect.
