Troubleshooting
===============

Every error CMA reports names the file, line or option involved. This page
collects the common ones. Two general tools:

* ``cma run --set verbosity=debug`` prints every intermediate step (the same
  detail is always in ``cma.log``).
* ``cma run --debug`` shows the Python traceback of an error.

Options
-------

``'cluser' is not a Concordant Modes option. Did you mean 'cluster'?``
    A misspelled option name in ``cma.toml`` or ``Options(...)``. Unknown
    names are errors rather than being ignored. ``cma options <part of a
    name>`` lists matching options.

``'gen_disps_b' has been removed ...``
    Older inputs set ``gen_disps_*`` and ``calc_*`` to control the stages of
    a run. CMA now works the stage out from the files on disk; delete those
    lines (see :doc:`running`).

``Invalid options: coords = 'delocalised': must be one of 'ZMAT', 'Custom', 'Delocalized'.``
    A value outside the allowed choices or of the wrong type. All invalid
    values are listed together.

``Option 'autosalcs' is set but currently has no effect.``
    A warning for options that are still accepted from old inputs but do
    nothing; the run continues. Remove the option to silence it.

Input files
-----------

``Missing input file(s) in ...``
    ``zmat`` and the template(s) for every level that still has to be
    computed must be in the run directory. Run ``cma init`` for starting
    points.

``zmat line 3 ('2 7') refers to atom 7 but the cart block has 3 atoms``
    Atom numbers in the ZMAT block count from 1 in the order of the cart
    block.

``zmat line 3 ('2 x3') is not a recognised coordinate line``
    The line does not match any coordinate form for the chosen ``coords``
    mode. A Z-matrix file (``O``, ``H 1``, ...) only works with
    ``coords = "ZMAT"``; see :doc:`zmat`.

``... is not a valid Cartesian line``
    Each cart line needs an element symbol and three numbers written with a
    decimal point.

``CMA does not know where to put the geometry in templateB.dat``
    Add a line containing ``{geometry}`` to the template (or set
    ``cart_insert_b``); see :doc:`templates`.

``Found fc_b.dat, but CMA reads level-B force constants from force_constants/fc_int_b.dat``
    A level-B Hessian under an old name. For a Cartesian Hessian set
    ``cart_fc_b = true`` and rename it ``force_constants/fc_cart_b.dat``.

Calculations
------------

``Level B: 41 of 43 calculations finished. ... Failed (output present but no success/energy line): 1 (17).``
    With ``cluster = "manual"`` CMA stops and lists what still has to run.
    "Failed" means ``output.dat`` exists but does not contain the program's
    normal-termination line or an energy. Look at that output, fix and rerun
    it, then run CMA again.

``Level A: 2 calculation(s) still have no successful output after the jobs left the queue``
    With ``slurm``/``sge`` the jobs ran but did not finish properly (time
    limit, memory, a crash). Their scheduler logs are in
    ``Disps<L>/slurm_logs``. Running CMA again resubmits only those
    calculations.

``DispsA was made from different inputs (DispsA/2/input.dat differs ...)``
    The inputs in ``DispsA`` no longer match what the current ``zmat``,
    template and options produce, so CMA will not mix them. If you meant to
    change something, move the old directory aside (``mv DispsA DispsA_old``)
    and run again; if not, undo the change.

The energy is not found although the calculation finished
    The output does not contain the line the preset looks for, for example
    because a Molpro input runs several methods. Set ``method_a``/
    ``method_b``, or give your own ``energy_regex_a``/``energy_regex_b``
    (:doc:`templates`). ``cma status`` shows which outputs count as failed.

``sbatch failed in .../DispsB: ...``
    The scheduler rejected the job script; its message follows. Common
    causes are a wrong ``queue`` (partition), an array larger than the
    cluster allows (lower ``max_array_size``) or a per-user job limit.

Results
-------

Negative frequencies
    Imaginary frequencies are printed as negative numbers. At a minimum they
    usually mean the reference geometry is not converged at the level of
    theory used, or a displacement size is too small for the numerical noise
    of the calculations.

``second_order is set but no level-A gradient was found ...``
    Level A was read from an ``fc_int_a.dat`` written by an older version,
    without ``fc_int_a.grad``. Delete ``force_constants/fc_int_a.dat`` so
    the force constants and gradient are recomputed from ``DispsA``.

Features
--------

``molsym_symmetry = True needs the MolSym package``
    Install it with ``pip install -e ".[symmetry]"``.

``CMA-2 (off_diag = 2) is not finished in this version``
    The CMA-2 implementation in this version is incomplete; use
    ``off_diag = 0`` (CMA-0). See :doc:`../theory/index`.
