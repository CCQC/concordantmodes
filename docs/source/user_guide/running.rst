Running the calculations
========================

How a run proceeds
------------------

Each ``cma run`` (or ``python main.py``) looks at the files in the run
directory and does the next thing needed, level B first, then level A:

=================================  =====================================================================
On disk                            What CMA does
=================================  =====================================================================
no ``Disps<L>`` directory           writes the displacement inputs
some outputs missing or failed      ``manual``: stops and lists them; ``slurm``/``sge``: submits only
                                    those and waits; ``custom``: submits only those and stops
all outputs finished                reads the energies and builds the force constants
=================================  =====================================================================

Finished calculations are never overwritten or resubmitted. A level-B
calculation counts as finished when its output has the success line and an
energy (see :doc:`templates`). ``cma status`` reports the same information
without running anything.

If ``Disps<L>`` exists but no longer matches the current ``zmat``, template
or options (CMA regenerates the inputs in memory and compares them), the run
stops rather than mix old and new results. Move the directory aside
(``mv DispsA DispsA_old``) and run again to start that level fresh.

Once a level is done its force constants and gradient are saved in the
``force_constants/`` folder (``fc_int_b.dat``/``.grad``,
``fc_int_a.dat``/``.grad``), and later runs read those files and ignore
``Disps<L>``. Because the gradient is saved too, a rerun from the files
gives the same result as reading ``DispsA``, including with
``second_order``. To pick up rerun or corrected calculations in ``DispsA``,
delete ``force_constants/fc_int_a.dat`` (and ``.grad``). With
``write_molden = true`` (the default) the level-A displacements are rebuilt
in memory when ``fc_int_a.dat`` is read, so the MOLDEN file is still
written; nothing is written to ``DispsA``.

CMA looks for these files in ``force_constants/`` first and then in the run
directory, so runs from older versions (which kept them next to ``zmat``)
continue where they left off; new files always go into ``force_constants/``.

To skip level B entirely (CMA-0B), provide its Hessian in
``force_constants/``: ``fc_int_b.dat`` (internal coordinates) or, with
``cart_fc_b = True``, ``fc_cart_b.dat`` (Cartesian, CFOUR FCMFINAL format),
optionally with a matching ``fc_cart_b.grad``.

Choosing how jobs are run
-------------------------

``cluster = "manual"`` (default)
    CMA only writes inputs. Run each ``Disps<L>/<n>/input.dat`` however you
    like so that it leaves ``output.dat`` next to it, then run CMA again.

``cluster = "slurm"``
    CMA submits the calculations as job arrays and waits, checking ``sacct``
    every ``poll_interval`` seconds. Arrays larger than ``max_array_size``
    (set it to your cluster's ``MaxArraySize``; 5000 on Sapelo2) are split.
    ``array_throttle = 100`` limits how many tasks run at once.
    ``slurm_array = false`` submits one job per directory instead. Scheduler
    logs go to ``Disps<L>/slurm_logs``.

``cluster = "sge"``
    CMA submits task arrays with ``qsub -t`` and waits, checking ``qacct``.

``cluster = "custom"``
    CMA runs ``custom_submit_str <dir>/sub_script.sh`` in every directory that
    still needs a calculation, then stops; run CMA again when they are done.

The job script
--------------

For ``slurm`` and ``sge`` CMA writes the job script itself from these
options:

==========================  =============================================================
``job_name``                scheduler job name
``queue``                   partition / queue
``nslots``                  cores per calculation (``$NSLOTS`` in the script)
``memory``                  memory per calculation, scheduler syntax (``"16GB"``)
``time_limit``              wall time per calculation
``scheduler_directives``    extra lines, e.g. ``["#SBATCH --constraint=EPYC"]``
``job_setup``               shell lines run first (``module load ...``, ``conda activate ...``)
``run_command_a`` / ``_b``  command that runs one calculation; the default per program is below
==========================  =============================================================

Default run commands:

* molpro: ``molpro -n $NSLOTS --nouse-logfile -o output.dat input.dat``
* psi4: ``psi4 -n $NSLOTS -i input.dat -o output.dat``
* orca: ``"$(command -v orca)" input.dat > output.dat``
* cfour: ``xcfour > output.dat 2>&1``

**Your own script.** ``job_template = "my_job.sh"`` uses your file instead.
The placeholders ``{job_name}``, ``{queue}``, ``{nslots}``, ``{memory}``,
``{time_limit}``, ``{directives}``, ``{job_setup}``, ``{run_command}``,
``{program}`` and ``{level}`` are filled in; everything else (including shell
syntax such as ``${VAR}``) is left untouched. The script must run one
calculation in its directory: for Slurm job arrays CMA adds a few lines after
the ``#SBATCH`` header that set ``SLURM_SUBMIT_DIR`` to the task's
displacement directory and ``cd`` into it.

Cluster profiles
----------------

``cluster_profile`` applies a named set of the options above; anything you set
yourself still takes precedence.

``sapelo2_hfslab``
    UGA GACRC Sapelo2 with the HFS lab's Molpro container
    (``/work/hfslab/containers``) and ``psi4_dlpno`` conda environment: the
    job scripts the package used before profiles existed, submitted as job
    arrays of up to 5000 tasks.

``vulcan_ccqc``
    The CCQC Vulcan SGE scripts (CFOUR only).

New profiles go in ``PROFILES`` in ``concordantmodes/options.py``.

When something goes wrong
-------------------------

See :doc:`troubleshooting` for the common error messages and what to do about
them.
