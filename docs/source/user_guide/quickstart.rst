Quickstart: water with Molpro
=============================

Install the package first (:doc:`installation`); this gives you the
``cma`` command.

Set up a run directory
----------------------

.. code-block:: bash

   mkdir water && cd water
   cma init --program molpro

This writes four files:

``zmat``
    The molecule (Cartesian geometry) and its internal coordinates. The
    starter file is water; see :doc:`zmat` for the format.
``templateB.dat`` / ``templateA.dat``
    Molpro inputs for the cheap (B) and expensive (A) levels. CMA replaces the
    ``{geometry}`` line with the coordinates of each displaced structure and
    ``{natoms}`` with the number of atoms. See :doc:`templates`.
``cma.toml``
    The options. The defaults are a reasonable start; ``cma options`` lists
    every option.

Run it
------

.. code-block:: bash

   cma run

With the default ``cluster = "manual"`` CMA writes the level-B inputs to
``DispsB/1``, ``DispsB/2``, ... and stops with a message saying what to run.
Run each ``input.dat`` so it writes ``output.dat`` in the same directory
(an example job script is left in ``DispsB/sub_script.sh``), then run
``cma run`` again. CMA reads the level-B results, writes the level-A inputs to
``DispsA``, and stops again; after those finish, a third ``cma run`` prints
the final frequencies.

At any point ``cma status`` shows how far each level has got:

.. code-block:: text

   Level B: 41 of 43 calculations finished; failed: 17; not run: 42
   Level A: not started

Running ``cma run`` again is always safe: finished calculations are never
redone, and only missing or failed ones are run (or resubmitted).

Let CMA run the jobs
--------------------

On a Slurm or SGE cluster, set ``cluster = "slurm"`` (or ``"sge"``) and the
job resources in ``cma.toml``; CMA then submits the calculations, waits for
them and carries on to the end in a single ``cma run``:

.. code-block:: toml

   cluster = "slurm"
   queue = "batch"
   nslots = 4
   memory = "16GB"
   time_limit = "12:00:00"
   job_setup = "module load Molpro"

On UGA's Sapelo2 with the HFS lab setup, ``cluster_profile = "sapelo2_hfslab"``
sets all of this at once. See :doc:`running`.

Results
-------

The console shows the level-B and final frequencies, the total energy
distribution and the frequency shifts; ``cma.log`` has the complete record.
The force constants are written to the ``force_constants/`` folder and the
normal modes to ``MOLDEN``. :doc:`outputs` describes every file.

Using a Python script instead
-----------------------------

Everything above also works from a ``main.py``:

.. code-block:: python

   from concordantmodes import ConcordantModes, Options

   options = Options(coords="Delocalized", program_a="molpro", program_b="molpro")
   status = ConcordantModes(options).run()   # "finished" or "waiting"

Run it with ``python main.py`` as many times as ``cma run`` above.
