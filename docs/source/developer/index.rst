Developer guide
===============

Setting up
----------

.. code-block:: bash

   git clone https://github.com/CCQC/ConcordantModes.git
   cd ConcordantModes
   pip install -e ".[tests,docs]"

Running the tests
-----------------

.. code-block:: bash

   pytest

The tests can be started from any directory; ``concordantmodes/tests/conftest.py``
switches to the tests directory, where the reference data lives
(``concordantmodes/tests/ref_data``). ``test_usability.py`` includes complete
CMA runs against a small fake Molpro (``ref_data/usability/fake_molpro.py``)
that computes water energies from a model potential, so the whole workflow is
tested without a quantum chemistry program.

``test_f_convert.py::test_f_convert2int`` is a known failure that predates
the current code.

Code style: format with ``black --target-version py310``.

How the code is organised
-------------------------

A run (:meth:`ConcordantModes.run <concordantmodes.cma.ConcordantModes.run>`)
goes through these modules:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Role
   * - ``options.py``
     - The option table (``OPTION_SPECS``), validation, cluster profiles.
   * - ``zmat.py``
     - Reads and checks ``zmat``; builds the internal coordinates.
   * - ``s_vectors.py``
     - Wilson B matrix (vectorised), its second derivative, redundancy
       projection.
   * - ``g_matrix.py``, ``gf_method.py``, ``ted.py``
     - G matrix, GF eigenproblem, total energy distribution.
   * - ``algorithm.py``
     - Which force constants (index pairs) a level needs.
   * - ``transf_disp.py``
     - Displaced Cartesian geometries for those force constants.
   * - ``workflow.py``
     - Decides from the files on disk what each level still needs; stops
       (``CMAWaiting``) or submits.
   * - ``directory_tree.py``
     - Writes ``Disps<L>/<n>/input.dat`` from the templates.
   * - ``submit.py``, ``cluster_templates.py``
     - Job scripts and submission/polling for Slurm, SGE and custom commands.
   * - ``reap.py``, ``programs.py``
     - Read energies/gradients from the outputs; per-program patterns.
   * - ``force_constant.py``
     - Finite-difference force constants and gradients.
   * - ``f_convert.py``, ``f_read.py``, ``g_read.py``, ``fc_files.py``
     - Coordinate transformations and the ``force_constants/`` files.
   * - ``cli.py``, ``log.py``, ``errors.py``
     - The ``cma`` command, logging, exception types.

Conventions
~~~~~~~~~~~

* Raise :class:`~concordantmodes.errors.CMAError` with a message that says
  what is wrong and how to fix it; never a bare ``RuntimeError``.
* Use :class:`~concordantmodes.errors.CMAWaiting` to stop a run that has to
  wait for calculations.
* Log through ``logger = logging.getLogger(__name__)``; results the user
  needs go to ``INFO``, everything else to ``DEBUG`` (``debug_print`` and
  ``info_print`` in ``log.py`` take ``print``-style arguments). Do not use
  ``print`` in library code.

Adding an option
----------------

Add an ``OptionSpec`` to ``_SPECS`` in ``options.py`` with its default,
group, one-line description and allowed values (``choices``, ``types`` or a
``check`` function). That is all: validation, ``cma options`` and the
:doc:`../reference/options` page are generated from it. If an option is
removed, add it to ``REMOVED_OPTIONS`` with a sentence telling users what to
do instead.

Supporting a new program
------------------------

1. Add an entry to ``PROGRAMS`` in ``programs.py``: the success pattern, the
   energy pattern (and per-method patterns if the program prints several),
   gradient patterns if known, and the input file name.
2. Add the name to ``DirectoryTree.PROG_LIST``.
3. Add a default run command to ``DEFAULT_RUN_COMMANDS`` in
   ``cluster_templates.py`` and a template to ``TEMPLATES`` in ``cli.py`` for
   ``cma init``.
4. Add a test with a short real output to ``test_usability.py``
   (see ``test_molpro_presets_on_real_output``), and a row to the table in
   :doc:`../user_guide/templates`.

Adding a cluster profile
------------------------

Add an entry to ``PROFILES`` in ``options.py``: a description and the option
values it sets. Most profiles only need ``cluster``, ``queue``, ``job_setup``
and the run commands. For a completely different job script, add it as
``builtin:<name>`` in ``cluster_templates.build_job_script`` (see
``sapelo_template.py``). Then document it in
:doc:`../user_guide/running`.

Building this manual
--------------------

.. code-block:: bash

   cd docs
   make html            # output in docs/build/html
   # or: sphinx-build -W -b html source build/html

``reference/options.rst`` is generated on every build from the option table;
do not edit it by hand. Read the Docs builds the manual from
``.readthedocs.yaml`` on every push once the repository is connected at
https://readthedocs.org, with warnings treated as errors.
