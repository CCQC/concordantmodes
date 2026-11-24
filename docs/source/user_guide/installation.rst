Installation
============

Concordant Modes needs Python 3.10 or newer, NumPy, SciPy, SymPy and
QCElemental. The quantum chemistry programs it drives (Molpro, Psi4, ORCA or
CFOUR) are installed separately; CMA only writes their inputs and reads their
outputs.

With conda
----------

.. code-block:: bash

   git clone https://github.com/CCQC/ConcordantModes.git
   cd ConcordantModes
   conda env create -f devtools/conda-envs/base.yaml
   conda activate concordantmodes
   pip install -e .

With pip only
-------------

.. code-block:: bash

   git clone https://github.com/CCQC/ConcordantModes.git
   cd ConcordantModes
   python -m venv .venv && source .venv/bin/activate
   pip install -e .

``pip install -e .`` installs the ``concordantmodes`` Python package and the
``cma`` command. Check it with:

.. code-block:: bash

   cma --help
   python -c "import concordantmodes; print(concordantmodes.__version__)"

Optional extras
---------------

``pip install -e ".[symmetry]"``
    MolSym, needed only for ``molsym_symmetry = true`` (symmetry-adapted
    coordinates; see :doc:`../tutorials/symmetry`).

``pip install -e ".[tests]"``
    pytest, to run the test suite (:doc:`../developer/index`).

``pip install -e ".[docs]"``
    Sphinx and the theme, to build this manual.

On a cluster
------------

Install CMA in the same environment you submit jobs from. CMA itself does
little computing (it writes inputs, reads outputs and diagonalises small
matrices), so it can run on a login node while the displacement calculations
run as batch jobs; see :doc:`running`.
