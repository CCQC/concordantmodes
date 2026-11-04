Concordant Modes
================

**Concordant Modes** (the Concordant Mode Approach, CMA) computes harmonic
vibrational frequencies at a high level of theory for a fraction of the usual
cost. Normal modes from an inexpensive method (level B) are used as the
coordinates for the expensive one (level A), so the level-A force-constant
matrix is very nearly diagonal and only its diagonal (plus, optionally, a few
selected couplings) has to be computed.

.. code-block:: bash

   pip install -e .          # from a clone of the repository
   cma init --program molpro # write starter input files
   cma run                   # run, or continue, the calculation

Where to start
--------------

:doc:`user_guide/index`
    Installation, a first calculation, the input files and how calculations
    are run on a workstation or a cluster.

:doc:`tutorials/index`
    Worked examples: water, methanol in delocalized coordinates, CMA-0B,
    gradients and symmetry.

:doc:`theory/index`
    What CMA does and why it works: the GF method, concordant normal
    coordinates, CMA-0A, CMA-0B and CMA-2.

:doc:`reference/index`
    Every option, the ``cma`` command, the files CMA reads and writes, and the
    Python API.

:doc:`developer/index`
    Running the tests, how the code is organised, and how to add a program,
    a cluster profile or an option.

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/index
   tutorials/index
   theory/index
   reference/index
   developer/index
   citing
