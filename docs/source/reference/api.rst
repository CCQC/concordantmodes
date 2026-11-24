Python API
==========

Most users only need :class:`~concordantmodes.options.Options` and
:class:`~concordantmodes.cma.ConcordantModes`:

.. code-block:: python

   from concordantmodes import ConcordantModes, Options

   options = Options(coords="Delocalized", program_a="molpro", program_b="molpro")
   status = ConcordantModes(options).run()   # "finished" or "waiting"

The remaining classes are the building blocks of a run and are useful for
scripting individual steps.

Running a calculation
---------------------

.. autoclass:: concordantmodes.cma.ConcordantModes
   :members: run

.. autoclass:: concordantmodes.options.Options
   :members: validate, regexes, changed, no_effect_set

.. autofunction:: concordantmodes.options.option_reference

Errors
------

.. automodule:: concordantmodes.errors
   :members:

Geometry and coordinates
------------------------

.. autoclass:: concordantmodes.zmat.Zmat
   :members: run, zmat_read, zmat_process, check_input, check_coordinate_lines

.. autoclass:: concordantmodes.s_vectors.SVectors
   :members: run

.. autoclass:: concordantmodes.g_matrix.GMatrix
   :members: run

.. autoclass:: concordantmodes.int2cart.Int2Cart

Vibrational analysis
--------------------

.. autoclass:: concordantmodes.gf_method.GFMethod
   :members: run

.. autoclass:: concordantmodes.ted.TED
   :members: run

.. autoclass:: concordantmodes.f_convert.FcConv
   :members: run, print_const

Displacements and force constants
---------------------------------

.. autoclass:: concordantmodes.algorithm.Algorithm
   :members: run

.. autoclass:: concordantmodes.transf_disp.TransfDisp
   :members: run

.. autoclass:: concordantmodes.force_constant.ForceConstant
   :members: run

Running and reading calculations
--------------------------------

.. autoclass:: concordantmodes.directory_tree.DirectoryTree
   :members: run

.. autoclass:: concordantmodes.submit.Submit
   :members: run, plan_array_chunks, array_blocks

.. autoclass:: concordantmodes.reap.Reap
   :members: run, check_dir

.. automodule:: concordantmodes.workflow
   :members: preflight, prepare_level, status

.. automodule:: concordantmodes.programs
   :members: energy_regex, success_regex, gradient_regex, program_name

.. automodule:: concordantmodes.cluster_templates
   :members: build_job_script, run_command

Files
-----

.. autoclass:: concordantmodes.f_read.FcRead
   :members: run

.. autoclass:: concordantmodes.g_read.GrRead
   :members: run

.. automodule:: concordantmodes.fc_files
   :members:

.. autoclass:: concordantmodes.molden_writer.MoldenWriter
   :members: run

Symmetry
--------

.. autoclass:: concordantmodes.symmetry.Symmetry
   :members: run

Logging
-------

.. automodule:: concordantmodes.log
   :members: setup_logging
