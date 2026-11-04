Results and output files
========================

Console and log
---------------

With the default ``verbosity = "normal"`` CMA prints its progress and, at the
end:

* the level-B harmonic frequencies;
* the final (CMA) frequencies;
* the total energy distribution (TED): how much each internal coordinate
  contributes to each normal mode, in percent;
* the zero-point vibrational energy in cm\ :sup:`-1`, kcal mol\ :sup:`-1` and
  hartree;
* a table of the shift of every frequency from level B to level A, with the
  RMS and largest shift.

``verbosity = "quiet"`` prints only warnings and what you need to do next;
``"debug"`` prints every intermediate matrix. Whatever the console shows, the
complete debug-level record of every run is appended to ``cma.log``
(``log_file = ""`` turns this off).

Files
-----

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Contents
   * - ``DispsB/<n>/``, ``DispsA/<n>/``
     - One directory per displaced geometry: ``input.dat`` (``ZMAT`` for
       CFOUR), and after the calculation ``output.dat``. Directory ``1`` is
       the reference geometry.
   * - ``force_constants/fc_int_b.dat`` / ``.grad``
     - Level-B force constants and gradient in the internal coordinates.
   * - ``force_constants/fc_int_a.dat`` / ``.grad``
     - Level-A force constants and gradient in the level-B normal
       coordinates.
   * - ``force_constants/fc_cart_a.dat``
     - Final Cartesian force constants (CFOUR ``FCMFINAL`` format); with
       ``second_order`` also ``fc_cart_a.grad``.
   * - ``MOLDEN``
     - Geometry, frequencies and normal modes for Molden, Jmol, Avogadro and
       similar viewers (``write_molden = false`` turns it off).
   * - ``cma.log``
     - Full log of every run.

The formats are described in :doc:`../reference/files`.

Reusing results
---------------

Once a level is finished its force constants are saved, and later runs read
them instead of the ``Disps`` directories. That makes rerunning a finished
calculation cheap (for example with a different ``verbosity`` or to rewrite
the MOLDEN file). To make CMA read ``DispsA`` again after rerunning or fixing
calculations there, delete ``force_constants/fc_int_a.dat`` and
``fc_int_a.grad``; for level B delete ``fc_int_b.*``.
