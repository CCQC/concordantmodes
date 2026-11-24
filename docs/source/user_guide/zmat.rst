The zmat file
=============

``zmat`` holds the molecule and the internal coordinates the normal modes are
expressed in. It has two blocks; blank lines are ignored. CMA checks the file
before parsing and reports the line number of anything it cannot read.

The cart block
--------------

.. code-block:: text

   cart begin
        H        -0.00000000    -1.42462540     0.99592408
        O         0.00000000     0.00000000    -0.12550454
        H        -0.00000000     1.42462540     0.99592408
   cart end

One atom per line: an element symbol (optionally followed by an isotope
number, e.g. ``H2``) and three coordinates written with a decimal point.
Coordinates are in Bohr unless ``cart_coords = "Angstrom"``.

**Two geometries.** A line ``---`` splits the block into two structures with
the same atoms in the same order. The first is used for level B (the normal
modes); the second is the geometry at which the level-A calculations are
done. With a single structure both levels use it.

The ZMAT block
--------------

What goes between ``ZMAT begin`` and ``ZMAT end`` depends on the ``coords``
option. Atoms are numbered from 1 in the order of the cart block.

``coords = "ZMAT"``
    A Z-matrix without values: each line is an atom symbol followed by the
    atoms it is bonded to, makes an angle with, and makes a torsion with::

       C
       O 1
       H 1 2
       H 1 2 3

``coords = "Custom"``
    Exactly the internal coordinates to use, one per line:

    ===================  ==========================================================
    ``i j``              bond stretch i-j
    ``i j k``            angle bend i-j-k (j is the vertex)
    ``i j k l T``        torsion about the j-k bond
    ``i j k l O``        out-of-plane angle of the j-i bond from the k-j-l plane
    ``i j k l L``        linear bend
    ``i j k l Lx``       linear bend, first component
    ``i j k l Ly``       linear bend, second component
    ===================  ==========================================================

``coords = "Delocalized"``
    Only the bonds (``i j`` lines) are needed. CMA builds every angle,
    torsion and out-of-plane coordinate from that connectivity and makes the
    set non-redundant. With ``covalent_radii = True`` the bonds are found
    from the geometry and the ZMAT block can be left out.

Groups of atoms (centroids)
---------------------------

Any position in a coordinate can be the centroid of several atoms. Start each
position with ``;`` and list its atoms:

.. code-block:: text

   ; 2 ; 3 4 5            bond from atom 2 to the centroid of atoms 3, 4 and 5
   ; 6 ; 1 ; 3 4 5        angle 6-1-centroid(3,4,5)
   ; 6 ; 2 ; 1 ; 3 4 5 T  torsion

Complete examples for every mode are in the ``examples`` directory.
