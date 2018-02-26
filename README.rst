GRanada Relativistic Runaway (GRRR) code
========================================

`grrr` is a numerical code to simulate high energy particles in atmospheric
physics.  In particular it was designed to simulate electrons moving in a
self-consistent electric field, as described in [Luque2014]_.

Setting up `grrr`
^^^^^^^^^^^^^^^^^

You can download the `grrr` code from its
`main repository <https://github.com/aluque/grrr>`. 

`grrr` is mostly written in Python but the time-critical parts are written
in a C library that you have to compile.  You can do it with::

  make

This will produce a file named `libgrrr.so` that will be loaded at runtime.
Therefore it must be visible to your dynamic linker.  In Linux system you
can make sure that this is the case by setting `LD_LIBRARY_PATH`::

  export LD_LIBRARY_PATH=path/to/grrr

In a Mac OS X system change that to::

  export DYLD_LIBRARY_PATH=path/to/grrr


Note that `grrr` requires at least Python 3.0 and a few Python libraries:
numpy, scipy, matplotlib, h5py, and PyYAML.



.. [Luque2014] *Relativistic Runaway Ionization Fronts*, A. Luque, *Physical Review Letters*, **112**, 045003 doi:10.1103/PhysRevLett.112.045003 (2014).
