Installation of PyImfit
=======================

Note that since PyImfit is only meant to work with Python 3
(specifically, versions 3.8 or later), I reference ``pip3`` instead of
just ``pip`` in the example installation commands below. If your version
of ``pip`` automatically installs into Python 3, then you don’t need to
explicitly specify ``pip3``.

Standard Installation: macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Precompiled Intel (x86-64) and Apple silicon (arm64) binary versions
(“wheels”) of PyImfit for macOS can be installed from PyPI via ``pip``:

::

   $ pip3 install pyimfit

PyImfit requires the following Python libraries/packages (which will
automatically be installed by ``pip`` if they are not already present):

-  Numpy
-  Scipy

Astropy is also useful for reading in FITS files as numpy arrays (and is
required by the unit tests).

Note that binary installs for Apple silicon (arm64) Macs are only
available for Python version 3.10 and later (binary installs for Intel
Macs are also available for Python 3.8 and 3.9).

Standard Installation: Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyImfit can also be installed on Linux using ``pip``. Since this
involves building from source, you will need to have a working
C++-11-compatible compiler (e.g., GCC version 4.8.1 or later); this is
probably true for any reasonably modern Linux installation. (**Note:** a
64-bit Linux system is required.)

::

   $ pip3 install pyimfit   [or "pip3 install --user pyimfit", if installing for your own use]

If the installation fails with a message containing something like
“fatal error: Python.h: No such file or directory”, then you may be
missing headers files and static libraries for Python development; see
`this Stackexchange
question <https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory>`__
for guidance on how to do that.

Conda Installations:
~~~~~~~~~~~~~~~~~~~~

PyImfit can be installed into conda environments on macOS and Linux,
via:

::

   $ conda install -c conda-forge perwin::pyimfit

Note that conda installation is only available for Python version 3.10
and later.
