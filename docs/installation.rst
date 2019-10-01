Installation of PyImfit
=======================

Note that since PyImfit is only meant to work with Python 3
(specifically, version 3.5 or later on Linux and version 3.6 or later on
macOS), I reference ``pip3`` instead of just ``pip`` in the example
installation commands below. If your version of ``pip`` automatically
installs into Python 3, then you don't need to explicitly specify
``pip3``.

Standard Installation: macOS
----------------------------

A precompiled binary version ("wheel") of PyImfit for macOS can be
installed from PyPI via ``pip``:

::

    $ pip3 install pyimfit

PyImfit requires the following Python libraries/packages (which will
automatically be installed by ``pip`` if they are not already present):

-  Numpy
-  Scipy

Astropy is also useful for reading in FITS files as numpy arrays (and is
required by the unit tests).

Standard Installation: Linux
----------------------------

PyImfit can also be installed on Linux using ``pip``. Since this
involves building from source, you will need to have a working
C++-11-compatible compiler (e.g., GCC version 4.8.1 or later); this is
probably true for any reasonably modern Linux installation. (**Note:** a
64-bit Linux system is required.)

::

    $ pip3 install pyimfit   [or "pip3 install --user pyimfit", if installing for your own use]
