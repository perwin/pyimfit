Installation of PyImfit
=======================

Note that since PyImfit is only meant to work with Python 3
(specifically, versions 3.8 or later), I reference ``pip3`` instead of
just ``pip`` in the example installation commands below. If your version
of ``pip`` automatically installs into Python 3, then you don’t need to
explicitly specify ``pip3``.

Standard Installation: macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A precompiled Intel (x86-64) binary version (“wheel”) of PyImfit for
macOS can be installed from PyPI via ``pip``:

::

   $ pip3 install pyimfit

PyImfit requires the following Python libraries/packages (which will
automatically be installed by ``pip`` if they are not already present):

-  Numpy
-  Scipy

Astropy is also useful for reading in FITS files as numpy arrays (and is
required by the unit tests).

Installation on Apple Silicon (e.g., M1, M2) Macs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At present, the precompiled version of PyImfit is an x86-64 (“Intel”)
binary. This will not work with arm64 (Apple Silicon, including M1 and
M2 chips) versions of Python. However, you *can* run PyImfit on an
arm64-based Mac by running an x86-64 version of Python. How you do this
depends on how you prefer to install and use Python (e.g., python.org
installers vs conda vs Homebrew); there are various suggestions
available online for how to do this, depending on how you prefer to
install Python.

(The plan is to have a future release of PyImfit that natively supports
arm64 as well as x86-64.)

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
