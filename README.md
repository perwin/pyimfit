# PyImfit

PyImfit is a Python wrapper for [Imfit](https://github.com/perwin/imfit), a C++-based program for fitting
2D models to scientific images. It is specialized for fitting astronomical images of galaxies, but can in 
principle be used to fit any 2D Numpy array of data. The underlying, Imfit-based library -- and thus the main
part of PyImfit's computation -- is multithreaded and naturally takes advantage of multiple CPU cores, and can
thus be very fast.

[Changelog](./CHANGELOG.md)

[comment]: <> ([![Build Status]&#40;https://travis-ci.org/perwin/pyimfit.svg?branch=master&#41;]&#40;https://travis-ci.org/perwin/pyimfit&#41;)

[comment]: <> (![PyImfit]&#40;https://github.com/perwin/pyimfit/workflows/PyImfit/badge.svg&#41;)
![PyImfit](https://github.com/perwin/pyimfit/actions/workflows/python-package.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyimfit/badge/?version=latest)](http://pyimfit.readthedocs.io/en/latest/?badge=latest)


## A Simple Example of Use

Assuming you want to fit an astronomical image (stored as a FITS file) named `galaxy.fits` using a model defined
in an Imfit configuration file named `config_galaxy.dat` (models can also be defined from within Python):

    from astropy.io import fits
    import pyimfit
    
    imageFile = "<path-to-FITS-file-directory>/galaxy.fits"
    imfitConfigFile = "<path-to-config-file-directory>/config_galaxy.dat"

    # read in image data, convert to proper double-precision, little-endian format
    image_data = pyimfit.FixImage(fits.getdata(imageFile))

    # construct model from config file (this can also be done programmatically within Python)
    model_desc = pyimfit.ModelDescription.load(imfitConfigFile)

    # create an Imfit object, using the previously created model configuration
    imfit_fitter = pyimfit.Imfit(model_desc)

    # load the image data and image characteristics (the specific values are
    # for a typical SDSS r-band image, where a sky-background value of 130.14
    # has already been subtracted), and then do a standard fit
    # (using default chi^2 statistics and Levenberg-Marquardt solver)
    imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
    
    # check the fit and print the resulting best-fit parameter values
    if imfit_fitter.fitConverged is True:
        print("Fit converged: chi^2 = {0}, reduced chi^2 = {1}".format(imfit_fitter.fitStatistic,
            imfit_fitter.reducedFitStatistic))
        bestfit_params = imfit_fitter.getRawParameters()
        print("Best-fit parameter values:", bestfit_params)


See the Jupyter notebook `pyfit_emcee.ipynb` in the `docs` subdirectory for
an example of using PyImfit with the Markov Chain Monte Carlo code [`emcee`](http://dfm.io/emcee/current/). (Online
version of notebook available [here](https://pyimfit.readthedocs.io/en/latest/pyimfit_emcee.html).)

Online documentation: [https://pyimfit.readthedocs.io/en/latest/](https://pyimfit.readthedocs.io/en/latest/).

Also useful: [Onine documentation for Imfit](https://imfit.readthedocs.io); and the main Imfit manual in PDF format:
[imfit_howto.pdf](http://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf)


## Requirements and Installation

PyImfit is designed to work with modern versions of Python 3 (3.8 or later on Linux and Intel macOS; 3.10 or later
on Apple Silicon macOS); no support for Python 2 is planned.

### Standard installation via pip: macOS

A precompiled binary version ("wheel") of PyImfit for macOS can be installed from PyPI via `pip`:

    $ pip3 install pyimfit

PyImfit requires the following Python libraries/packages (which will automatically be installed
by `pip` if they are not already present):

* Numpy
* Scipy

The `requests` package will also be installed if not already present, though this is only used
for running the test script.

The `astropy.io` package is also useful for reading in FITS files as numpy arrays (and is required by the
unit tests).


### Standard installation via pip: Linux

PyImfit can also be installed on Linux using `pip`. Since this involves building from source,
you will need to have a working C++-11-compatible compiler (e.g., GCC version 4.8.1 or later);
this is probably true for any reasonably modern Linux installation. (**Note:** a 64-bit Linux
system is required.)

    $ pip3 install pyimfit   [or "pip3 install --user pyimfit", if installing for your own use]

If the installation fails with a message containing something like "fatal error: Python.h: 
No such file or directory", then you may be missing headers files and static libraries for
Python development; see [this Stackexchange question](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory)
for guidance on how to deal with that.


### Installation via conda

PyImfit can also be installed into a conda environment on macOS or Linux, via

    $ conda install -c conda-forge perwin::pyimfit

Note that this only works for Python 3.10 or later.



### Building the Whole Thing from Source

To build PyImfit from the Github source, you will need the following:

   * Most of the same external (C/C++) libraries that Imfit requires: specifically 
   [FFTW3](https://www.fftw.org) [version 3], [GNU Scientific Library](https://www.gnu.org/software/gsl/) [version 2.0
   or later!], and [NLopt](https://nlopt.readthedocs.io/en/latest/)
   
   * This Github repository (use `--recurse-submodules` to ensure the Imfit repo is also downloaded)
           
           $ git clone --recurse-submodules git://github.com/perwin/pyimfit.git

   * A reasonably modern C++ compiler -- e.g., GCC 4.8.1 or later, or any C++-11-aware version of 
   Clang++/LLVM that includes support for OpenMP. See below for special notes about using
   the Apple-built version of Clang++ that comes with Xcode for macOS.


#### Steps for building PyImfit from source:

1. Install necessary external libraries (FFTW3, GSL, NLopt)

    * These can be installed from source, or via package managers (e.g., Homebrew on macOS)
        
    * Note that version 2.0 or later of GSL is required! (For Ubuntu, this means
    the `libgsl-dev` package for Ubuntu 16.04 or later.)

2. Clone the PyImfit repository

       $ git clone --recurse-submodules git://github.com/perwin/pyimfit.git

3. Build the Python package

   * **[macOS only:] First, specify a valid, OpenMP-compatible C++ compiler**
   
         $ export CC=<c++-compiler-name>; export CXX=<c++-compiler-name>
        
    (Note that you need to point CC and CXX to the *same*, Open-MP-compatible C++ compiler!
    This should not be necessary on a Linux system, assuming the default compiler is standard GCC.)
    
      * Versions of Apple's Clang compiler from Xcode 9 or later *can* compile OpenMP code, but you
      will need to also install the OpenMP library (e.g., `brew install libomp` if using Homebrew).
      See [here](https://iscinumpy.gitlab.io/post/omp-on-high-sierra/) for more details.
   
   * Build and install PyImfit!
   
      * For testing purposes (installs a link to current directory in your usual package-install location)

            $ python3 setup.py develop

      * For general installation (i.e., actually installs the package in your usual package-install location)

            $ python3 setup.py install


## Credits

PyImfit originated as [python-imfit](https://github.com/streeto/python-imfit), written by André Luiz de Amorim; 
the current, updated version is by Peter Erwin.

(See [the Imfit manual](http://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf) for additional credits.)


## License

PyImfit is licensed under version 3 of the GNU Public License.

