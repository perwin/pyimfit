# PyImfit

PyImfit is a Python wrapper for [Imfit](), a C++-based astronomical image-fitting 
program.

**WARNING: This is currently a work in progress, and is not yet ready for general use!**

## A Simple Example of Use

Assuming you want to fit an image named `galaxy.fits` using a model defined
in an Imfit configuration file named `config_galaxy.dat`:

    from astropy.io import fits
    import pyimfit
    
    imageFile = "<path-to-FITS-file-directory>/galaxy.fits"
    imfitConfigFile = "<path-to-config-file-directory>/config_galaxy.dat"

    # read in image data, convert to proper double-precisions, little-endian format
    image_data = pyimfit.FixImage(fits.getdata(imageFile))

    # construct model from config file; construct new Imfit fitter based on model
    model_desc = pyimfit.ModelDescription.load(configFile)

    # create an Imfit object, using the previously loaded model configuration
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
an example of using PyImfit with the Markov Chain Monte Carlo code [`emcee`](http://dfm.io/emcee/current/).


## Requirements and Installation

PyImfit is designed to work with modern versions of Python 3 (nominally 3.5 or later); no support for 
Python 2.7 is planned.

### Standard Installation

Precompiled binary versions of PyImfit for Linux and macOS can be **[not yet, but soon!]** installed via PyPI:

    $ pip install pyimfit

PyImfit requires the following Python libraries/packages (which will automatically be installed
by pip if they are not already present):

* Numpy
* Scipy
* Astropy

### Building from Source

To build PyImfit from source, you will need the following:

1. This Github repository (use `--recurse-submodules` to ensure the Imfit repo is also downloaded)

       $ git clone --recurse-submodules git://github.com/perwin/pyimfit.git


2. [SCons](http://scons.org)

3. Cython (can be installed via pip)

4. A reasonably modern C++ compiler -- e.g., GCC 4.8.1 or later, or 
any C++-11-aware version of Clang++/LLVM that includes OpenMP support 
(note that this does *not* include the Apple-built version of Clang++
that comes with Xcode for macOS, since that does not include OpenMP support).


Steps for building PyImfit from source:

1. Build the static-library version of Imfit

        $ cd imfit
        $ scons libimfit.a


2. Build the Python package

        $ cd ..
        $ python3 setup.py XXX


## Credits

PyImfit originated as [python-imfit](https://github.com/streeto/python-imfit), written by André Luiz de Amorim; 
the current, updated version is by Peter Erwin.


## License

PyImfit is licensed under version 3 of the GNU Public License.

