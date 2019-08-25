# Overview of PyImfit

PyImfit is a Python wrapper around the (C++-based) image-fitting program 
[Imfit](https://www.mpe.mpg.de/~erwin/code/imfit) 
([Github site](https://github.com/perwin/imfit)).

**Terminology:**

**Imfit** (boldface) refers to the C++ program (and the C++-based library that PyImfit is
built upon).

PyImfit is the general name for this Python package; `pyimfit` is the official Python
name (e.g., `import pyimfit`).

Finally, `Imfit` refers to the `pyimfit.Imfit` class, which does most of the work.



## For Those Already Familiar with Imfit

If you've already used the command-line version of **Imfit**, here are the essential things to know:

   * PyImfit operates on 2D NumPy arrays instead of FITS files; to use a FITS file, read it into
   Python via, e.g., [astropy.io.fits](http://docs.astropy.org/en/stable/io/fits/).
   
   * Models (and initial parameter values and parameter limits for a fit) are specified via the 
   ModelDescription class. The utility function `parse_config_file` 
   will read a standard **Imfit** configuration file and return an instance of that class with the
   model specification. (Or you can build up a ModelDescription instance by programmatically
   specifying components from within Python.)
   
   * Fitting is done by instantiating an `Imfit` object with a ModelDescription object as
   input, then adding a 2D NumPy array as the data to be fit (along with, optionally, mask
   and error images, the image A/D gain value, etc.) with the `loadData` method, and then calling 
   the `doFit` method (along with the minimization algorithm to use). Or just call the `fit` method 
   and supply the data image, etc., as part of its input.
   
   * Once the fit is finished, information about the fit (final &chi;<sup>2</sup> value, best-fit paremeter
   values, etc.) and the best-fitting model image can be obtained by querying properties and methods 
   of the `Imfit` object.

See [Sample Usage](./sample_usage.html) for a simple example of how this works.


## The Basics

There are three basic things you can do with PyImfit:

   1. Generate model images
   
   2. Fit models to a pre-existing (data) image
   
   3. Generate &chi;<sup>2</sup> or likelihood values from a comparison of model and data (e.g., for use
   with other fitting software, MCMC analysis, etc.)

In **Imfit** (and PyImfit), a "model" consists of one or more *image functions* from a library built into
**Imfit**, sharing one or more common locations within an image and added together to form a
summed model image. Each image function can generate a 2D image; the final model image is the sum
of its component image functions. Optionally, the summed model image can then be convolved with a user-supplied
Point-Spread Function (PSF) image to simulate the effects of seeing and telescope optics. (For greater accuracy,
subsections of the image can be oversampled on a finer pixel scale and convolved with a
correspondingly oversampled PSF image or images; these subsection are then downsampled back to
the native image pixel scale.)


### Specify the model (and its parameters)

The model is specified by an instance of the ModelDescription class.

For the command-line program, this is done via a "configuration" text file, which has a
specific format described in [the Imfit manual (PDF)](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf),
or in [this page of the online docs](https://imfit.readthedocs.io/en/latest/config_file_format.html).

If you have a configuration file, you can load it via the convenience function `parse_config_file`

    model_desc = pyimfit.parse_config_file(configFilePath)

where `configFilePath` is a string specifying the path to the configuration file.

You can also construct a ModelDescription instance programmatically from within Python; see below
for a very simple example, or [Sample Usage](./sample_usage.html) for a slightly more
complicated example.

(You can get a list of the available image functions -- "Sersic", "Exponential", etc. -- from the package-level 
variable `pyimfit.imageFunctionList`, and you can get a list of the parameter names for each 
function from `pyimfit.imageFunctionDict`. These functions are described in detail in
[the Imfit manual (PDF)](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf).)

Once you have a ModelDescription object describing the model, you can create an instance of
the `Imfit` class based on the model; optionally, if you want the model to be convolved
with a PSF, you can also supply the PSF image (in the form of a 2D NumPy array):

    imfitter = pyimfit.Imfit(model_desc)
    imfitter = pyimfit.Imfit(model_desc, psfImage)

#### Creating a model; setting parameter values and limits

Each image-function parameter within a model can have a "current" value (e.g., the initial guess for 
the fitting process, the result from the fit, etc.) and either: a set of lower and upper limits for 
possible values **or** the keyword "fixed", which means the parameter value should be kept constant during fits.

**Note**: Unless otherwise specified, all size values are in pixels, and all intensity/surface-brightness
values are in counts/pixel. (Photometric zero points are not needed except for the optional case of
computing model magnitudes; see below.)

A very simple example of constructing a model:

    model_desc = pyimfit.SimpleModelDescription()
    # define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
    # (note that Imfit treats image coordinates using the IRAF/Fortran numbering scheme: the lower-left
    # pixel in the image has coordinates (x,y) = (1,1))
    model_desc.x0.setValue(105, [95,115])
    model_desc.y0.setValue(62, [52,72])

    # create an Exponential image function, then define the parameter initial values and limits
    disk = pyimfit.make_imfit_function('Exponential')
    # set initial values, lower and upper limits for central surface brightness I_0, scale length h;
    # specify that ellipticity is to remain fixed
    disk.I_0.setValue(100.0, [0.0, 500.0])
    disk.h.setValue(25, [10,50])
    disk.PA.setValue(40, [0, 180])
    disk.ell.setValue(0.5, fixed=True)

    model_desc.addFunction(disk)




### Fit a model to the data

#### Specify the data (and optional mask) image

The data image must be a 2D NumPy array (internally, it will be converted to double-precision 
floating point with native byte order, if it isn't already).

You pass in the data image to the previously generated `Imfit` object (`imfitter`) using the latter's
`loadData` method:

    imfitter.loadData(data_im)

You can also specify a mask image, which should be a NumPy integer or float array where values
= 0 indicate *good* pixels, and values > 0 indicate bad pixels that should not be used
in the fit. Alternatively, if the data array is a NumPy MaskedArray, then *its* mask will be used.
(If the data array is a MaskedArray *and* you supply a separate mask image, then the final
mask will be the composition of the data array's mask and the mask image.)

    imfitter.loadData(data_im, mask=mask_im)


#### Image-description parameters, statistical models and fit statistics

When calling the `loadData` method, you can tell the `Imfit` object about the statistical model you want to use: 
what the assumed uncertainties are for the data values, and what "fit statistic" is to be minimized during 
the fitting process.

   * &chi;<sup>2</sup> with data-based errors (default): the default is a standard &chi;<sup>2</sup> approach
   using per-pixel Gaussian errors, with the assumption that the errors (sigma values) can be approximated by
   the square root of the data values.

   * &chi;<sup>2</sup> with model-based errors: Alternately, you can specify *model-based* errors, where the 
   sigma values are the square root of the *model* values (these are automatically recomputed for every 
   iteration of the fitting process).

   * &chi;<sup>2</sup> with user-supplied errors: You can also supply a noise/error array which is the same
   size as the data array and holds per-pixel sigma or variance values precomputed in some fashion (e.g., from
   an image-reduction pipeline).
   
   * Poisson-based ("Poisson Maximum-Likelihood-Ratio"): Finally, you can specify that individual pixel 
   errors come from the model assuming a true Poisson process (rather than the Gaussian approximation to Poisson 
   statistics that's used in the &chi;<sup>2</sup> approaches). This is particularly appropriate when individual 
   pixel values of the data are low.

You can also tell the `Imfit` object useful things about the data values: what A/D gain conversion
was applied, any Gaussian read noise, any constant background value that was previously subtracted from the
data image, etc. (You do not need to do this if you are supplying your own noise/errror array.)

Whatever you chose, you can specify this as part of the call to `loadData`, e.g.

    # default chi^2, assuming an A/D gain of 4.5 e-/ADU and Gaussian read noise with sigma^2 = 0.7 e-
    imfitter.loadData(data_im, gain=4.5, read_noise=0.7)
    
    # chi^2 with model-based errors
    imfitter.loadData(data_im, gain=4.5, read_noise=0.7, use_model_for_errors=True)
    
    # chi^2 with a NumPy variance array `variances_im` (gain and read noise are not needed)
    imfitter.loadData(data_im, error=variances_im, error_type="variance")
    
    # Poisson Maximum-Likelihood-Ratio statistics (read noise is not used in this mode)
    imfitter.loadData(data_im, gain=4.5, use_poisson_mlr=True)
   

#### Performing the Fit

To actually perform the fit, you call the `doFit` method on the `Imfit` object. You can specify which
of the three different minimization algorithms you want to use with the `solver` keyword; the
default is "LM" for the Levenberg-Marquardt minimizer.

   * "LM" = Levenberg-Marquardt (the default): this is a fast, gradient-descent-based minimizer.
   
   * "NM" = Nelder-Mead Simplex: slower, possibly less likely to be trapped in local minimum of
   the fit landscape.
   
   * "DE" = Differential Evolution: genetic-algorithm-based; very slow; probably least likely
   to be trapped in local minima. (This method ignores the initial parameter guesses, instead choosing
   random values selected from within the lower and upper parameter bounds.)
   

E.g.,

    # default Levenberg-Marquardt fit
    result = imfitter.doFit()
    
    # fit using Nelder-Mead simplex
    result = imfitter.doFit(solver='NM')

**Feedback from the fit:** By default, the `Imfit` object is silent during the fitting process.
If you want to see feedback, you can set the `verbose` keyword of the `doFit()` method: `verbose=1`
will print out periodic updates of the current fit statistic (e.g., &chi;<sup>2</sup>;
`verbose=2` will also print the current best-fit parameter values of the model each time it
prints the current fit statistic.

**WARNING:** Currently, there is no way to interrupt a fit once it has started! (Other than
killing the underlying Python process, that is. This may change in the future.)


#### Shortcut: Load data and do the fit in one step

A shortcut is to call the `fit` method on the `Imfit` object. This lets you supply the data image
(along with the optional mask), specify the statistical model (&chi;<sup>2</sup>, etc.) and (optionally)
the minimization algorithm and verbosity, and start the fit all in one go

    result = imfitter.fit(data_im, gain=4.5, use_poisson_mlr=True, solver="NM", verbose=1)


#### Inspecting the results of a fit

The Imfit object returns an instance of the FitResult class, which is closely based on the `OptimizeResult`
class of `scipy.optimize` and is basically a Python dict with attribute access

There are three or four basic things you might want to look at in the FitResult object
when the fit finishes. You can get these things from the FitResult object that's returned
from the `doFit()` method, or by querying the Imfit object; the examples below show each
possibility.

   1. See if the fit actually converged (either `True` or `False`):
   
            result.fitConverged
            imfitter.fitConverged
            
   2. See the value of the final fit statistic, and related values:
   
            result.fitStat   # final chi^2 or PMLR value
            result.reducedFitStat   # reduced version of same
            result.aic   # corresponding Akaike Information Criterion value
            result.bic   # corresponding Bayesian Information Criterion value
            
            imfitter.fitStatistic
            imfitter.reducedFitStatistic
            imfitter.AIC
            imfitter.BIC

   3.A. Get the best-fit parameter values in the form of a 1D NumPy array:
   
            bestfit_parameters = result.params
            bestfit_parameters = imfitter.getRawParameters()

   3.B. Get the 1-sigma uncertainties on the best-fit parameter values in the form of a 1D NumPy array.
   Note that these are only produced if the default Levenberg-Marquardt solver was used, and are
   fairly crude estimates that should be used with caution. A somewhat better approach might be to
   do [bootstrap resampling](./bootstrap.html), or even 
   [use a Markov Chain Monte Carlo code such as "emcee"](./pyimfit_emcee.html).
   
            bestfit_parameters_errs = results.paramErrs
            bestfit_parameters_errs = imfit_fitter.getParameterErrors()

Other things you might be interested in:

   1. Get the best-fitting model image (a 2D NumPy array)

             bestfit_model_im = imfitter.getModelImage()

   2. Get fluxes and magnitudes for the best-fitting model -- note that what is returned is
   a tuple of the total flux/magnitude and a NumPy array of the fluxes/magnitudes for the
   individual components of the model (in the order they are listed in the model):
   
             # get the total flux (counts or whatever the pixel values are) and the
             # individual-component fluxes
             (totalFlux, componentFluxes) = imfitter.getModelFluxes()
             
             # get total and individual-component magnitudes, if you know the zero point
             # for your image (25.72 in this example)
             (totalMag, componentMagnitudes) = imfitter.getModelMagnitudes(zeroPoint=25.72)

Of course, you might also want to inspect the residuals of the fit; since your data image and
the output best-fit model image are both NumPy arrays, this is simple enough:

    residual_im = data_im - bestfit_model_im


### Generate a model image (without fitting)

Sometimes you may want to generate model images without fitting any data. In this case, 
you can call the `getModelImage` method on the `Imfit` object without running the fit.

    model_im = imfitter.getModelImage(shape=image_shape)

where `image_shape` is a 2-element integer tuple defining the image shape in the usual
NumPy fashion (i.e., an image with n_rows and n_colums has shape=(n_columns,n_rows)).

If the `Imfit` object (`imfitter`) already has a data image assigned to it, then the 
output image will have the same dimensions as the data image, and you do not need to
specify the shape.

Note that by default this will generate a model image using the current parameter values
of the model (the initial values, if no fit has been done, or the best-fit values if
a fit *has* been done). You can specify that a *different* set of parameter values
(in the form of a 1-D NumPy array of the correct length) should be used to compute the 
model via the `newParameters` keyword:

    model_im = imfitter.getModelImage(newParameters=parameter_array)
