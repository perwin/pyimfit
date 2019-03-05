Overview of PyImfit
===================

PyImfit is a Python wrapper around the (C++-based) image-fitting program
`Imfit <https://www.mpe.mpg.de/~erwin/code/imfit>`__ (`Github
site <https://github.com/perwin/imfit>`__).

**Terminology:**

**Imfit** (boldface) refers to the C++ program (and the C++-based
library that PyImfit is built upon).

PyImfit is the general name for this Python package; ``pyimfit`` is the
official Python name (e.g., ``import pyimfit``).

Finally, ``Imfit`` refers to the ``pyimfit.Imfit`` class, which does
most of the work.

For Those Already Familiar with Imfit
-------------------------------------

If you've already used the command-line version of **Imfit**, here are
the essential things to know:

-  PyImfit operates on 2D numpy arrays instead of FITS files; to use a
   FITS file, read it into Python via, e.g.,
   `astropy.io.fits <http://docs.astropy.org/en/stable/io/fits/>`__.

-  Models (and initial parameter values and parameter limits for a fit)
   are specified via the ModelDescription class. The utility function
   ``parse_config_file`` will read a standard **Imfit** configuration
   file and return an instance of that class with the model
   specification. (Or you can build up a ModelDescription instance by
   programmatically specifying components from within Python.)

-  Fitting is done by instantiating an ``Imfit`` object with a
   ModelDescription object as input, then adding a 2D numpy array as the
   data to be fit (along with, optionally, mask and error images, image
   A/D gain, etc.) with the ``loadData`` method, and then calling the
   ``doFit`` method (along with the minimization algorithm to use). Or
   just call the ``fit`` method and supply the data image, etc., as part
   of its input.

-  Once the fit is finished, information about the fit (final χ2 value,
   best-fit paremeter values, etc.) and the best-fitting model image can
   be obtained by querying properties and methods of the ``Imfit``
   object.

See `Sample Usage <./sample_usage.html>`__ for a simple example of how
this works.

The Basics
----------

There are three basic things you can do with PyImfit:

1. Generate model images

2. Fit models to a pre-existing (data) image

3. Generate χ2 or likelihood values from a comparison of model and data
   (e.g., for use with an alternate fitting approach, MCMC analysis,
   etc.)

In **Imfit** (and PyImfit), a "model" consists of one or more *image
functions* from a library built into **Imfit**, sharing one or more
common locations within an image and added together to form a summed
model image. Optionally, the summed model image can then be convolved
with a user-supplied Point-Spread Function (PSF) image to simulate the
effects of seeing and telescope optics. (For greater accuracy,
subsections of the image can be oversampled on a finer pixel scale and
convolved with a correspondingly oversampled PSF image or images; these
subsection are then downsampled back to the native image pixel scale.)

Specify the model (and its parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is specified by an instance of the ModelDescription class.

For the command-line program, this is done via a "configuration" text
file, which has a specific format described in `the Imfit manual
(PDF) <http://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf>`__,
or in `this page of the online
docs <https://imfit.readthedocs.io/en/latest/config_file_format.html>`__.

If you have a configuration file, you can load it via the convenience
function ``parse_config_file``

::

    model_desc = pyimfit.parse_config_file(configFilePath)

where ``configFilePath`` is a string specifying the path to the
configuration file.

You can also construct a ModelDescription instance programmatically from
within Python; see `Sample Usage <./sample_usage.html>`__ for a simpe
example.

Once you have a ModelDescription object describing the model, you can
create an instance of the ``Imfit`` class based on the model;
optionally, if you want the model to be convolved with a PSF, you can
also supply the PSF image (in the form of a 2D numpy array):

::

    imfitter = pyimfit.Imfit(model_desc)
    imfitter = pyimfit.Imfit(model_desc, psfImage)

Fit a model to the data
~~~~~~~~~~~~~~~~~~~~~~~

Specify the data (and optional mask) image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data image is a 2D numpy array; it should be in double-precision
floating point with native byte order. Any image you create
programmatically within Python will almost certainly be in this format
already; if you load an image from a FITS file, then it's a good idea to
run it through the convenience function ``FixImage``, which will ensure
the image is in the right format:

::

    fits_data_im = fits.getdata(pathToImage)
    data_im = pyimfit.FixImage(fits_data_im)

You then pass in the data image to the previously generated ``Imfit``
object (\`imfitter'), along with an (optional) mask image:

::

    imfitter.loadData(data_im)

You can also specify a mask image, which should be a numpy integer or
float array where values [UPDATE WITH BETTER DESCRIPTION OF MASK
FORMATS, INCLUDING NUMPY MASKED ARRAY]

::

    imfitter.loadData(data_im, mask=mask_im)

Image-description parameters, statistical models and fit statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When calling the ``loadData`` method, you can tell the ``Imfit`` object
about the statistical model you want to use: what the assumed
uncertainties are for the data values, and what "fit statistic" is to be
minimized during the fitting process.

-  χ2 with data-based errors (default): the default is a standard χ2
   approach using per-pixel Gaussian errors, with the assumption that
   the errors (sigma values) are the square root of the data values.

-  χ2 with model-based errors: Alternately, you can specify
   *model-based* errors, where the sigma values are the square root of
   the *model* values (these are automatically recomputed for every
   iteration of the fitting process).

-  χ2 with user-supplied errors: You can also supply a noise/error array
   which is the same size as the data array and holds per-pixel sigma or
   variance values precomputed in some fashion (e.g., from an
   image-reduction pipeline).

-  Poisson-based ("Poisson Maximum-Likelihood-Ratio"): Finally, you can
   specify that individual pixel errors come from the model using a true
   Poisson model (rather than the Gaussian approximation to Poisson
   statistics that's used in the χ2 approaches). This is particularly
   apt when individual data pixel values are low.

You can also tell the ``Imfit`` object useful things about the data
values: what A/D gain conversion was applied, any Gaussian read noise,
any constant background value that was previously subtracted from the
data image, etc.

Whatever you chose, you can specify this as part of the call to
``loadData``, e.g.

::

    # default chi^2, assuming an A/D gain of 4.5 e-/ADU and Gaussian read noise with sigma^2 = 0.7 e-
    imfitter.loadData(data_im, gain=4.5, read_noise=0.7)

    # chi^2 with model-based errors
    imfitter.loadData(data_im, gain=4.5, read_noise=0.7, use_model_for_errors=True)

    # chi^2 with a variance array (assumed to already include read-noise contributions)
    imfitter.loadData(data_im, gain=4.5, error=variances, error_type="variance")

    # Poisson Maximum-Likelihood-Ratio statistics (read noise is not used in this mode)
    imfitter.loadData(data_im, gain=4.5, use_poisson_mlr=True)

Performing the Fit
^^^^^^^^^^^^^^^^^^

To actually perform the fit, you call the ``doFit`` method on the
``Imfit`` object. You can specify which of the three different
minimization algorithms you want to use with the ``solver`` keyword; the
default is "LM" for the Levenberg-Marquardt minimizer.

-  "LM" = Levenberg-Marquardt (the default): this is a fast,
   gradient-descent based minimizer.

-  "NM" = Nelder-Mead Simplex

-  "DE" = Differential Evolution

Shortcut: Load data and do the fit in one step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A shortcut is to call the ``fit`` method on the ``Imfit`` object. This
lets you supply the data image (along with the optional mask), specify
the statistical model (χ2, etc.) and the minimization algorithm, and
start the fit all in one go

::

    imfitter.fit(data_im, gain=4.5, use_poisson_mlr=True, solver="NM")

Inspecting the results of a fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three or four basic things you might want to look at when the
fit finishes:

1. See if the fit actually converged (this is a property of the
   ``Imfit`` object):

   ::

           imfitter.fitConverged

2. See the value of the final fit statistic, and related values (these
   are all properties of the ``Imfit`` object)

   ::

           imfitter.fitStatistic   # final chi^2 or PMLR value
           imfitter.reducedFitStatistic   # reduced version of same
           imfitter.AIC   # corresponding Akaike Information Criterion value
           imfitter.BIC   # corresponding Bayesian Information Criterion value

3. Get the best-fit parameter values

   ::

           # get the best-fit parameter values in the form of a 1D numpy array
           bestfit_parameters = imfit_fitter.getRawParameters()

4. Get the best-fitting model image

   ::

            # get the best-fit model image as a 2D numpy array
            bestfit_model_im = imfitter.getModelImage()

Of course, you might also want to inspect the residuals of the fit;
since your data image and the output best-fit model image are both numpy
arrays, this is simple enough:

::

    residual_im = data_im - bestfit_model_im

Generate a model image (without fitting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may just want to generate model images without fitting any
data. In this case, you can call the ``getModelImage`` method on the
``Imfit`` object without running the fit.

::

    model_im = imfitter.getModelImage(shape=image_shape)

where ``image_shape`` is a 2-element integer tuple defining the image
shape in the usual numpy fashion (i.e., an image with n\_rows and
n\_colums has shape=(n\_columns,n\_rows)).

If the ``Imfit`` object (``imfitter``) already has a data image assigned
to it, then the output image will have the same dimensions as the data
image, and you do not need to specify the shape.
