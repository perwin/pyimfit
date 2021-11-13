PSF Convolution
===============

As part of the modeling process, PyImfit can convolve the generated
model image with a Point-Spread Function (PSF) image to simulate the
effects of telescope optics, atmospheric seeing, etc.

The simplest approach is to use a single PSF image with the same
intrinsic resolution (i.e., with pixels that have the same intrinsic
size as the data and model images) to convolve the entire image.

However, it is also possible to convolve one or more subsections of the
image with higher-resolution (oversampled) PSF images. In this case, the
specified subsections are first computed as higher-resolution model
images, convolved with the higher-resolution PSF image, and then
downsampled to the data image resolution and copied into the appropriate
location in the final model image.

Requirements for PSF images
---------------------------

PSF images should be square, 2D NumPy floating-point arrays, ideally
with the center of the PSF kernel in the center of the image – so square
images with an odd number of pixels are the best approach. They can come
from any source: FITS images of stars, FITS images from telescope
modeling software, NumPy arrays generated in Python, etc.

Basic PSF Convolution
---------------------

To make use of PSF convolution in a model, the PSF image should be
supplied when you instantiate the Imfit object that will be used for
fitting. E.g.,

::

   imfit_fitter = pyimfit.Imfit(model_description, psf=psf_image)

By default, the PSF image will be automatically normalized so that the
sum of its pixel values = 1, so you do not need to normalize it first.
If you *don’t* want the PSF image normalized (as is the case for, e.g.,
some interferomteric PSFs), then set the ``psfNormalization`` keyword to
False:

::

   imfit_fitter = pyimfit.Imfit(model_description, psf=psf_image, psfNormalization=False)

Convolving with Oversampled PSFs
--------------------------------

PyImfit allows you to designate one or more subsections of an image to
be modeled and convolved with a PSF in an “oversampled” mode, using a
PSF image that is oversampled with respect to the data image.

Thus, if you specify a 10x10-pixel subsection of the image and supply a
PSF image that is oversampled by a factor of 5, that part of the model
will be computed in a 50x50-pixel grid (plus appropriate padding around
the edges), convolved with the oversampled PSF image, and finally
5x5-downsampled and copied into the specified 10x10-pixel subsection of
the model image.

To specify an oversampling region, you create a PsfOversampling object;
this is easiest to do with the ``MakePsfOversampler()`` function. You
then place the PsfOversampling object(s) into a list and add the list to
the Imfit object as part of the call to the ``loadData`` method.

For example: assuming that ``oversampledPsf_image`` is a NumPy array for
a PSF that is oversampled by a factor of 5 (i.e., each data pixel
corresponds to a 5x5 array of pixels in the oversampled PSF image) and
you want the oversampled region within the data image to span (x,y) =
[35:45,50:60]:

::

   psfOsamp = pyimfit.MakePsfOversampler(oversampledPsf_image, 5, (35,45, 50,60))
   osampleList = [psfOsamp]
   imfit_fitter.loadData(data_imate, psf_oversampling_list=osampleList, ...)

**Important:** The image section is specified using 1-based
(Fortran/IRAF) indexing, where the lower-left pixel of the image has
coordinates (x,y) = (1,1), and it *includes* the endpoints. Thus,
``pyimfit.MakePsfOversampler(osampPsfImage, 5, (35,45, 50,60))`` will in
Python/NumPy terms apply to the image region [34:45,49:60].
