Sample Usage
============

The following assumes an interactive Python session (such as an iPython
session or Jupyter notebook).

Read an image from a FITS file, read a model description from an
Imfit-format text file, and then fit the model to the data:

::

   from astropy.io import fits
   import pyimfit


   # 1. A simple fit to an image (no PSF or mask)

   imageFile = "<path-to-FITS-file-directory>/ic3478rss_256.fits"
   configFile = "<path-to-config-file-directory>/config_exponential_ic3478_256.dat"

   # read in image data
   image_data = fits.getdata(imageFile)

   # construct model (ModelDescription object) from config file
   model_desc = pyimfit.ModelDescription.load(configFile)

   # create an Imfit object, using the previously loaded model configuration
   imfit_fitter = pyimfit.Imfit(model_desc)

   # load the image data and image characteristics and do a standard fit
   # (using default chi^2 statistics and Levenberg-Marquardt solver)
   fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
   print(fit_result)

   # check the fit and print the resulting best-fit parameter values
   if imfit_fitter.fitConverged is True:
       print("Fit converged: chi^2 = {0}, reduced chi^2 = {1}".format(imfit_fitter.fitStatistic,
           imfit_fitter.reducedFitStatistic))
       print("Best-fit parameter values:")
       print(imfit_fitter.getRawParameters())


   # 2. Same basic model and data, but now with PSF convolution and a mask

   # Load PSF image from FITS file, then create Imfit fitter with model + PSF
   psfImageFile = "<path-to-FITS-file-directory>/psf_moffat_35.fits"
   psf_image_data = fits.getdata(psfImageFile)

   imfit_fitter2 = pyimfit.Imfit(model_desc, psf=psf_image_data)

   # load the image data and characteristics, and also a mask image, but don't run the fit yet
   maskImageFile = "<path-to-FITS-file-directory>/mask.fits"
   mask_image_data = fits.getdata(maskImageFile)

   imfit_fitter2.loadData(image_data, mask=mask_image_data, gain=4.725, read_noise=4.3, original_sky=130.14)

   # do the fit, using Nelder-Mead simplex (instead of default Levenberg-Marquardt) as the solver
   fit_result2 = imfit_fitter2.doFit(solver="NM")

You can also programmatically construct a model within Python (rather
than having to read it from a text file):

::

   # define a function for making a simple bulge+disk model, where both components 
   # share the same central coordinate (SimpleModelDescription class)
   def galaxy_model(x0, y0, PA_bulge, ell_bulge, n, I_e, r_e, PA_disk, ell_disk, I_0, h):
       model = pyimfit.SimpleModelDescription()
       # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
       model.x0.setValue(x0, [x0 - 10, x0 + 10])
       model.y0.setValue(y0, [y0 - 10, y0 + 10])
       
       bulge = pyimfit.make_imfit_function('Sersic', label='bulge')
       bulge.PA.setValue(PA_bulge, [0, 180])
       bulge.ell.setValue(ell_bulge, [0, 1])
       bulge.I_e.setValue(I_e, [0, 10*I_e])
       bulge.r_e.setValue(r_e, [0, 10*r_e])
       bulge.n.setValue(n, [0.5, 5])
       
       disk = pyimfit.make_imfit_function('Exponential', label='disk')
       disk.PA.setValue(PA_disk, [0, 180])
       disk.ell.setValue(ell_disk, [0, 1])
       disk.I_0.setValue(I_0, [0, 10*I_0])
       disk.h.setValue(h, [0, 10*h])
       
       model.addFunction(bulge)
       model.addFunction(disk)

       return model


   model_desc = galaxy_model(x0=33, y0=33, PA_bulge=90.0, ell_bulge=0.2, n=4, I_e=1, 
                           r_e=25, pa_disk=90.0, ell_disk=0.5, I_0=1, h=25)

   imfit_fitter = pyimfit.Imfit(model_desc)

   # etc.

Another way to construct the model is by defining it using a set of
nested Python dicts, and passing the parent dict to the
``ModelObject.dict_to_ModelDescription`` function:

::

   # define a function for making a simple bulge+disk model, where both components 
   # share the same central coordinate; this version uses dicts internally

   def galaxy_model(x0, y0, PA_bulge, ell_bulge, n, I_e, r_e, PA_disk, ell_disk, I_0, h):
       # dict describing the bulge (first define the parameter dict, with initial values
       # and lower & upper limits for each parameter)
       p_bulge = {'PA': [PA_bulge, 0, 180], 'ell_bulge': [ell, 0, 1], 'n': [n, 0.5, 5], 
                   'I_e': [I_e, 0.0, 10*I_e], 'r_e': [r_e, 0.0, 10*r_e]}
       bulge_dict = {'name': "Sersic", 'label': "bulge", 'parameters': p_bulge}
       # do the same thing for the disk component
       p_disk = {'PA': [PA_disk, 0, 180], 'ell_disk': [ell, 0, 1], 'I_0': [I_0, 0, 10*I_0],
                   'h': [h, 0.0, 10*h]}
       disk_dict = {'name': "Exponential", 'label': "disk", 'parameters': p_disk}

       # make dict for the function set that combines the bulge and disk components
       # with a single shared center, and then a dict for the whole model
       funcset_dict = {'X0': [x0, x0 - 10, x0 + 10], 'Y0': [y0, y0 - 10, y0 + 10], 
                       'function_list': [bulge_dict, disk_dict]}
       model_dict = {'function_sets': [funcset_dict]}

       model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
       return model


   model_desc = galaxy_model(x0=33, y0=33, PA_bulge=90.0, ell_bulge=0.2, n=4, I_e=1, 
                           r_e=25, pa_disk=90.0, ell_disk=0.5, I_0=1, h=25)

   imfit_fitter = pyimfit.Imfit(model_desc)

   # etc.

You can get a list of PyImfit’s image functions (“Sersic”,
“Exponential”, etc.) from the package-level variable
``pyimfit.imageFunctionList``, and you can get a list of the parameter
names for each image function from ``pyimfit.imageFunctionDict``. Full
descriptions of the individual image functions and their parameters can
be found in `the Imfit manual
(PDF) <https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf>`__
