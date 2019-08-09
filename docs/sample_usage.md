# Sample Usage

The following assumes an interactive Python session (such as an iPython session
or Jupyter notebook).

Read an image from a FITS file, read a model description from an Imfit-format text file,
and then fit the model to the data:

    from astropy.io import fits
    import pyimfit
    
    
    # 1. A simple fit to an image (no PSF or mask)
    
    imageFile = "<path-to-FITS-file-directory>/ic3478rss_256.fits"
    imfitConfigFile = "<path-to-config-file-directory>/config_exponential_ic3478_256.dat"

    # read in image data
    image_data = fits.getdata(imageFile)

    # construct model from config file
    model_desc = pyimfit.ModelDescription.load(configFile)

    # create an Imfit object, using the previously loaded model configuration
    imfit_fitter = pyimfit.Imfit(model_desc)

    # load the image data and image characteristics and do a standard fit
    # (using default chi^2 statistics and Levenberg-Marquardt solver)
    imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
    
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
    imfit_fitter2.doFit(solver="NM")


You can also programmatically construct a model within Python (rather than having
to read it from a text file):

    # define a function for a simple bulge+disk model, where both components share the same
    # central coordinate (SimpleModelDescription class)
    def galaxy_model(x0, y0, PA, ell, I_e, r_e, n, I_0, h):
        model = pyimfit.SimpleModelDescription()
        # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
        model.x0.setValue(x0, [x0 - 10, x0 + 10])
        model.y0.setValue(y0, [y0 - 10, y0 + 10])
        
        bulge = pyimfit.make_imfit_function('Sersic', label='bulge')
        bulge.I_e.setValue(I_e, [1e-33, 10*I_e])
        bulge.r_e.setValue(r_e, [1e-33, 10*r_e])
        bulge.n.setValue(n, [0.5, 5])
        bulge.PA.setValue(PA, [0, 180])
        bulge.ell.setValue(ell, [0, 1])
        
        disk = pyimfit.make_imfit_function('Exponential', label='disk')
        disk.I_0.setValue(I_0, [1e-33, 10*I_0])
        disk.h.setValue(h, [1e-33, 10*h])
        disk.PA.setValue(PA, [0, 180])
        disk.ell.setValue(ell, [0, 1])
        
        model.addFunction(bulge)
        model.addFunction(disk)
    
        return model
    
    
    model_desc = galaxy_model(x0=33, y0=33, PA=90.0, ell=0.5, I_e=1, 
                            r_e=25, n=4, I_0=1, h=25)

    imfit_fitter = pyimfit.Imfit(model_desc)

    # etc.

You can get a list of PyImfit's image functions ("Sersic", "Exponential", etc.) from the package-level 
variable `pyimfit.imageFunctionList`, and you can get a list of the parameter names for each image 
function from `pyimfit.imageFunctionDict`. Full descriptions of the individual image functions and
their parameters can be found in [the Imfit manual (PDF)](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf)
