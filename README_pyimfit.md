# Pyimfit

This is a Python wrapper for the astronomical image-fitting program Imfit.

Online documentation: [https://pyimfit.readthedocs.io/en/latest/](https://pyimfit.readthedocs.io/en/latest/).


## Sample Usage

The following assumes an interactive Python session (such as an iPython session
or Jupyter notebook):

    from astropy.io import fits
    import pyimfit
    
    imageFile = "<path-to-FITS-file-directory>/ic3478rss_256.fits"
    imfitConfigFile = "<path-to-config-file-directory>/config_exponential_ic3478_256.dat"

    # read in image data, convert to proper double-precisions, little-endian format
    image_data = fits.getdata(imageFile)

    # construct model from config file; construct new Imfit fitter based on model,;
    model_desc = pyimfit.ModelDescription.load(imfitConfigFile)

    # create an Imfit object, using the previously loaded model configuration
    imfit_fitter = pyimfit.Imfit(model_desc)

    # load the image data and image characteristics and do a standard fit
    # (using default chi^2 statistics and Levenberg-Marquardt solver)
    result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
    
    # check the fit and print the resulting best-fit parameter values
    if result.fitConverged is True:
        print("Fit converged: chi^2 = {0}, reduced chi^2 = {1}".format(imfit_fitter.fitStatistic,
            result.reducedFitStat))
        bestfit_params = result.params
        print("Best-fit parameter values:", bestfit_params)
