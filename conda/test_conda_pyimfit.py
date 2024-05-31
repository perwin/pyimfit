# Simple test script to see if we can a) import pymfit; and b) run a simple fit
# using it.
# Run (within the appropriate conda environment) as
#    $ python test_conda_pyimfit.py

print("Starting test...")
import sys
import pyimfit
from astropy.io import fits

vinfo = sys.version_info
print("Python version {0}.{1}".format(vinfo[0], vinfo[1]))

ff = "/Users/erwin/coding/imfit/examples/"
imageFile = ff + "ic3478rss_256.fits"
configFile = ff + "config_exponential_ic3478_256.dat"

image_data = fits.getdata(imageFile)
model_desc = pyimfit.ModelDescription.load(configFile)
imfit_fitter = pyimfit.Imfit(model_desc)
print("Doing the fit...")
fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
print(fit_result)
print("Done")
