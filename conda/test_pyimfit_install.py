#!/usr/bin/env python3

# Simple test script to see if we can a) import pymfit; and b) run a simple fit
# using it. Uses files in the default Imfit examples/ directory (downloads and
# unpacks this if it isn't in the lcoal directory).

import sys, os, tarfile
import requests
import pyimfit
from astropy.io import fits


# UPDATE THIS TO POING TO WHERE THE IMFIT-EXAMPLES DIRECTORY IS LOCATED
BASE_DIR_ERWIN = "/Users/erwin/coding/imfit/examples/bob/"
EXAMPLES_URL = "https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_examples.tar.gz"

IMAGE_FILE = "ic3478rss_256.fits"
CONFGI_FILE = "config_sersic_ic3478_256.dat"



def main( argv ):
    # By default, we look for a pre-existing Imfit examples/ subdirectory in the current directory
    # If not found, we look in BASE_DIR_ERWIN; if not found there, we download and unpack it
    # from the Imfit webpage at MPE
    if os.path.exists("./examples"):
        baseDir = "./examples/"
    elif os.path.exists(BASE_DIR_ERWIN):
        baseDir = BASE_DIR_ERWIN
    else:
        print("ERROR: Unable to locate pre-existing examples directory.")
        print("Downloading and unpacking examples directory...")
        r = requests.get(EXAMPLES_URL, allow_redirects=True)
        open('examples.tar.gz', 'wb').write(r.content)
        tar = tarfile.open("examples.tar.gz")
        tar.extractall(filter='data')
        tar.close()
        baseDir = "./examples/"
        print("Done.")

    imageFile = baseDir + IMAGE_FILE
    configFile = baseDir + CONFGI_FILE

    print("\nStarting test...")
    vinfo = sys.version_info
    print("Python version {0}.{1}".format(vinfo[0], vinfo[1]))
    print("PyImfit version {0}".format(pyimfit.__version__))

    filesExist = True
    if not os.path.exists(imageFile):
        print("ERROR: Unable to locate image file (path = %s" % imageFile)
        filesExist = False
    if not os.path.exists(configFile):
        print("ERROR: Unable to locate Imfit config file (path = %s" % configFile)
        filesExist = False

    if filesExist:
        image_data = fits.getdata(imageFile)
        model_desc = pyimfit.ModelDescription.load(configFile)
        imfit_fitter = pyimfit.Imfit(model_desc)
        print("Doing the fit...")
        fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
        print(fit_result)
        print("Done!\n")


if __name__ == '__main__':
    main(sys.argv)
