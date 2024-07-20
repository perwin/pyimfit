#!/usr/bin/env python3

# Simple test script to see if we can a) import pymfit; and b) run a simple, single-Sersic fit
# using it. Uses files in the default Imfit examples/ directory (downloads and unpacks this if
# it isn't in the lcoal directory).

import sys, os, tarfile, argparse
import pyimfit
import requests
from astropy.io import fits


# UPDATE THIS TO POING TO WHERE THE IMFIT-EXAMPLES DIRECTORY IS LOCATED
# (if BASE_DIR_ERWIN does not exist/cannot be found, the files will be downloaded from EXAMPLES_URL)
BASEDIR_PACKAGE = "../pyimfit/data/"
BASE_DIR_ERWIN = "/Users/erwin/coding/imfit/examples/"
EXAMPLES_URL = "https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_examples.tar.gz"

IMAGE_FILE = "ic3478rss_256.fits"
CONFGI_FILE = "config_sersic_ic3478_256.dat"



def main( argv ):
    # parser = argparse.ArgumentParser(
    #     prog='ProgramName',
    #     description='What the program does',
    #     epilog='Text at the bottom of help')

    parser = argparse.ArgumentParser(prog="test_pyimfit_install.py")
    parser.add_argument("--solver", help="Tell Imfit whch solver (LM, NM, DE) to use [default = LM]", default="LM")
    args = parser.parse_args()
    if args.solver not in ["lm", "LM", "nm", "NM", "de", "DE"]:
        print("ERROR: unrecodnized solver name ('{0}')".format(args.solver))
        return -1

    # By default, we look for a pre-existing Imfit examples/ subdirectory in the current directory
    # If not found, we look in BASE_DIR_ERWIN; if not found there, we download and unpack it
    # from the Imfit webpage at MPE
    if os.path.exists("./examples"):
        baseDir = "./examples/"
        print("Using files in local examples/ directory.")
    elif os.path.exists(BASEDIR_PACKAGE):
        baseDir = BASEDIR_PACKAGE
        print("Using files in package examples/ directory.")
    elif os.path.exists(BASE_DIR_ERWIN):
        baseDir = BASE_DIR_ERWIN
        print("Using files in BASE_DIR_ERWIN examples/ directory.")
    else:
        print("WARNING: Unable to locate pre-existing examples directory.")
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
        if args.solver in ["lm", "LM"]:
            print("Using Levenberg-Marquardt solver...")
            fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14)
        elif args.solver in ["nm", "NM"]:
            print("Using Nelder-Mead solver...")
            fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14,
                                          solver="NM")
        elif args.solver in ["de", "DE"]:
            print("Using Differential Evolution solver...")
            fit_result = imfit_fitter.fit(image_data, gain=4.725, read_noise=4.3, original_sky=130.14,
                                          solver="DE")
        print(fit_result)
        print("Done!\n")


if __name__ == '__main__':
    main(sys.argv)
