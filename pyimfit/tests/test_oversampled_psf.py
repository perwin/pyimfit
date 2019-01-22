# Code for testing fitting and computing of fit-statistic values

#   $IMFIT -c tests/imfit_reference/config_imfit_2gauss_small.dat \
#      tests/twogaussian_psf+2osamp_noisy.fits --psf=tests/psf_moffat_35.fits \
#      --overpsf tests/psf_moffat_35_oversamp3.fits --overpsf_scale 3 --overpsf_region 35:45,35:45 \
#      --overpsf_region 10:20,5:15

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..fitting import Imfit
from ..descriptions import ModelDescription
from ..pyimfit_lib import FixImage, function_description

TOLERANCE = 1.0e-6

testDataDir = "../data/"
imageFile = testDataDir + "twogaussian_psf+2osamp_noisy.fits"
psfFile = testDataDir + "psf_moffat_35.fits"
osampPsfFile = testDataDir + "psf_moffat_35_oversamp3.fits"
configFile = testDataDir + "config_imfit_2gauss_small.dat"


imdata = FixImage(fits.getdata(imageFile))
psfImage = FixImage(fits.getdata(psfFile))
osampPsfImage = FixImage(fits.getdata(osampPsfFile))

# construct model from config file
model_desc = ModelDescription.load(configFile)


# before fit
initial_params = np.array([40.0,40.0,0,0,105,1.5, 15,10,0,0,95,1.5,100])

# after fit -- verified as the correct result
#    Fit without PSF
parameter_vals_correct1 = np.array([40.17071,39.913263, 0,0,44.186443,1.560288,
                                    14.762384,10.257356, 0,0,36.063185,1.566889, 100.0])
fitstat_correct1 = 2548.249387
reduced_fitstat_correct1 = 1.022572
aic_correct1 = 2564.307195
bic_correct1 = 2610.841755

#    Fit with PSF (no oversampled PSFs)
parameter_vals_correct2 = np.array([40.1703,39.9137, 0,0,109.668,1.00101,
                                    14.7626,10.2573, 0,0,88.6079,1.01015, 100.0])
fitstat_correct2 = 2491.627432
reduced_fitstat_correct2 = 0.999850
aic_correct2 = 2507.685240
bic_correct2 = 2554.219800

#    Full fit with PSF *and* oversampled PSFs
parameter_vals_correct3 = np.array([40.5036,40.2470, 0,0,109.583,1.00132,
                                   15.0959,10.5906, 0,0,88.5157,1.01064, 100.0])
fitstat_correct3 = 2491.977492
reduced_fitstat_correct3 = 0.999991
aic_correct3 = 2508.03530
bic_correct3 = 2554.56986




def test_no_psf():
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(imdata)
    # fit with defautl LM solver
    imfit_fitter.doFit()

    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct1, rtol=TOLERANCE)

def test_single_psf():
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(imdata)
    # fit with defautl LM solver
    imfit_fitter.doFit()

    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct2, rtol=TOLERANCE)
