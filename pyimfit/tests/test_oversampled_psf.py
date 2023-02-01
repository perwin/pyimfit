# Code for testing fitting and computing of fit-statistic values using oversampled PSF convolution

# The final test (test_oversampled_psf) implements the following:
# imfit -c tests/imfit_reference/config_imfit_2gauss_small.dat tests/twogaussian_psf+2osamp_noisy.fits \
#       --psf=tests/psf_moffat_35.fits --overpsf tests/psf_moffat_35_oversamp3.fits --overpsf_scale 3 \
#       --overpsf_region 35:45,35:45 --overpsf_region 10:20,5:15 --mlr

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..fitting import Imfit, MakePsfOversampler
from ..descriptions import ModelDescription
from ..pyimfit_lib import PsfOversampling, FixImage

# Note that we use looser tolerances in cases where running on Linux doesn't agree
# with results from running on macOS
TOLERANCE = 1.0e-6

testDataDir = "../data/"
imageFile = testDataDir + "twogaussian_psf+2osamp_noisy.fits"
psfFile = testDataDir + "psf_moffat_35.fits"
osampPsfFile = testDataDir + "psf_moffat_35_oversamp3.fits"
configFile = testDataDir + "config_imfit_2gauss_small.dat"


imdata = fits.getdata(imageFile)
psfImage = fits.getdata(psfFile)
# NOTE: we need to apply FixImage here since the PsfOversampling class *requires* correct
# input image format to instantiate
osampPsfImage_orig = fits.getdata(osampPsfFile)
osampPsfImage_fixed = FixImage(osampPsfImage_orig)

# construct model from config file
model_desc = ModelDescription.load(configFile)


# before fit
initial_params = np.array([40.0,40.0,0,0,105,1.5, 15,10,0,0,95,1.5,100])

# after fit
#    Fit without PSF
parameter_vals_correct1 = np.array([40.17068,39.91326, 0,0,44.186931,1.560369,
                                   14.762382,10.257345, 0,0,36.0647,1.566958, 100.0])
fitstat_correct1 = 2548.249387
reduced_fitstat_correct1 = 1.022572
aic_correct1 = 2564.307195
bic_correct1 = 2610.841755

#    Fit with single PSF
parameter_vals_correct2 = np.array([40.17035,39.913708, 0,0,109.653515,1.001128,
                                   14.762638,10.257345, 0,0,88.605483,1.01021, 100.0])
fitstat_initial_correct2 = 155995.348653
fitstat_correct2 = 2490.900766
reduced_fitstat_correct2 = 0.999559
aic_correct2 = 2506.958574
bic_correct2 = 2553.493134

#    Full fit with PSF *and* oversampled PSFs
parameter_vals_correct3 = np.array([40.503607,40.24703, 0,0,109.57082,1.001424,
                                   15.095914,10.590642, 0,0,88.491562,1.01085, 100.0])
fitstat_initial_correct3 = 169433.517301
fitstat_correct3 = 2491.977492
reduced_fitstat_correct3 = 0.999991
aic_correct3 = 2508.03530
bic_correct3 = 2554.56986



def test_no_psf():
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(imdata, use_poisson_mlr=True)
    # fit with defautl LM solver
    imfit_fitter.doFit()

    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct1, rtol=1.0e-5)


def test_single_psf():
    imfit_fitter = Imfit(model_desc, psf=psfImage, maxThreads=8)
    imfit_fitter.loadData(imdata, use_poisson_mlr=True, gain=1000)
    # fit with defautl LM solver
    imfit_fitter.doFit()

    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct2, rtol=1.0e-4)


def test_oversampled_psf():
    imfit_fitter = Imfit(model_desc, psf=psfImage, maxThreads=8)
    # construct list of PsfOversampling objects
    osample1 = PsfOversampling(osampPsfImage_fixed, 3, "35:45,35:45", 0, 0, True)
    #osample2 = PsfOversampling(osampPsfImage, 3, "10:20,5:15", 0, 0, True)
    osample2 = MakePsfOversampler(osampPsfImage_orig, 3, (10,20,5,15))
    osampleList = [osample1, osample2]
    imfit_fitter.loadData(imdata, use_poisson_mlr=True, gain=1000, psf_oversampling_list=osampleList)
    fitstat = imfit_fitter.computeFitStatistic(initial_params)
    assert fitstat == pytest.approx(fitstat_initial_correct3, TOLERANCE)

    # fit with defautl LM solver
    imfit_fitter.doFit()
    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct3, rtol=1.0e-3)
