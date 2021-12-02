# Code for testing fitting and computing of fit-statistic values
# This should probably be seen as a combination of unit tests and regression tests.

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..fitting import Imfit, FitError
from ..descriptions import ModelDescription
from ..pyimfit_lib import FixImage, make_imfit_function

TOLERANCE = 1.0e-6

testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"

image_ic3478 = FixImage(fits.getdata(imageFile))

# construct model from config file; construct new Imfit fitter based on model,;
# add data & do fit
model_desc = ModelDescription.load(configFile)

imfit_fitter1 = Imfit(model_desc)
imfit_fitter1b = Imfit(model_desc)

imfit_fitter1.fit(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
imfit_fitter1b.fit(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)

# before fit
# imfit -c examples/config_exponential_ic3478_256.dat examples/ic3478rss_256.fits --gain=4.725 --readnoise=4.3 \
# --original_sky=130.14 --fitstat-only
initial_params = np.array([129.0,129.0, 18.0,0.2,100.0,25.0])
fitstat_initial_correct = 466713.929408
reduced_fitstat_initial_orrect = 7.122141
aic_initial_correct = 466725.930690
bic_initial_correct = 466780.471537

alternate_params = np.array([128.0,128.0, 25.0,0.3,90.0,20.0])
fitstat_alternate_correct = 650851.591595

# after fit
# imfit -c examples/config_exponential_ic3478_256.dat examples/ic3478rss_256.fits --gain=4.725 --readnoise=4.3 \
# --original_sky=130.14
parameter_vals_correct = np.array([128.8540, 129.1028, 19.72664, 0.2315203, 316.3133, 20.52197])
fitstat_correct = 136470.399329
reduced_fitstat_correct = 2.082564
aic_correct = 136482.400611
bic_correct = 136536.941458



# model with 2 function blocks
imfitTestDir = '/Users/erwin/coding/imfit/tests/'
#configFile2 = imfitTestDir + 'imfit_reference/config_imfit_2gauss_small.dat'
configFile2 = testDataDir + "config_imfit_2gauss_small.dat"
model_desc2 = ModelDescription.load(configFile2)

# imdata2 = FixImage(fits.getdata(imfitTestDir + 'twogaussian_psf+2osamp_noisy.fits'))
# psfImage = FixImage(fits.getdata(imfitTestDir + 'psf_moffat_35.fits'))
imdata2 = FixImage(fits.getdata(testDataDir + 'twogaussian_psf+2osamp_noisy.fits'))
psfImage = FixImage(fits.getdata(testDataDir + 'psf_moffat_35.fits'))

# before fit
initial_params2 = np.array([40.0,40.0,0,0,105,1.5, 15,10,0,0,95,1.5,100])

# after fit (no PSF) -- verified as the correct result
parameter_vals_correct2 = np.array([40.17068,39.91326, 0,0,44.186931,1.560369,
                                   14.762382,10.257345, 0,0,36.0647,1.566958, 100.0])
fitstat_correct2 = 2548.249387
reduced_fitstat_correct2 = 1.022572
aic_correct2 = 2564.307195
bic_correct2 = 2610.841755


# after fit with PSF -- verified as the correct result
parameter_vals_correct2_psf = np.array([40.17035,39.913708, 0,0,109.653515,1.001128,
                                   14.762638,10.257345, 0,0,88.605483,1.01021, 100.0])
fitstat_initial_correct2_psf = 155995.348653
fitstat_correct2_psf = 2490.900766
reduced_fitstat_correct2_psf = 0.999559
aic_correct2_psf = 2506.958574
bic_correct2_psf = 2553.493134


# Data and model for testing fits with masks
imageFile_n3073 = testDataDir + "n3073rss_small.fits"
imageFile_n3073_mask = testDataDir + "n3073rss_small_mask.fits"
image_n3073 = FixImage(fits.getdata(imageFile_n3073))
image_n3073_mask = FixImage(fits.getdata(imageFile_n3073_mask))
image_n3073_mask_bool = image_n3073_mask.astype(bool)
image_n3037_maskedarray = np.ma.array(image_n3073, mask=image_n3073_mask_bool)

configFile_n3073 = testDataDir + "config_n3073.dat"
model_desc_n3073 = ModelDescription.load(configFile_n3073)

# imfit -c tests/imfit_reference/config_imfit_2gauss_small.dat tests/twogaussian_psf+2osamp_noisy.fits --psf tests/psf_moffat_35.fits --mlr

#   POISSON-MLR STATISTIC = 2490.900766    (2492 DOF)
#   INITIAL POISSON-MLR STATISTIC = 155995.348653
#         NPAR = 13
#        NFREE = 8
#      NPEGGED = 0
#      NITER = 13
#       NFEV = 109
#
# Reduced Chi^2 equivalent = 0.999559
# AIC = 2506.958574, BIC = 2553.493134
#
# X0		40.1703 # +/- 0.0064
# Y0		39.9137 # +/- 0.0064
# FUNCTION Gaussian
# PA		      0 # +/- 0
# ell		      0 # +/- 0
# I_0		109.654 # +/- 0.61606
# sigma		1.00113 # +/- 0.0029238
#
# X0		14.7626 # +/- 0.0078
# Y0		10.2573 # +/- 0.0078
# FUNCTION Gaussian
# PA		      0 # +/- 0
# ell		      0 # +/- 0
# I_0		88.6055 # +/- 1.2156
# sigma		1.01021 # +/- 0.008236


def test_get_fitResult_no_fit():
    # Fitting Exponential to 256x256-pixel SDSS r-band image of IC 3478 (no PSF convolution)
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
    # Should raise FitError if fit not actually done
    with pytest.raises(FitError):
        result = imfit_fitter.getFitResult()

def test_get_fitResult():
    result = imfit_fitter1.getFitResult()

    assert result.solverName == "LM"
    assert result.fitConverged == True
    assert result.nIter == 11
    assert result.fitStat == pytest.approx(136470.39932882253)
    assert result.fitStatReduced == pytest.approx(2.0825637010349847)
    assert result.aic == pytest.approx(136482.40061069775)
    assert result.bic == pytest.approx(136536.94145815627)

def test_get_fitResult_from_doFit():
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
    result = imfit_fitter.doFit()

    assert result.solverName == "LM"
    assert result.fitConverged == True
    assert result.nIter == 11
    assert result.fitStat == pytest.approx(136470.39932882253)
    assert result.fitStatReduced == pytest.approx(2.0825637010349847)
    assert result.aic == pytest.approx(136482.40061069775)
    assert result.bic == pytest.approx(136536.94145815627)


def test_fitted_param_values():
    pvals = np.array(imfit_fitter1.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct, rtol=TOLERANCE)

def test_fit_statistics():
    fitstat = imfit_fitter1.fitStatistic
    assert fitstat == pytest.approx(fitstat_correct, TOLERANCE)
    reducedfitstat = imfit_fitter1.reducedFitStatistic
    assert reducedfitstat == pytest.approx(reduced_fitstat_correct, TOLERANCE)
    aic = imfit_fitter1.AIC
    assert aic == pytest.approx(aic_correct, TOLERANCE)
    bic = imfit_fitter1.BIC
    assert bic == pytest.approx(bic_correct, TOLERANCE)


def test_compute_fit_statistics_badinput():
    badArray = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        fitstat = imfit_fitter1.computeFitStatistic(badArray)

def test_compute_fit_statistics_goodinput():
    fitstat = imfit_fitter1.computeFitStatistic(initial_params)
    assert fitstat == pytest.approx(fitstat_initial_correct, TOLERANCE)
    # try it again with different parameters
    fitstat = imfit_fitter1.computeFitStatistic(alternate_params)
    assert fitstat == pytest.approx(fitstat_alternate_correct, TOLERANCE)


# check that calling Imfit.loadData and then Imfit.doFit works as we expect it
def test_new_commands():
    imfit_fitter = Imfit(model_desc)
    imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
    # fit with defautl LM solver
    imfit_fitter.doFit()

    assert imfit_fitter.fitConverged == True
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct, rtol=TOLERANCE)


def test_nodata():
    imfit_fitter = Imfit(model_desc)
    with pytest.raises(Exception):
        imfit_fitter.runBootstrap(10)




def test_2functionblocks_no_PSF():
    imfit_fitter2 = Imfit(model_desc2)
    imfit_fitter2.loadData(imdata2, use_poisson_mlr=True)
    # fit with defautl LM solver
    imfit_fitter2.doFit()

    assert imfit_fitter2.fitConverged == True
    pvals = np.array(imfit_fitter2.getRawParameters())
    # lower tolerance to allow test to pass on Linux
    assert_allclose(pvals, parameter_vals_correct2, rtol=1.0e-5)


def test_2functionblocks_with_PSF():
    imfit_fitter2b = Imfit(model_desc2, psf=psfImage)
    imfit_fitter2b.loadData(imdata2, use_poisson_mlr=True, gain=1000)
    fitstat = imfit_fitter2b.computeFitStatistic(initial_params2)
    assert fitstat == pytest.approx(fitstat_initial_correct2_psf, TOLERANCE)
    # fit with defautl LM solver
    imfit_fitter2b.doFit()

    assert imfit_fitter2b.fitConverged == True
    pvals = np.array(imfit_fitter2b.getRawParameters())
    # lower tolerance to allow test to pass on Linux
    assert_allclose(pvals, parameter_vals_correct2_psf, rtol=1.0e-4)


def test_masking():
    """Test to see that masking works with separate mask image *and* with
    Numpy MaskedArray instance.
    """
    imfit_fitter3a = Imfit(model_desc_n3073)
    imfit_fitter3a.loadData(image_n3073, mask=image_n3073_mask)
    imfit_fitter3a.doFit()

    imfit_fitter3b = Imfit(model_desc_n3073)
    imfit_fitter3b.loadData(image_n3037_maskedarray)
    imfit_fitter3b.doFit()
