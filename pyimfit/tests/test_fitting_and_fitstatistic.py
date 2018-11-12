# Code for testing fitting and computing of fit-statistic values

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..fitting import Imfit
from ..model import ModelDescription
from ..pyimfit_lib import FixImage, function_description

TOLERANCE = 1.0e-6

testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"


image_ic3478 = FixImage(fits.getdata(imageFile))


# construct model from config file; construct new Imfit fitter based on model,;
# add data & do fit
model_desc = ModelDescription.load(configFile)

imfit_fitter1 = Imfit(model_desc)

imfit_fitter1.fit(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)


# before fit
initial_params = np.array([129.0,129.0, 18.0,0.2,100.0,25.0])
fitstat_initial_correct = 466713.929408
reduced_fitstat_initial_orrect = 7.122141
aic_initial_correct = 466725.930690
bic_initial_correct = 466780.471537


# after fit
parameter_vals_correct = np.array([128.8540, 129.1028, 19.72664, 0.2315203, 316.3133, 20.52197])
fitstat_correct = 136470.399329
reduced_fitstat_correct = 2.082564
aic_correct = 136482.400611
bic_correct = 136536.941458


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
