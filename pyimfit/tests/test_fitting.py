# Unit tests for fitting.py module of pyimfit
# Execute via
#    $ pytest test_fitting.py

import os
import math
import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits

from ..fitting import Imfit
from ..descriptions import ModelDescription
from ..pyimfit_lib import FixImage, make_imfit_function



baseDir = "/Users/erwin/coding/pyimfit/pyimfit/tests/"
testDataDir = baseDir + "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"

image_ic3478 = FixImage(fits.getdata(imageFile))

# ModelDescription object for fitting Exponential function to image of IC 3478
model_desc = ModelDescription.load(configFile)



class TestImfit(object):

    def setup_method( self ):
        self.modelDesc = model_desc

    def test_Imfit_optionsDict_simple( self ):
        imfit_fitter1 = Imfit(self.modelDesc)
        assert imfit_fitter1._modelDescr.optionsDict == {}

    def test_Imfit_optionsDict_updates( self ):
        imfit_fitter2 = Imfit(self.modelDesc)
        imfit_fitter2.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)

        optionsDict_correct1 = {'GAIN': 4.725, 'READNOISE': 4.3, 'ORIGINAL_SKY': 130.14}
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct1

        # update with empty dict should not change anything
        imfit_fitter2._updateModelDescription({})
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct1

        # now test actually updating things
        keywords_new = {'gain': 10.0, 'read_noise': 0.5}
        optionsDict_correct2 = {'GAIN': 10.0, 'READNOISE': 0.5, 'ORIGINAL_SKY': 130.14}
        imfit_fitter2._updateModelDescription(keywords_new)
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct2

        # now test adding an entry
        keywords_new = {'n_combined': 5}
        optionsDict_correct3 = {'GAIN': 10.0, 'READNOISE': 0.5, 'ORIGINAL_SKY': 130.14, 'NCOMBINED': 5}
        imfit_fitter2._updateModelDescription(keywords_new)
        assert imfit_fitter2._modelDescr.optionsDict == optionsDict_correct3

    def test_Imfit_get_fluxes( self ):
        # Fitting Exponential to 256x256-pixel SDSS r-band image of IC 3478 (no PSF convolution)
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # fit with defautl LM solver
        imfit_fitter.doFit()
        # get fluxes
        (totalFlux, fluxArray) = imfit_fitter.getModelFluxes()
        totalFlux_correct = 643232.3971123401
        fluxArray_correct = np.array([totalFlux_correct])
        assert totalFlux == totalFlux_correct
        assert fluxArray == fluxArray_correct

    def test_Imfit_get_mags( self ):
        # Fitting Exponential to 256x256-pixel SDSS r-band image of IC 3478 (no PSF convolution)
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # fit with defautl LM solver
        imfit_fitter.doFit()

        # get magnitudes -- first, use built-in zeroPoint property
        imfit_fitter.zeroPoint = 30.0
        (totalMag, magsArray) = imfit_fitter.getModelMagnitudes()
        totalMag_correct = 30 - 2.5*math.log10(643232.3971123401)
        magsArray_correct = np.array([totalMag_correct])
        assert totalMag == totalMag_correct
        assert magsArray == magsArray_correct

        # get magnitudes -- now, use parameter zero point
        (totalMag, magsArray) = imfit_fitter.getModelMagnitudes(zeroPoint=20)
        totalMag_correct = 20 - 2.5*math.log10(643232.3971123401)
        magsArray_correct = np.array([totalMag_correct])
        assert totalMag == totalMag_correct
        assert magsArray == magsArray_correct
