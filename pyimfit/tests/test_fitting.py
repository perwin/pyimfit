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
from ..descriptions import FunctionSetDescription, ModelDescription, ParameterDescription
from ..pyimfit_lib import FixImage, make_imfit_function



baseDir = "/Users/erwin/coding/pyimfit/pyimfit/tests/"
testDataDir = baseDir + "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"

image_ic3478 = FixImage(fits.getdata(imageFile))

# ModelDescription object for fitting Exponential function to image of IC 3478
model_desc = ModelDescription.load(configFile)

# Simple FlatSky (constant-value) image
flatSkyFunc = make_imfit_function("FlatSky")
funcSet = FunctionSetDescription("sky", ParameterDescription('X0', 1.0), ParameterDescription('Y0', 1.0), [flatSkyFunc])
model_desc_flatsky = ModelDescription([funcSet])



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

    def test_Imfit_get_fluxes_newParameters( self ):
        # Exponential model
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # get fluxes using user-specified parameter vector
        userParams = np.array([129.0,129.0, 20.0, 0.25, 50.0, 20.0])
        (totalFlux, fluxArray) = imfit_fitter.getModelFluxes(userParams)
        totalFlux_correct = 94247.7796076937
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

    def test_Imfit_get_mags_newParameters(self):
        # Exponential model
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)

        # get magnitudes -- first, use built-in zeroPoint property
        imfit_fitter.zeroPoint = 30.0
        userParams = np.array([129.0,129.0, 20.0, 0.25, 50.0, 20.0])
        (totalMag, magsArray) = imfit_fitter.getModelMagnitudes(userParams)
        totalMag_correct = 30 - 2.5 * math.log10(94247.7796076937)
        magsArray_correct = np.array([totalMag_correct])
        assert totalMag == totalMag_correct
        assert magsArray == magsArray_correct

        # get magnitudes -- now, use parameter zero point
        (totalMag, magsArray) = imfit_fitter.getModelMagnitudes(userParams, zeroPoint=20)
        totalMag_correct = 20 - 2.5*math.log10(94247.7796076937)
        magsArray_correct = np.array([totalMag_correct])
        assert totalMag == totalMag_correct
        assert magsArray == magsArray_correct


class TestImfit_ImageGeneration(object):

    def setup_method( self ):
        self.modelDesc = model_desc_flatsky

    def test_Imfit_getImage( self ):
        output_correct = np.zeros(4).reshape((2,2))
        imfit_fitter = Imfit(self.modelDesc)
        outputImage = imfit_fitter.getModelImage((2,2))
        assert_allclose(outputImage, output_correct)

    def test_Imfit_getImage_catchImageSizeChange( self ):
        imfit_fitter = Imfit(self.modelDesc)
        outputImage = imfit_fitter.getModelImage((2,2))
        with pytest.raises(ValueError):
            outputImage = imfit_fitter.getModelImage((4,4))

    def test_Imfit_getImage_newParameters( self ):
        imfit_fitter = Imfit(self.modelDesc)

        output_correct1 = np.zeros(4).reshape((2,2)) + 5.0
        newParams1 = np.array([1.0, 1.0, 5.0])
        outputImage1 = imfit_fitter.getModelImage((2,2), newParameters=newParams1)
        assert_allclose(outputImage1, output_correct1)

        output_correct2 = np.zeros(4).reshape((2,2)) - 15.0
        newParams2 = np.array([1.0, 1.0, -15.0])
        outputImage2 = imfit_fitter.getModelImage(newParameters=newParams2)
        assert_allclose(outputImage2, output_correct2)

