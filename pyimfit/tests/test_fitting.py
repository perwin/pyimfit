# Unit tests for fitting.py module of pyimfit
# Execute via
#    $ pytest test_fitting.py

import os
import math
import pytest
from collections import OrderedDict
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits

from ..fitting import FitError, FitResult, Imfit
from ..descriptions import FunctionSetDescription, ModelDescription, ParameterDescription
from ..pyimfit_lib import make_imfit_function



#baseDir = "/Users/erwin/coding/pyimfit/pyimfit/tests/"
#testDataDir = baseDir + "../data/"
testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"
true_bestfit_params_ic3478 = np.array([128.8540,129.1028, 19.7266,0.23152,316.313,20.522])
true_errs_ic3478 = np.array([0.0239,0.0293, 0.21721,0.0015524,0.61962,0.034674])

imageFile2 = testDataDir + "n3073rss_small.fits"
maskFile2 = testDataDir + "n3073rss_small_mask.fits"
configFile2 = testDataDir + "config_n3073.dat"

image_ic3478 = fits.getdata(imageFile)
image_n3073 = fits.getdata(imageFile2)
mask_n3073 = fits.getdata(maskFile2)

# ModelDescription object for fitting Exponential function to image of IC 3478
model_desc = ModelDescription.load(configFile)
# ModelDescription object for fitting Sersic + Exponential function to image of NGC 3073
model_desc2 = ModelDescription.load(configFile2)

# Simple FlatSky (constant-value) image
flatSkyFunc = make_imfit_function("FlatSky")
funcSet = FunctionSetDescription("sky", ParameterDescription('X0', 1.0), ParameterDescription('Y0', 1.0), [flatSkyFunc])
model_desc_flatsky = ModelDescription([funcSet])



class TestImfit(object):

    def setup_method( self ):
        self.modelDesc = model_desc

    def test_Imfit_instantiate_from_dict(self):
        # Exponential model for fitting IC 3478
        p = {'PA': [18.0, 0.0, 90.0], 'ell': [0.2, 0.0, 1.0], 'I_0': [100.0, 0.0, 500.0],
             'h': [25.0, 0.0, 100.0]}
        fDict = {'name': "Exponential", 'label': '', 'parameters': p}
        fsetDict = {'X0': [129.0, 125.0, 135.0], 'Y0': [129.0, 125.0, 135.0], 'function_list': [fDict]}
        # options_dict = OrderedDict()
        # options_dict.update( {"GAIN": 4.725, "READNOISE": 4.3, "ORIGINAL_SKY": 130.14} )
        model_dict_input = {"function_sets": [fsetDict]}

        imfit_fitter = Imfit(model_dict_input)
        imfit_fitter_correct = Imfit(self.modelDesc)
        # check for equality of ModelDescription components
        assert imfit_fitter._modelDescr.getModelAsDict() == imfit_fitter_correct._modelDescr.getModelAsDict()


    def test_Imfit_optionsDict_simple( self ):
        imfit_fitter1 = Imfit(self.modelDesc)
        assert imfit_fitter1._modelDescr.optionsDict == {}

    def test_Imfit_columnNames_one_function( self ):
        imfit_fitter1 = Imfit(self.modelDesc)
        columnNames_correct = ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "h_1"]
        columnNames = imfit_fitter1.numberedParameterNames
        assert imfit_fitter1.numberedParameterNames == columnNames_correct

    def test_Imfit_columnNames_two_functions( self ):
        imfit_fitter2 = Imfit(model_desc2)
        columnNames_correct = ['X0_1', 'Y0_1', 'PA_1', 'ell_1', 'n_1', 'I_e_1', 'r_e_1',
                                'PA_2', 'ell_2', 'I_0_2', 'h_2']
        assert imfit_fitter2.numberedParameterNames == columnNames_correct

    def test_Imfit_bad_fit_start( self ):
        """Test that we catch error of trying to start a fit without any data"""
        imfit_fitter1 = Imfit(self.modelDesc)
        with pytest.raises(Exception):
            imfit_fitter1.doFit()

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

    def test_Imfit_simple_fit( self ):
        # Fitting Exponential to 256x256-pixel SDSS r-band image of IC 3478 (no PSF convolution)
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # fit with defautl LM solver
        imfit_fitter.doFit()
        assert imfit_fitter.fitTerminated == False
        assert imfit_fitter.fitConverged == True
        assert imfit_fitter.fitStatistic == approx(136470.399329, rel=1e-10)
        assert imfit_fitter.AIC == approx(136482.400611, rel=1e-10)
        bestfit_params = imfit_fitter.getRawParameters()
        assert_allclose(bestfit_params, true_bestfit_params_ic3478, rtol=1e-5)
        bestfit_errs = imfit_fitter.getParameterErrors()
        assert_allclose(bestfit_errs, true_errs_ic3478, rtol=1e-2)

    def test_Imfit_simple_fit_altftol( self ):
        # Fitting Exponential to 256x256-pixel SDSS r-band image of IC 3478 (no PSF convolution)
        # this time we explicitly use ftol=1e-6 (instead of default 1e-8)
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # fit with defautl LM solver
        imfit_fitter.doFit(ftol=1e-6)
        assert imfit_fitter.fitTerminated == False
        assert imfit_fitter.fitConverged == True
        assert imfit_fitter.fitStatistic == approx(136470.402986, rel=1e-10)
        assert imfit_fitter.AIC == approx(136482.404268, rel=1e-10)

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
        assert_allclose(totalFlux, totalFlux_correct, rtol=1.0e-9)
        assert_allclose(fluxArray, fluxArray_correct, rtol=1.0e-9)

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
        print(self.modelDesc)
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
        assert_allclose(totalMag, totalMag_correct, rtol=1.0e-10)
        assert_allclose(magsArray, magsArray_correct, rtol=1.0e-10)

        # get magnitudes -- now, use parameter zero point
        (totalMag, magsArray) = imfit_fitter.getModelMagnitudes(zeroPoint=20)
        totalMag_correct = 20 - 2.5*math.log10(643232.3971123401)
        magsArray_correct = np.array([totalMag_correct])
        assert_allclose(totalMag, totalMag_correct, rtol=1.0e-10)
        assert_allclose(magsArray, magsArray_correct, rtol=1.0e-10)

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

    def test_Imfit_getRawParameters(self):
        # Exponential model for fitting IC 3478
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        # should return initial parameters from configFile ["config_exponential_ic3478_256.dat"]
        correctParams = np.array([129.0,129.0, 18.0, 0.2, 100.0, 25.0])
        outputParams = imfit_fitter.getRawParameters()
        assert_allclose(outputParams, correctParams, rtol=1.0e-9)

    def test_Imfit_getModelDict(self):
        # Exponential model for fitting IC 3478
        p = {'PA': [18.0, 0.0, 90.0], 'ell': [0.2, 0.0, 1.0], 'I_0': [100.0, 0.0, 500.0],
             'h': [25.0, 0.0, 100.0]}
        fDict = {'name': "Exponential", 'label': '', 'parameters': p}
        fsetDict = {'X0': [129.0, 125.0, 135.0], 'Y0': [129.0, 125.0, 135.0], 'function_list': [fDict]}
        options_dict = OrderedDict()
        options_dict.update( {"GAIN": 4.725, "READNOISE": 4.3, "ORIGINAL_SKY": 130.14} )
        model_dict_correct = {"function_sets": [fsetDict], "options": options_dict}

        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
        model_dict = imfit_fitter.getModelAsDict()
        assert model_dict == model_dict_correct

    def test_Imfit_catch_bad_parameters(self):
        """Check that we get ValueError exceptions when passing new-parameter-vector of wrong size
        """
        # Exponential model
        imfit_fitter = Imfit(self.modelDesc)
        imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)

        badParams = np.array([129.0,129.0, 20.0])   # not enough parameter values!
        # computeFitStatistic
        with pytest.raises(ValueError):
            fitstat = imfit_fitter.computeFitStatistic(badParams)
        # getModelImage
        with pytest.raises(ValueError):
            image = imfit_fitter.getModelImage(newParameters=badParams)
        # getModelFluxes
        with pytest.raises(ValueError):
            (totalFlux, fluxArray) = imfit_fitter.getModelFluxes(badParams)
        # getModelMagnitudes
        with pytest.raises(ValueError):
            (totalMag, magsArray) = imfit_fitter.getModelMagnitudes(badParams, zeroPoint=20)



class TestImfit_MultiComponent(object):

    def setup_method( self ):
        self.modelDesc2 = model_desc2

    def test_Imfit_setup( self ):
        imfit_fitter2 = Imfit(self.modelDesc2)
        model_desc2 = imfit_fitter2.getModelDescription()
        assert model_desc2 == self.modelDesc2

    def test_Imfit_multiComponent_get_fluxes_and_mags( self ):
        # Fitting Sersic + Exponential to 150x200-pixel SDSS r-band image of NGC 3073 (no PSF convolution)
        imfit_fitter2 = Imfit(self.modelDesc2)
        imfit_fitter2.loadData(image_n3073, mask=mask_n3073)
        imfit_fitter2.doFit()
        # get fluxes
        (totalFlux, fluxArray) = imfit_fitter2.getModelFluxes()
        totalFlux_correct = 1291846.609307
        fluxArray_correct = np.array([453777.572255, 838069.037053])
        assert_allclose(totalFlux, totalFlux_correct, rtol=1.0e-8)
        assert_allclose(fluxArray, fluxArray_correct, rtol=1.0e-7)

        # get magnitudes -- use parameter zero point
        (totalMag, magsArray) = imfit_fitter2.getModelMagnitudes(zeroPoint=20)
        totalMag_correct = 20 - 2.5*math.log10(totalFlux_correct)
        magsArray_correct = 20 - 2.5*np.log10(fluxArray_correct)
        assert_allclose(totalMag, totalMag_correct, rtol=1.0e-8)
        assert_allclose(magsArray, magsArray_correct, rtol=1.0e-7)


class TestImfit_ImageGeneration(object):

    def setup_method( self ):
        self.modelDesc = model_desc_flatsky

    def test_Imfit_getImage( self ):
        output_correct = np.zeros(4).reshape((2,2))
        print(self.modelDesc)
        imfit_fitter = Imfit(self.modelDesc)
        print(imfit_fitter)
        print("test_Imfit_getImage: about to call getModelImage...")
        outputImage = imfit_fitter.getModelImage((2,2))
        print(outputImage)
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

