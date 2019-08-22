"""
Created on 13/03/2014 by Andre

Modified by PE 2019

The primary purpose of this fie is to hold unit tests for the Imfit class
(in fitting.py), though some of these tests will be effectively duplicated elsewhere.

"""

import pytest

import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from pyimfit import Imfit, SimpleModelDescription, make_imfit_function, gaussian_psf
from ..pyimfit_lib import make_imfit_function


GAIN = 4.725
READ_NOISE = 4.3
ORIGINAL_SKY_IC3478 = 130.14
ORIGINAL_SKY_N3073 = 154.33

testDataDir = "../data/"
imageFile_ic3478 = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"

imageFile_n3073 = testDataDir + "n3073rss_small.fits"
imageFile_n3073_mask = testDataDir + "n3073rss_small_mask.fits"

image_ic3478 = fits.getdata(imageFile_ic3478)
shape_ic3478 = image_ic3478.shape
image_ic3478_consterror = np.ones(shape_ic3478)
image_n3073 = fits.getdata(imageFile_n3073)
image_n3073_mask = fits.getdata(imageFile_n3073_mask)


def create_model():
    model = SimpleModelDescription()
    model.x0.setValue(50, limits=[40,60])
    model.y0.setValue(50, limits=[40,60])
    
    bulge = make_imfit_function('Sersic', label='bulge')
    bulge.I_e.setValue(1.0, limits=[0.5,1.5])
    bulge.r_e.setValue(10, limits=[5,15])
    bulge.n.setValue(4, limits=[3,5])
    bulge.PA.setValue(45, limits=[30,60])
    bulge.ell.setValue(0.5, limits=[0,1])
    
    disk = make_imfit_function('Exponential', label='disk')
    disk.I_0.setValue(0.7, limits=[0.4,0.9])
    disk.h.setValue(15, limits=[10,20])
    disk.PA.setValue(60, limits=[45,90])
    disk.ell.setValue(0.2, limits=[0,0.5])
    
    model.addFunction(bulge)
    model.addFunction(disk)
    return model


def get_model_param_array(model):
    params = []
    for p in model.parameterList():
        params.append(p.value)
    return np.array(params)


model_orig = create_model()


def test_bad_instantiation():
    with pytest.raises(TypeError):
        imfit = Imfit()
    with pytest.raises(ValueError):
        imfit = Imfit(1.0)


# def test_fitting():
#     psf = gaussian_psf(2.5, size=9)
#     this_model = create_model()
#     imfit = Imfit(this_model, psf=psf, quiet=True)
#
#     noise_level = 0.1
#     shape = (100, 100)
#     image = imfit.getModelImage(shape)
#     noise = image * noise_level
#     image += (np.random.random(shape) * noise)
#     mask = np.zeros_like(image, dtype='bool')
#
#     imfit.fit(image, noise, mask)
#     model_fitted = imfit.getModelDescription()
#
#     orig_params = get_model_param_array(model_orig)
#     fitted_params = get_model_param_array(model_fitted)
#
#     assert_allclose(orig_params, fitted_params, rtol=noise_level)



# here, we load the data, then test for pre-fit conditions
def test_loadData_data_only():
    imfit = Imfit(model_orig, quiet=True)
    imfit.loadData(image_ic3478, original_sky=ORIGINAL_SKY_IC3478)
    # no fit performed yet, so...
    with pytest.raises(Exception) as exceptionInfo:
        z = imfit.fitConverged
    assert "Not fitted yet." in str(exceptionInfo.value)
    with pytest.raises(Exception) as exceptionInfo:
        z = imfit.fitError
    assert "Not fitted yet." in str(exceptionInfo.value)
    with pytest.raises(Exception) as exceptionInfo:
        z = imfit.fitTerminated
    assert "Not fitted yet." in str(exceptionInfo.value)

def test_loadData_bad_keyword():
    imfit = Imfit(model_orig, quiet=True)
    with pytest.raises(ValueError) as exceptionInfo:
        imfit.loadData(image_ic3478, original_sky=ORIGINAL_SKY_IC3478, bob=5.0)
    assert "Unknown kwarg: bob" in str(exceptionInfo.value)

def test_loadData_data_with_errorimage():
    imfit = Imfit(model_orig, quiet=True)
    imfit.loadData(image_ic3478, error=image_ic3478_consterror, original_sky=ORIGINAL_SKY_IC3478)

def test_loadData_data_bad_errorimage():
    imfit = Imfit(model_orig, quiet=True)
    with pytest.raises(ValueError) as exceptionInfo:
        imfit.loadData(image_n3073, error=image_ic3478_consterror, original_sky=ORIGINAL_SKY_N3073)
    assert "Error image (256,256) and data image (200,150) shapes do not match." in str(exceptionInfo.value)

def test_loadData_data_with_mask():
    imfit = Imfit(model_orig, quiet=True)
    imfit.loadData(image_n3073, mask=image_n3073_mask, original_sky=ORIGINAL_SKY_N3073)

def test_loadData_data_bad_mask():
    imfit = Imfit(model_orig, quiet=True)
    with pytest.raises(ValueError) as exceptionInfo:
        imfit.loadData(image_n3073, mask=image_ic3478, original_sky=ORIGINAL_SKY_N3073)
    assert "Mask image (256,256) and data image (200,150) shapes do not match." in str(exceptionInfo.value)


