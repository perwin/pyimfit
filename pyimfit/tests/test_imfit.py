"""
Created on 13/03/2014

@author: andre
"""

from pyimfit import Imfit, SimpleModelDescription, make_imfit_function, gaussian_psf
import numpy as np
from numpy.testing import assert_allclose


def create_model():
    model = SimpleModelDescription()
    model.x0.setValue(50, vmin=40, vmax=60)
    model.y0.setValue(50, vmin=40, vmax=60)
    
    bulge = make_imfit_function('Sersic', label='bulge')
    bulge.I_e.setValue(1.0, vmin=0.5, vmax=1.5)
    bulge.r_e.setValue(10, vmin=5, vmax=15)
    bulge.n.setValue(4, vmin=3, vmax=5)
    bulge.PA.setValue(45, vmin=30, vmax=60)
    bulge.ell.setValue(0.5, vmin=0, vmax=1)
    
    disk = make_imfit_function('Exponential', label='disk')
    disk.I_0.setValue(0.7, vmin=0.4, vmax=0.9)
    disk.h.setValue(15, vmin=10, vmax=20)
    disk.PA.setValue(60, vmin=45, vmax=90)
    disk.ell.setValue(0.2, vmin=0, vmax=0.5)
    
    model.addFunction(bulge)
    model.addFunction(disk)
    return model


def get_model_param_array(model):
    params = []
    for p in model.parameterList():
        params.append(p.value)
    return np.array(params)


def test_fitting():
    psf = gaussian_psf(2.5, size=9)
    model_orig = create_model()
    imfit = Imfit(model_orig, psf=psf, quiet=False)

    noise_level = 0.1
    shape = (100, 100)
    image = imfit.getModelImage(shape)
    noise = image * noise_level
    image += (np.random.random(shape) * noise)
    mask = np.zeros_like(image, dtype='bool')
    
    imfit.fit(image, noise, mask)
    model_fitted = imfit.getModelDescription()

    orig_params = get_model_param_array(model_orig)
    fitted_params = get_model_param_array(model_fitted)

    assert_allclose(orig_params, fitted_params, rtol=noise_level)


def test_setparameters():
    psf = gaussian_psf(2.5, size=9)
    model_orig = create_model()
    imfit = Imfit(model_orig, psf=psf, quiet=False)


if __name__ == '__main__':
    test_fitting()
    
