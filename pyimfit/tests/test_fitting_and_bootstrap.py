# Code for testing bootstrap mode of pyimfit

# $ imfit ic3478rss_256.fits -c config_exponential_ic3478_256.dat --gain=4.725 --readnoise=4.3 --sky=130.14 --seed=10 --bootstrap=10 --save-bootstrap=bootstrap_out.txt

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..utils import GetBootstrapOutput
from ..fitting import Imfit
from ..model import ModelDescription
from ..pyimfit_lib import FixImage, function_description

testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"
bootstrapReferenceFile = testDataDir + "bootstrap_out_ref.txt"

(refCols, refData) = GetBootstrapOutput(bootstrapReferenceFile)

image_ic3478 = FixImage(fits.getdata(imageFile))


# construct model from config file; construct new Imfit fitter based on model,;
# add data & do fit
model_desc = ModelDescription.load(configFile)

imfit_fitter = Imfit(model_desc)

imfit_fitter.fit(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)


parameter_vals_correct = np.array([128.8540, 129.1028, 19.72664, 0.2315203, 316.3133, 20.52197])


def test_readin_refdata():
    assert len(refCols) == 6
    assert refData.shape == (10,6)

def test_fitted_param_values():
    pvals = np.array(imfit_fitter.getRawParameters())
    assert_allclose(pvals, parameter_vals_correct, rtol=1e-6)

# generate bootstrap output from pyimfit
