# Code for testing bootstrap mode of pyimfit
#  ** INCOMPLETE **

# $ imfit ic3478rss_256.fits -c config_exponential_ic3478_256.dat --gain=4.725 --readnoise=4.3 --sky=130.14 --seed=10 --bootstrap=5 --save-bootstrap=bootstrap_out.txt

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits

from ..utils import GetBootstrapOutput
from ..fitting import Imfit
from ..descriptions import ModelDescription

TOLERANCE = 1.0e-6

testDataDir = "../data/"
imageFile = testDataDir + "ic3478rss_256.fits"
configFile = testDataDir + "config_exponential_ic3478_256.dat"
bootstrapReferenceFile = testDataDir + "bootstrap_out_ref.txt"

# before fit
initial_params = np.array([129.0,129.0, 18.0,0.2,100.0,25.0])

# after fit
parameter_vals_correct = np.array([128.8540, 129.1028, 19.72664, 0.2315203, 316.3133, 20.52197])
fitstat_correct = 136470.399329
reduced_fitstat_correct = 2.082564
aic_correct = 136482.400611
bic_correct = 136536.941458

columnNames_correct = ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "h_1"]

(refCols, refData) = GetBootstrapOutput(bootstrapReferenceFile)

# do the fit and the bootstrap resampling
image_ic3478 = fits.getdata(imageFile)
modeldesc= ModelDescription.load(configFile)
imfit_fitter = Imfit(modeldesc)
imfit_fitter.loadData(image_ic3478, gain=4.725, read_noise=4.3, original_sky=130.14)
imfit_fitter.doFit()
pvals_fit = np.array(imfit_fitter.getRawParameters())

# bootstrap resampling
output = imfit_fitter.runBootstrap(nIterations=5, seed=10)
columnNames, output = imfit_fitter.runBootstrap(nIterations=5, seed=10, getColumnNames=True)



def test_fit():
    assert_allclose(pvals_fit, parameter_vals_correct, rtol=TOLERANCE)

def test_readin_ref_bootstrap_data():
    assert len(refCols) == 6
    assert refData.shape == (5,6)

def test_new_bootstrap_data():
    assert output.shape == refData.shape
    assert_allclose(output, refData)

def test_column_names():
    assert len(columnNames) == 6
    assert columnNames == columnNames_correct

