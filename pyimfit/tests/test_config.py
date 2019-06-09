# Code for testing config.py module of pyimfit
# Execute via
#    $ pytest test_config.py

import copy
import pytest

from ..descriptions import ParameterDescription, FunctionDescription, FunctionSetDescription
from ..descriptions import ModelDescription
from ..config import read_parameter, read_function, read_function_set, read_options
from ..config import parse_config


# inputs and reference outputs for read_parameter
testLine_bad1 = "X0   "
testLine_bad2 = "X0\t100.1   500,1"
testLine_bad3 = "X0\t100.1   500"
testLine_bad4 = "X0\t100.1,500"

testLine_good1 = "X0  100.1"
testLine_good2 = "X0\t100.1"
testLine_good3 = "X0\t100.1   # this is a comment"
testLine_good4 = "X0\t100.1   fixed"
testLine_good5 = "X0\t100.1   1.0,500"

pdesc_ref_correct123 = ParameterDescription("X0", 100.1)
pdesc_ref_correct4 = ParameterDescription("X0", 100.1, fixed=True)
pdesc_ref_correct5 = ParameterDescription("X0", 100.1, 1.0, 500.0)


# inputs and reference outputs for read_function
testFunctionLine_bad1 = ["bob"]
testFunctionLine_bad2 = ["FUNCTION"]

testFunctionLines_good1 = ["FUNCTION Gaussian", "PA  10", "ell  0.5", "I_0  100", "sigma  10"]
testFunctionLines_good2 = ["FUNCTION Exponential",  "PA  20\tfixed", "ell  0.2 0.1,0.4",
                           "I_0  100  0,1e6", "h  100   1,400"]

fdesc_correct1 = FunctionDescription("Gaussian")
for line in testFunctionLines_good1[1:]:
    pdesc = read_parameter(line)
    fdesc_correct1.addParameter(pdesc)
fdesc_correct2 = FunctionDescription("Exponential")
for line in testFunctionLines_good2[1:]:
    pdesc = read_parameter(line)
    fdesc_correct2.addParameter(pdesc)


# inputs and reference outputs for read_function_set
testFunctionBlockLines_good1 = ["X0  100", "Y0  200", "FUNCTION Gaussian", "PA  10",
                                "ell  0.5", "I_0  100", "sigma  10"]
testFunctionBlockLines_good2 = ["X0  100\tfixed", "Y0  200\tfixed",
                                "FUNCTION Gaussian", "PA  10\tfixed", "ell  0.5 0.1,0.6",
                                "I_0  100  0,1e6", "sigma  10   fixed",
                                "FUNCTION Exponential",  "PA  20\tfixed", "ell  0.2 0.1,0.4",
                                "I_0  100  0,1e6", "h  100   1,400"]


fsetdesc_correct1 = FunctionSetDescription("function_block_1",
                                           x0param=ParameterDescription('X0', 100.0),
                                           y0param=ParameterDescription('Y0', 200.0))
fsetdesc_correct1.addFunction(fdesc_correct1)

fdesc_fbloc2_correct1 = FunctionDescription("Gaussian")
for line in testFunctionBlockLines_good2[3:7]:
    pdesc = read_parameter(line)
    fdesc_fbloc2_correct1.addParameter(pdesc)
fsetdesc_correct2 = FunctionSetDescription("function_block_2",
                                           x0param=ParameterDescription('X0', 100.0, fixed=True),
                                           y0param=ParameterDescription('Y0', 200.0, fixed=True))
fsetdesc_correct2.addFunction(fdesc_fbloc2_correct1)
fsetdesc_correct2.addFunction(fdesc_correct2)


# inputs and reference outputs for parse_config
testConfigLines_good1 = ["GAIN 4.5", "READNOISE\t0.5", "X0  100", "Y0  200",
                         "FUNCTION Gaussian", "PA  10",
                         "ell  0.5", "I_0  100", "sigma  10"]
testConfigLines_good2 = ["GAIN 4.5", "READNOISE\t0.5", "X0  100\tfixed", "Y0  200\tfixed",
                         "FUNCTION Gaussian", "PA  10\tfixed", "ell  0.5 0.1,0.6",
                         "I_0  100  0,1e6", "sigma  10   fixed",
                         "FUNCTION Exponential",  "PA  20\tfixed", "ell  0.2 0.1,0.4",
                         "I_0  100  0,1e6", "h  100   1,400"]

fsetdesc_correct1b = copy.copy(fsetdesc_correct1)
fsetdesc_correct1b._name = "fs0"
modeldesc_correct1 = ModelDescription([fsetdesc_correct1b],
                                      options={"GAIN": 4.5, "READNOISE": 0.5})
fsetdesc_correct2b = copy.copy(fsetdesc_correct2)
fsetdesc_correct2b._name = "fs0"
modeldesc_correct2 = ModelDescription([fsetdesc_correct2b],
                                      options={"GAIN": 4.5, "READNOISE": 0.5})




def test_read_parameter_bad( ):
    """Test to see that we raise ValueError exceptions when line is malformed."""
    with pytest.raises(ValueError):
        pdesc = read_parameter(testLine_bad1)
    with pytest.raises(ValueError):
        pdesc = read_parameter(testLine_bad2)
    with pytest.raises(ValueError):
        pdesc = read_parameter(testLine_bad3)
    with pytest.raises(ValueError):
        pdesc = read_parameter(testLine_bad4)

def test_read_parameter_good( ):
    """Test that we correctly read valid parameter lines."""
    pdesc1 = read_parameter(testLine_good1)
    assert pdesc1 == pdesc_ref_correct123
    pdesc2 = read_parameter(testLine_good2)
    assert pdesc2 == pdesc_ref_correct123
    pdesc3 = read_parameter(testLine_good3)
    assert pdesc3 == pdesc_ref_correct123
    pdesc4 = read_parameter(testLine_good4)
    assert pdesc4 == pdesc_ref_correct4
    pdesc5 = read_parameter(testLine_good5)
    assert pdesc5 == pdesc_ref_correct5



def test_read_function_bad( ):
    """Test to see that we raise ValueError exceptions when line is malformed."""
    with pytest.raises(ValueError):
        fdesc = read_function(testFunctionLine_bad1)
    with pytest.raises(ValueError):
        fdesc = read_function(testFunctionLine_bad2)

def test_read_function_good( ):
    """Test that we correctly read valid function lines."""
    fdesc1 = read_function(testFunctionLines_good1)
    assert fdesc1 == fdesc_correct1
    fdesc2 = read_function(testFunctionLines_good2)
    assert fdesc2 == fdesc_correct2

def test_read_function_attributes( ):
    """Test that FunctionDescription attributes (ParameterDescription objects)
    get set correctly."""
    fdesc2 = read_function(testFunctionLines_good2)
    assert fdesc2._funcName == "Exponential"
    assert fdesc2.PA == ParameterDescription("PA", 20.0, fixed=True)
    assert fdesc2.ell == ParameterDescription("ell", 0.2, 0.1, 0.4)
    assert fdesc2.I_0 == ParameterDescription("I_0", 100.0, 0.0, 1.0e6)
    assert fdesc2.h == ParameterDescription("h", 100.0, 1.0, 400.0)



def test_read_function_set_good( ):
    """Test that we correctly read valid function-block lines."""
    fsetdesc1 = read_function_set("function_block_1", testFunctionBlockLines_good1)
    assert fsetdesc1 == fsetdesc_correct1
    fsetdesc2 = read_function_set("function_block_2", testFunctionBlockLines_good2)
    assert fsetdesc2 == fsetdesc_correct2



def test_read_options_bad( ):
    """Test to see that we raise ValueError exceptions when line is malformed."""
    inputLines1 = ["GAIN\t\t4.5", "READNOISE"]
    with pytest.raises(ValueError):
        configDict = read_options(inputLines1)
    inputLines2 = ["FUNCTION   Gaussian"]
    with pytest.raises(ValueError):
        configDict = read_options(inputLines2)

def test_read_options_good( ):
    """Test that we correctly parse line containing image-description parameters."""
    inputLines = ["GAIN\t\t4.5", "READNOISE   10.0"]
    configDict = read_options(inputLines)
    correctDict = {"GAIN": 4.5, "READNOISE": 10.0}
    assert configDict == correctDict



def test_parse_config( ):
    """Test that we correctly read valid configuration-file lines."""
    modeldesc1 = parse_config(testConfigLines_good1)
    assert modeldesc1 == modeldesc_correct1
    modeldesc2= parse_config(testConfigLines_good2)
    assert modeldesc2 == modeldesc_correct2


