# Code for testing config.py module of pyimfit
# Execute via
#    $ pytest test_config.py

# NOTE: we run into problems comparing dictionaries, because Python 3.5 has old-style
# dictionaries with random ordering, while Python 3.6 and 3.7 have new-style dicts with
# insertion-ordering. Thus, we have to be careful any time we're going to compare
# objects with built-in dicts or OrderedDicts, and any time we compare output based
# on internal dicts or OrderedDicts.

from collections import OrderedDict
import copy
import pytest

from ..descriptions import ParameterDescription, FunctionDescription, FunctionSetDescription
from ..descriptions import ModelDescription
from ..config import read_parameter, read_function, read_function_set, read_options
from ..config import parse_config, parse_config_file


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
pdesc_ref_correct5 = ParameterDescription("X0", 100.1, [1.0, 500.0])


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
testFunctionBlockLines_bad1 = ["X0  100", "FUNCTION Gaussian", "PA  10",
                                "ell  0.5", "I_0  100", "sigma  10"]
testFunctionBlockLines_bad2 = ["X  100", "Y0 200", "FUNCTION Gaussian", "PA  10",
                                "ell  0.5", "I_0  100", "sigma  10"]


fsetdesc_correct1 = FunctionSetDescription("", x0param=ParameterDescription('X0', 100.0),
                                           y0param=ParameterDescription('Y0', 200.0))
fsetdesc_correct1.addFunction(fdesc_correct1)

fdesc_fblock2_correct1 = FunctionDescription("Gaussian")
for line in testFunctionBlockLines_good2[3:7]:
    pdesc = read_parameter(line)
    fdesc_fblock2_correct1.addParameter(pdesc)
fsetdesc_correct2 = FunctionSetDescription("", x0param=ParameterDescription('X0', 100.0, fixed=True),
                                           y0param=ParameterDescription('Y0', 200.0, fixed=True))
fsetdesc_correct2.addFunction(fdesc_fblock2_correct1)
fsetdesc_correct2.addFunction(fdesc_correct2)


# inputs and reference outputs for parse_config
testConfigLines_good1 = ["GAIN 4.5", "READNOISE\t0.5", "X0  100", "Y0  200",
                         "FUNCTION Gaussian", "PA  10",
                         "ell  0.5", "I_0  100", "sigma  10"]
testConfigLines_good2 = ["GAIN 4.5", "READNOISE\t0.5", "ORIGINAL_SKY\t\t154.33",
                         "X0  100\tfixed", "Y0  200\tfixed",
                         "FUNCTION Gaussian", "PA  10\tfixed", "ell  0.5 0.1,0.6",
                         "I_0  100  0,1e6", "sigma  10   fixed",
                         "FUNCTION Exponential",  "PA  20\tfixed", "ell  0.2 0.1,0.4",
                         "I_0  100  0,1e6", "h  100   1,400"]
TEST_CONFIG_FILE = "../data/config_exponential_ic3478_256.dat"

# use an OrderedDict for the input to ensure proper ordering of the internal
# OrderedDicts, especially under Python 3.5.
optionsDict_ord = OrderedDict()
optionsDict_ord["GAIN"] = 4.5
optionsDict_ord["READNOISE"] = 0.5
optionsDict_ord2 = OrderedDict()
optionsDict_ord2["GAIN"] = 4.5
optionsDict_ord2["READNOISE"] = 0.5
optionsDict_ord2["ORIGINAL_SKY"] = 154.33

fsetdesc_correct1b = copy.copy(fsetdesc_correct1)
# fsetdesc_correct1b._name = "fs0"
modeldesc_correct1 = ModelDescription([fsetdesc_correct1b],
                                      options=optionsDict_ord)
fsetdesc_correct2b = copy.copy(fsetdesc_correct2)
# fsetdesc_correct2b._name = "fs0"
modeldesc_correct2 = ModelDescription([fsetdesc_correct2b],
                                      options=optionsDict_ord2)

# correct result for reading from file TEST_CONFIG_FILE
p1 = ParameterDescription("PA", 18.0, [0.0,90.0])
p2 = ParameterDescription("ell", 0.2, [0, 1])
p3 = ParameterDescription("I_0", 100.0, [0.0,500.0])
p4 = ParameterDescription("h", 25.0, [0.0, 100.0])
paramDescList = [p1, p2, p3, p4]
function_from_file = FunctionDescription("Exponential", parameters=paramDescList)
fsetdesc_from_file_correct = FunctionSetDescription("",
                                           x0param=ParameterDescription('X0', 129.0, [125,135]),
                                           y0param=ParameterDescription('Y0', 129.0, [125,135]),
                                           functionList=[function_from_file])
modeldesc_from_file_correct = ModelDescription([fsetdesc_from_file_correct])




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
    assert fdesc2.ell == ParameterDescription("ell", 0.2, [0.1, 0.4])
    assert fdesc2.I_0 == ParameterDescription("I_0", 100.0, [0.0, 1.0e6])
    assert fdesc2.h == ParameterDescription("h", 100.0, [1.0, 400.0])



def test_read_function_set_good( ):
    """Test that we correctly read valid function-block lines."""
    fsetdesc1 = read_function_set("", testFunctionBlockLines_good1)
    assert fsetdesc1 == fsetdesc_correct1
    fsetdesc2 = read_function_set("", testFunctionBlockLines_good2)
    assert fsetdesc2 == fsetdesc_correct2

def test_read_function_set_bad( ):
    """Test that we correctly catch bad function-block lines."""
    with pytest.raises(ValueError):
        fsetdesc1 = read_function_set("", testFunctionBlockLines_bad1)
    with pytest.raises(ValueError):
        fsetdesc2 = read_function_set("", testFunctionBlockLines_bad2)



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
    correctDict = OrderedDict()
    correctDict["GAIN"] = 4.5
    correctDict["READNOISE"] = 10.0
    assert configDict == correctDict

    # check to see that we skip unrecognized image-description parameter
    inputLines2 = ["GAIN\t\t4.5", "READNOISE   10.0", "EVANESCENCE   -2.5"]
    configDict2 = read_options(inputLines2)
    assert configDict2 == correctDict

    # check to see that we recognize makeimage-mode parameters
    correctDict3 = OrderedDict()
    correctDict3["NCOLS"] = 100
    correctDict3["NROWS"] = 110
    correctDict3["GAIN"] = 4.5
    correctDict3["READNOISE"] = 10.0
    inputLines3 = ["NCOLS\t\t100", "NROWS    110", "GAIN\t\t4.5", "READNOISE   10.0"]
    configDict3 = read_options(inputLines3)
    assert configDict3 == correctDict3


def test_parse_config( ):
    """Test that we correctly read valid configuration-file lines."""
    modeldesc1 = parse_config(testConfigLines_good1)
    assert modeldesc1 == modeldesc_correct1
    modeldesc2= parse_config(testConfigLines_good2)
    assert modeldesc2 == modeldesc_correct2

def test_parse_config_file( ):
    """Test that we correctly read and parse a valid configuration file."""
    modeldesc_from_file = parse_config_file(TEST_CONFIG_FILE)
    # easiest thing to do is compare string representations
    assert str(modeldesc_from_file) == str(modeldesc_from_file_correct)
