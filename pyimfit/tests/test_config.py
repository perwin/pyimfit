# Test code for config.py module of pyimfit
# Execute via
#    $ pytest test_config.py

import pytest

from ..model import ParameterDescription, FunctionDescription
from ..config import read_function, read_parameter

# inputs and reference outputs for read_function
testFunctionLine_bad1 = ["bob"]
testFunctionLine_bad2 = ["FUNCTION"]

testFunctionLines_good1 = ["FUNCTION Gaussian", "PA  10", "ell  0.5", "I_0  100", "sigma  10"]

fdesc_good = FunctionDescription("Gaussian")
for line in testFunctionLines_good1[1:]:
	pdesc = read_parameter(line)
	fdesc_good.addParameter(pdesc)


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

pdesc_ref_good123 = ParameterDescription("X0", 100.1)
pdesc_ref_good4 = ParameterDescription("X0", 100.1, fixed=True)
pdesc_ref_good5 = ParameterDescription("X0", 100.1, 1.0, 500.0)



def test_read_function_bad( ):
	"""Test to see that we raise ValueError exceptions when line is misformed."""
	with pytest.raises(ValueError):
		fdesc = read_function(testFunctionLine_bad1)
	with pytest.raises(ValueError):
		fdesc = read_function(testFunctionLine_bad2)

def test_read_function_good( ):
	"""Test that we correctly read valid function-block lines."""
	fdesc1 = read_function(testFunctionLines_good1)
	assert fdesc1 == fdesc_good


	
def test_read_parameter_bad( ):
	"""Test to see that we raise ValueError exceptions when line is misformed."""
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
	assert pdesc1 == pdesc_ref_good123
	pdesc2 = read_parameter(testLine_good2)
	assert pdesc2 == pdesc_ref_good123
	pdesc3 = read_parameter(testLine_good3)
	assert pdesc3 == pdesc_ref_good123
	pdesc4 = read_parameter(testLine_good4)
	assert pdesc4 == pdesc_ref_good4
	pdesc5 = read_parameter(testLine_good5)
	assert pdesc5 == pdesc_ref_good5


