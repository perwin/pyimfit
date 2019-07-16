# Unit tests for descriptions.py module of pyimfit
# Execute via
#    $ pytest test_descriptions.py

import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

from ..descriptions import ParameterDescription, FunctionDescription, FunctionSetDescription
from ..descriptions import ModelDescription, SimpleModelDescription

CONFIG_EXAMPLE_EXPONENTIAL = "../data/config_exponential_ic3478_256.dat"
CONFIG_EXAMPLE_2BLOCKS = "../data/config_imfit_2gauss_small.dat"



class TestParameterDescription(object):

    def test_ParameterDescription_simple( self ):
        pdesc = ParameterDescription('X0', 100.0)
        assert pdesc.name == "X0"
        assert pdesc.value == 100.0
        assert pdesc.limits is None
        assert pdesc.fixed == False


    def test_ParameterDescription_complex( self ):
        pdesc1 = ParameterDescription('X0', 100.0, 50.0, 150.0)
        assert pdesc1.name == "X0"
        assert pdesc1.value == 100.0
        assert pdesc1.limits == (50.0,150.0)
        assert pdesc1.fixed == False

        pdesc2 = ParameterDescription('X0', 100.0, fixed=True)
        assert pdesc2.name == "X0"
        assert pdesc2.value == 100.0
        assert pdesc2.limits is None
        assert pdesc2.fixed == True


    def test_ParameterDescription_complex2( self ):
        pdesc = ParameterDescription('X0', 100.0, [50.0, 150.0])
        assert pdesc.name == "X0"
        assert pdesc.value == 100.0
        assert pdesc.limits == (50.0,150.0)
        assert pdesc.fixed == False


    def test_ParameterDescription_setValue( self ):
        pdesc = ParameterDescription('X0', 100.0)
        pdesc.setValue(150.0, 10.0, 1e5)
        assert pdesc.name == "X0"
        assert pdesc.value == 150.0
        assert pdesc.limits == (10.0,1e5)
        assert pdesc.fixed == False


    def test_ParameterDescription_setThings( self ):
        pdesc = ParameterDescription('X0', 100.0, 50.0, 150.0)
        assert pdesc.name == "X0"
        assert pdesc.value == 100.0
        assert pdesc.limits == (50.0,150.0)
        assert pdesc.fixed == False

        # test setTolerance
        pdesc.setTolerance(0.1)
        assert pdesc.limits == approx((90.0, 110.0))

        # test setLimits
        pdesc.setLimits(10.0, 200.0)
        assert pdesc.limits == approx((10.0, 200.0))

        # test setLimitsRel
        pdesc.setLimitsRel(5, 25)
        assert pdesc.limits == approx((95.0, 125.0))


    def test_ParameterDescription_getString(self):
        pdesc1 = ParameterDescription('X0', 100.0)
        outString1_correct = "X0\t\t100.0"
        pdesc2 = ParameterDescription('X0', 100.0, fixed=True)
        outString2_correct = "X0\t\t100.0\t\tfixed"
        pdesc3 = ParameterDescription('X0', 100.0, 50.0, 150.0)
        outString3_correct = "X0\t\t100.0\t\t50.0,150.0"
        # user requested no limits be printed
        outString3b_correct = "X0\t\t100.0"
        # user supplied error value
        outString3c_correct = "X0\t\t100.0\t\t# +/- 0.501"

        outString1 = pdesc1.getStringDescription()
        assert outString1 == outString1_correct
        outString2 = pdesc2.getStringDescription()
        assert outString2 == outString2_correct
        outString3 = pdesc3.getStringDescription()
        assert outString3 == outString3_correct

        # user requested no limits be printed
        outString3b = pdesc3.getStringDescription(noLimits=True)
        assert outString3b == outString3b_correct
        # user supplied error value
        outString3c = pdesc3.getStringDescription(error=0.501)
        assert outString3c == outString3c_correct



class TestFunctionDescription(object):

    def setup_method( self ):
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, 0.1,0.8)
        self.p3 = ParameterDescription("I_0", 100.0, 10.0, 1e3)
        self.p4 = ParameterDescription("sigma", 10.0, 5.0,20.0)
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]

    def test_FunctionDescription_simple( self ):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        assert fdesc1._funcName == "Gaussian"
        assert fdesc1._label == "blob"
        assert fdesc1.PA == self.p1
        assert fdesc1.ell == self.p2
        assert fdesc1.I_0 == self.p3
        assert fdesc1.sigma == self.p4
        plist = fdesc1.parameterList()
        assert plist == self.paramDescList

    def test_FunctionDescription_getStrings(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)

        lines_correct = ["FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\t\tfixed\n",
                         "ell\t\t0.5\t\t0.1,0.8\n",
                         "I_0\t\t100.0\t\t10.0,1000.0\n",
                         "sigma\t\t10.0\t\t5.0,20.0\n"]
        outputLines = fdesc1.getStringDescription()
        assert outputLines == lines_correct

    def test_FunctionDescription_getStrings_bestfit(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)

        lines_correct = ["FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\n",
                         "ell\t\t0.5\n",
                         "I_0\t\t100.0\n",
                         "sigma\t\t10.0\n"]
        outputLines = fdesc1.getStringDescription(noLimits=True)
        assert outputLines == lines_correct

    def test_FunctionDescription_getStrings_bestfit_with_errors(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        errorValues = np.array([0.0, 0.001073, 3.4567, 0.333])

        lines_correct = ["FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\t\t# +/- 0.0\n",
                         "ell\t\t0.5\t\t# +/- 0.001073\n",
                         "I_0\t\t100.0\t\t# +/- 3.4567\n",
                         "sigma\t\t10.0\t\t# +/- 0.333\n"]
        outputLines = fdesc1.getStringDescription(errors=errorValues)
        assert outputLines == lines_correct



class TestFunctionSetDescription(object):

    def setup_method( self ):
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, 0.1,0.8)
        self.p3 = ParameterDescription("I_0", 100.0, 10.0, 1e3)
        self.p4 = ParameterDescription("sigma", 10.0, 5.0,20.0)
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]

        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, 180.0, 220.0)


    def test_FunctionSetDescription_simple( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        assert fsetdesc1.name == "fs0"
        assert fsetdesc1._contains("blob") is True
        assert fsetdesc1.blob == self.fdesc1
        assert fsetdesc1.x0 == self.x0_p
        assert fsetdesc1.y0 == self.y0_p

    def test_FunctionSetDescription_parameterList( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        plist_correct = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        assert fsetdesc1.parameterList() == plist_correct

    def test_FunctionSetDescription_parameterList_badInput( self ):
        # x0 and y0 should be ParameterDescription values, not floats
        with pytest.raises(ValueError):
            fsetdesc1 = FunctionSetDescription('fs0', 100.0, 200.0, self.functionList)

    def test_FunctionSetDescription_catchErrors( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        # do we catch the case of trying to add a non-FunctionDescription object?
        with pytest.raises(ValueError):
            fsetdesc1.addFunction(self.p1)
        # do we catch the case of trying to add a duplicate function?
        with pytest.raises(KeyError):
            fsetdesc1.addFunction(self.fdesc1)

    def test_FunctionSetDescription_getStrings(self):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)

        lines_correct = ["X0\t\t100.0\t\tfixed\n",
                         "Y0\t\t200.0\t\t180.0,220.0\n",
                        "FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\t\tfixed\n",
                         "ell\t\t0.5\t\t0.1,0.8\n",
                         "I_0\t\t100.0\t\t10.0,1000.0\n",
                         "sigma\t\t10.0\t\t5.0,20.0\n"]
        outputLines = fsetdesc1.getStringDescription()
        assert outputLines == lines_correct

    def test_FunctionSetDescription_getStrings_no_limits(self):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)

        lines_correct = ["X0\t\t100.0\n",
                         "Y0\t\t200.0\n",
                        "FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\n",
                         "ell\t\t0.5\n",
                         "I_0\t\t100.0\n",
                         "sigma\t\t10.0\n"]
        outputLines = fsetdesc1.getStringDescription(noLimits=True)
        assert outputLines == lines_correct

    def test_FunctionSetDescription_getStrings_with_errors(self):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)

        errorValues = np.array([0.0, 2.456, 0.0, 0.001073, 3.4567, 0.333])

        lines_correct = ["X0\t\t100.0\t\t# +/- 0.0\n",
                         "Y0\t\t200.0\t\t# +/- 2.456\n",
                         "FUNCTION Gaussian   # blob\n",
                         "PA\t\t0.0\t\t# +/- 0.0\n",
                         "ell\t\t0.5\t\t# +/- 0.001073\n",
                         "I_0\t\t100.0\t\t# +/- 3.4567\n",
                         "sigma\t\t10.0\t\t# +/- 0.333\n"]
        outputLines = fsetdesc1.getStringDescription(errors=errorValues)
        assert outputLines == lines_correct



class TestModelDescription(object):

    def setup_method( self ):
        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, 180.0, 220.0)
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, 0.1,0.8)
        self.p3 = ParameterDescription("I_0", 100.0, 10.0, 1e3)
        self.p4 = ParameterDescription("sigma", 10.0, 5.0,20.0)
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fullParamDescList = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]
        self.fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        self.fsetList = [self.fsetdesc1]

    def test_ModelDescription_simple( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        assert modeldesc1.functionSetIndices() == [0]
        assert modeldesc1.functionList() == ['Gaussian']
        assert modeldesc1.parameterList() == self.fullParamDescList

    def test_ModelDescription_getParamLimits( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        pLimits = modeldesc1.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

    def testModelDescription_get_and_set_options( self ):
        modeldesc1 = ModelDescription(self.fsetList)

        assert {} == modeldesc1.optionsDict
        optionsDict = {"GAIN": 4.5, "READNOISE": 0.9}
        modeldesc1.updateOptions(optionsDict)
        assert optionsDict == modeldesc1.optionsDict
        optionsDict2 = {"GAIN": 10.5, "READNOISE": 0.9, "ORIGINAL_SKY": 45.01}
        modeldesc1.updateOptions(optionsDict2)
        assert optionsDict2 == modeldesc1.optionsDict

    def test_ModelDescription_load_from_file( self ):
        x0_p = ParameterDescription("X0", 129.0, 125,135)
        y0_p = ParameterDescription("Y0", 129.0, 125,135)
        p1 = ParameterDescription("PA", 18.0, 0,90)
        p2 = ParameterDescription("ell", 0.2, 0.0,1)
        p3 = ParameterDescription("I_0", 100.0, 0, 500)
        p4 = ParameterDescription("h", 25, 0,100)
        paramDescList = [p1, p2, p3, p4]
        fullParamDescList = [x0_p, y0_p, p1, p2, p3, p4]
        fdesc = FunctionDescription('Exponential', "blob", paramDescList)

        modeldesc2 = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        assert modeldesc2.functionSetIndices() == [0]
        assert modeldesc2.functionList() == ['Exponential']
        assert modeldesc2.parameterList() == fullParamDescList

        input_params_correct = np.array([129.0,129.0, 18.0,0.2,100.0,25.0])
        assert_allclose(modeldesc2.getRawParameters(), input_params_correct)

    def test_ModelDescription_load_from_file_2blocks( self ):
        modeldesc2blocks = ModelDescription.load(CONFIG_EXAMPLE_2BLOCKS)

    def test_ModelDescription_getStrings(self):
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)

        x0_p = ParameterDescription("X0", 129.0, 125,135)
        y0_p = ParameterDescription("Y0", 129.0, 125,135)
        p1 = ParameterDescription("PA", 18.0, 0,90)
        p2 = ParameterDescription("ell", 0.2, 0.0,1)
        p3 = ParameterDescription("I_0", 100.0, 0, 500)
        p4 = ParameterDescription("h", 25, 0,100)
        paramDescList = [p1, p2, p3, p4]
        fullParamDescList = [x0_p, y0_p, p1, p2, p3, p4]

        assert modeldesc.functionSetIndices() == [0]
        assert modeldesc.functionList() == ['Exponential']
        assert modeldesc.parameterList() == fullParamDescList

        lines_correct = ["\n", "X0\t\t129.0\t\t125.0,135.0\n",
                        "Y0\t\t129.0\t\t125.0,135.0\n",
                        "FUNCTION Exponential\n",
                        "PA\t\t18.0\t\t0.0,90.0\n",
                        "ell\t\t0.2\t\t0.0,1.0\n",
                        "I_0\t\t100.0\t\t0.0,500.0\n",
                        "h\t\t25.0\t\t0.0,100.0\n"]
        outputLines = modeldesc.getStringDescription()
        print(outputLines)
        assert outputLines == lines_correct

    def test_ModelDescription_getStrings_with_options(self):
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        optionsDict = {"GAIN": 4.5, "READNOISE": 0.9}
        modeldesc.updateOptions(optionsDict)

        lines_correct = ["GAIN\t\t4.5\n", "READNOISE\t\t0.9\n", "\n",
                         "X0\t\t129.0\t\t125.0,135.0\n",
                        "Y0\t\t129.0\t\t125.0,135.0\n",
                        "FUNCTION Exponential\n",
                        "PA\t\t18.0\t\t0.0,90.0\n",
                        "ell\t\t0.2\t\t0.0,1.0\n",
                        "I_0\t\t100.0\t\t0.0,500.0\n",
                        "h\t\t25.0\t\t0.0,100.0\n"]
        outputLines = modeldesc.getStringDescription(saveOptions=True)
        print(outputLines)
        assert outputLines == lines_correct

    def test_ModelDescription_getStrings_without_options(self):
        # get strings without options, even though model *has* options
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        optionsDict = {"GAIN": 4.5, "READNOISE": 0.9}
        modeldesc.updateOptions(optionsDict)

        lines_correct = ["\n",
                         "X0\t\t129.0\t\t125.0,135.0\n",
                        "Y0\t\t129.0\t\t125.0,135.0\n",
                        "FUNCTION Exponential\n",
                        "PA\t\t18.0\t\t0.0,90.0\n",
                        "ell\t\t0.2\t\t0.0,1.0\n",
                        "I_0\t\t100.0\t\t0.0,500.0\n",
                        "h\t\t25.0\t\t0.0,100.0\n"]
        outputLines = modeldesc.getStringDescription(saveOptions=False)
        print(outputLines)
        assert outputLines == lines_correct

    def test_ModelDescription_getStrings_with_errors(self):
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        errorValues = np.array([1.45, 0.72, 1.01, 0.00456, 12.345, 3.0024])

        lines_correct = ["\n", "X0\t\t129.0\t\t# +/- 1.45\n",
                        "Y0\t\t129.0\t\t# +/- 0.72\n",
                        "FUNCTION Exponential\n",
                        "PA\t\t18.0\t\t# +/- 1.01\n",
                        "ell\t\t0.2\t\t# +/- 0.00456\n",
                        "I_0\t\t100.0\t\t# +/- 12.345\n",
                        "h\t\t25.0\t\t# +/- 3.0024\n"]
        outputLines = modeldesc.getStringDescription(errors=errorValues)
        print(outputLines)
        assert outputLines == lines_correct



class TestSimpleModelDescription(object):

    def setup_method( self ):
        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, 180.0, 220.0)
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, 0.1,0.8)
        self.p3 = ParameterDescription("I_0", 100.0, 10.0, 1e3)
        self.p4 = ParameterDescription("sigma", 10.0, 5.0,20.0)
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fullParamDescList = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]
        self.fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        self.fsetList = [self.fsetdesc1]
        # bad example: 2 function sets
        self.fsetdesc2 = FunctionSetDescription('fs1', self.x0_p, self.y0_p, self.functionList)
        self.fsetList_bad = [self.fsetdesc1, self.fsetdesc2]

    def test_SimpleModelDescription_bad( self ):
        # this attempts to instantiate a SimpleModelDescription instance with *two*
        # function sets, which is one more the SimpleModelDescription can handle
        modeldesc_bad = ModelDescription(self.fsetList_bad)
        with pytest.raises(ValueError):
            simplemodeldesc = SimpleModelDescription(modeldesc_bad)

    def test_SimpleModelDescription_simple( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        simplemodeldesc = SimpleModelDescription(modeldesc1)
        print(dir(simplemodeldesc))
        # NOTE: the following does NOT work!
        #assert simplemodeldesc.name == "fs0"
        # properties of SimpleModelDescription
        assert simplemodeldesc.x0 == self.x0_p
        assert simplemodeldesc.y0 == self.y0_p

        assert simplemodeldesc.functionSetIndices() == [0]
        assert simplemodeldesc.functionList() == ['Gaussian']

        pLimits = simplemodeldesc.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

    def testSimpleModelDescription_get_and_set_options(self):
        modeldesc1 = ModelDescription(self.fsetList)
        simplemodeldesc = SimpleModelDescription(modeldesc1)

        assert {} == simplemodeldesc.optionsDict
        optionsDict = {"GAIN": 4.5, "READNOISE": 0.9}
        simplemodeldesc.updateOptions(optionsDict)
        assert optionsDict == simplemodeldesc.optionsDict
        optionsDict2 = {"GAIN": 10.5, "READNOISE": 0.9, "ORIGINAL_SKY": 45.01}
        simplemodeldesc.updateOptions(optionsDict2)
        assert optionsDict2 == simplemodeldesc.optionsDict



