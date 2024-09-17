# Unit tests for descriptions.py module of pyimfit
# Execute via
#    $ pytest test_descriptions.py

import pytest
from pytest import approx
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose

from ..descriptions import ParameterDescription, FunctionDescription, FunctionSetDescription
from ..descriptions import ModelDescription, SimpleModelDescription

CONFIG_EXAMPLE_EXPONENTIAL = "../data/config_exponential_ic3478_256.dat"
CONFIG_EXAMPLE_2SETS = "../data/config_imfit_2gauss_small.dat"
CONFIG_EXAMPLE_MULTISET = "../data/config_imfit_multiset.dat"


class TestParameterDescription(object):

    def test_ParameterDescription_simple( self ):
        pdesc = ParameterDescription('X0', 100.0)
        assert pdesc.name == "X0"
        assert pdesc.value == 100.0
        assert pdesc.limits is None
        assert pdesc.fixed == False


    def test_ParameterDescription_complex( self ):
        pdesc1 = ParameterDescription('X0', 100.0, [50.0, 150.0])
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
        pdesc.setValue(150.0, [10.0, 1e5])
        assert pdesc.name == "X0"
        assert pdesc.value == 150.0
        assert pdesc.limits == (10.0,1e5)
        assert pdesc.fixed == False

    def test_ParameterDescription_setValue_limits( self ):
        pdesc1 = ParameterDescription('X0', 100.0)
        pdesc1.setValue(150.0, [200.0, 1e5])
        assert pdesc1.value == 150.0
        assert pdesc1.limits == (150.0,1e5)
        assert pdesc1.fixed == False
        pdesc2 = ParameterDescription('X0', 100.0)
        pdesc2.setValue(150.0, [100.0, 120])
        assert pdesc2.value == 150.0
        assert pdesc2.limits == (100.0,150.0)
        assert pdesc2.fixed == False

    def test_ParameterDescription_setValue_bad( self ):
        with pytest.raises(ValueError):
            pdesc = ParameterDescription('X0', 100.0, 50.0)
        with pytest.raises(ValueError):
            pdesc = ParameterDescription('X0', 100.0, [150,100])


    def test_ParameterDescription_setThings( self ):
        pdesc = ParameterDescription('X0', 100.0, [50.0, 150.0])
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
        pdesc.setLimits(101, 200)
        assert pdesc.limits == approx((100.0, 200.0))
        pdesc.setLimits(50, 90)
        assert pdesc.limits == approx((50.0, 100.0))

        # test setLimitsRel
        pdesc.setLimitsRel(5, 25)
        assert pdesc.limits == approx((95.0, 125.0))

    def test_ParameterDescription_setThings_bad( self ):
        pdesc = ParameterDescription('X0', 100.0, [50.0, 150.0])
        assert pdesc.name == "X0"
        assert pdesc.value == 100.0
        assert pdesc.limits == (50.0,150.0)
        assert pdesc.fixed == False

        # test setTolerance -- tolerance must be between 0 and 1
        with pytest.raises(ValueError):
            pdesc.setTolerance(-0.1)
        with pytest.raises(ValueError):
            pdesc.setTolerance(10.0)

        # test setLimits -- lower limit must be < upper limit
        with pytest.raises(ValueError):
            pdesc.setLimits(10.0, 5.0)

        # test setLimitsRel -- relative offsets should both be positive
        with pytest.raises(ValueError):
            pdesc.setLimitsRel(-1,2)
        with pytest.raises(ValueError):
            pdesc.setLimitsRel(1,-2)


    def test_ParameterDescription_getString(self):
        pdesc1 = ParameterDescription('X0', 100.0)
        outString1_correct = "X0\t\t100.0"
        pdesc2 = ParameterDescription('X0', 100.0, fixed=True)
        outString2_correct = "X0\t\t100.0\t\tfixed"
        pdesc3 = ParameterDescription('X0', 100.0, [50.0, 150.0])
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

    def test_ParameterDescription_str(self):
        pdesc1 = ParameterDescription('X0', 100.0)
        outString1_correct = "X0\t\t100.0"
        pdesc2 = ParameterDescription('X0', 100.0, fixed=True)
        outString2_correct = "X0\t\t100.0\t\tfixed"
        pdesc3 = ParameterDescription('X0', 100.0, [50.0, 150.0])
        outString3_correct = "X0\t\t100.0\t\t50.0,150.0"
        # user requested no limits be printed
        outString3b_correct = "X0\t\t100.0"
        # user supplied error value
        outString3c_correct = "X0\t\t100.0\t\t# +/- 0.501"

        outString1 = str(pdesc1)
        assert outString1 == outString1_correct
        outString2 = str(pdesc2)
        assert outString2 == outString2_correct
        outString3 = str(pdesc3)
        assert outString3 == outString3_correct

    def test_getParamInfoList(self):
        pdesc1 = ParameterDescription('X0', 100.0)
        outList1_correct = [100.0]
        pdesc2 = ParameterDescription('X0', 100.0, fixed=True)
        outList2_correct = [100.0, "fixed"]
        pdesc3 = ParameterDescription('X0', 100.0, [50.0, 150.0])
        outList3_correct = [100.0, 50.0, 150.0]
        outList1 = pdesc1.getParamInfoList()
        assert outList1 == outList1_correct
        outList2 = pdesc2.getParamInfoList()
        assert outList2 == outList2_correct
        outList3 = pdesc3.getParamInfoList()
        assert outList3 == outList3_correct


class TestFunctionDescription(object):

    def setup_method( self ):
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, [0.1,0.8])
        self.p3 = ParameterDescription("I_0", 100.0, [10.0, 1e3])
        self.p4 = ParameterDescription("sigma", 10.0, [5.0,20.0])
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

    def test_FunctionDescription_getParamNames(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)

        paramNames_correct = ["PA", "ell", "I_0", "sigma"]
        outputNames = fdesc1.parameterNameList()
        assert outputNames == paramNames_correct

    def test_FunctionDescription_getStrings(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)

        lines_correct = ["FUNCTION Gaussian   # LABEL blob\n",
                         "PA\t\t0.0\t\tfixed\n",
                         "ell\t\t0.5\t\t0.1,0.8\n",
                         "I_0\t\t100.0\t\t10.0,1000.0\n",
                         "sigma\t\t10.0\t\t5.0,20.0\n"]
        outputLines = fdesc1.getStringDescription()
        assert outputLines == lines_correct

        # test the str() representation:
        str_correct = "".join(lines_correct)
        output_str = str(fdesc1)
        assert output_str == str_correct

    def test_FunctionDescription_getStrings_bestfit(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)

        lines_correct = ["FUNCTION Gaussian   # LABEL blob\n",
                         "PA\t\t0.0\n",
                         "ell\t\t0.5\n",
                         "I_0\t\t100.0\n",
                         "sigma\t\t10.0\n"]
        outputLines = fdesc1.getStringDescription(noLimits=True)
        assert outputLines == lines_correct

    def test_FunctionDescription_getStrings_bestfit_with_errors(self):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        errorValues = np.array([0.0, 0.001073, 3.4567, 0.333])

        lines_correct = ["FUNCTION Gaussian   # LABEL blob\n",
                         "PA\t\t0.0\t\t# +/- 0.0\n",
                         "ell\t\t0.5\t\t# +/- 0.001073\n",
                         "I_0\t\t100.0\t\t# +/- 3.4567\n",
                         "sigma\t\t10.0\t\t# +/- 0.333\n"]
        outputLines = fdesc1.getStringDescription(errors=errorValues)
        assert outputLines == lines_correct

    def test_DictToFunction_noParamLimits(self):
        # start out with parameters having no limits
        ref_p1 = ParameterDescription("PA", 0.0)
        ref_p2 = ParameterDescription("ell", 0.5)
        ref_p3 = ParameterDescription("I_0", 100.0)
        ref_p4 = ParameterDescription("sigma", 10.0)
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)

        p = {'PA': 0.0, 'ell': 0.5, 'I_0': 100.0, 'sigma': 10.0}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fdesc_from_dict = FunctionDescription.dict_to_FunctionDescription(fDict)
        assert fdesc_from_dict == fdesc_correct

    def test_DictToFunction_withParamLimits(self):
        ref_p1 = ParameterDescription("PA", 0.0, fixed=True)
        ref_p2 = ParameterDescription("ell", 0.5, limits=[0.1,0.8])
        ref_p3 = ParameterDescription("I_0", 100.0, limits=[10.0, 1e3])
        ref_p4 = ParameterDescription("sigma", 10.0, limits=[5.0,20.0])
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)

        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1,0.8], 'I_0': [100.0, 10.0,1e3],
             'sigma': [10.0, 5.0,20.0]}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fdesc_from_dict = FunctionDescription.dict_to_FunctionDescription(fDict)
        assert fdesc_from_dict == fdesc_correct

    def test_getFunctionAsDict(self):
        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1,0.8], 'I_0': [100.0, 10.0,1e3],
             'sigma': [10.0, 5.0,20.0]}
        fDict_correct = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        # round-trip test
        fdesc_from_dict = FunctionDescription.dict_to_FunctionDescription(fDict_correct)
        fDict = fdesc_from_dict.getFunctionAsDict()
        assert fDict == fDict_correct


class TestFunctionSetDescription(object):

    def setup_method( self ):
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, [0.1,0.8])
        self.p3 = ParameterDescription("I_0", 100.0, [10.0, 1e3])
        self.p4 = ParameterDescription("sigma", 10.0, [5.0,20.0])
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]
        self.N_PARAMS_CORRECT = 4 + 2

        self.p1_2 = ParameterDescription("I_tot", 1000.0, [0,1e5])
        self.paramDescList2 = [self.p1_2]
        self.fdesc2 = FunctionDescription('PointSource', "nsc", self.paramDescList2)
        self.functionList2 = [self.fdesc2]
        self.N_PARAMS_CORRECT2 = 1 + 2

        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, [180.0, 220.0])


    def test_FunctionSetDescription_simple( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        assert fsetdesc1.label == "fs0"
        # check to see if FunctionDescription object with name "blob" (part of self.functionList)
        # can be accessed as attribute
        assert fsetdesc1._contains("blob") is True
        assert fsetdesc1.hasPointSources is False
        assert fsetdesc1.blob == self.fdesc1
        assert fsetdesc1.x0 == self.x0_p
        assert fsetdesc1.y0 == self.y0_p
        assert fsetdesc1.nParameters == self.N_PARAMS_CORRECT

    def test_FunctionSetDescription_simple2( self ):
        fsetdesc1 = FunctionSetDescription(None, self.x0_p, self.y0_p, self.functionList)
        assert fsetdesc1._contains("blob") is True
        assert fsetdesc1.hasPointSources is False
        assert fsetdesc1.blob == self.fdesc1
        assert fsetdesc1.label is None
        assert fsetdesc1.x0 == self.x0_p
        assert fsetdesc1.y0 == self.y0_p
        assert fsetdesc1.nParameters == self.N_PARAMS_CORRECT

    def test_FunctionSetDescription_keywords_noname( self ):
        fsetdesc1 = FunctionSetDescription(x0param=self.x0_p, y0param=self.y0_p, functionList=self.functionList)
        assert fsetdesc1.label is None
        assert fsetdesc1._contains("blob") is True
        assert fsetdesc1.hasPointSources is False
        assert fsetdesc1.blob == self.fdesc1
        assert fsetdesc1.x0 == self.x0_p
        assert fsetdesc1.y0 == self.y0_p
        assert fsetdesc1.nParameters == self.N_PARAMS_CORRECT

    def test_FunctionSetDescription_hasPointSource( self ):
        fsetdesc2 = FunctionSetDescription(x0param=self.x0_p, y0param=self.y0_p, functionList=self.functionList2)
        assert fsetdesc2.label is None
        assert fsetdesc2._contains("nsc") is True
        assert fsetdesc2.hasPointSources is True
        assert fsetdesc2.nsc == self.fdesc2
        assert fsetdesc2.x0 == self.x0_p
        assert fsetdesc2.y0 == self.y0_p
        assert fsetdesc2.nParameters == self.N_PARAMS_CORRECT2

    def test_FunctionSetDescription_parameterList( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        plist_correct = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        assert fsetdesc1.parameterList() == plist_correct

    def test_FunctionSetDescription_parameterList_badInput( self ):
        # x0 and y0 should be ParameterDescription values, not floats
        with pytest.raises(ValueError):
            fsetdesc1 = FunctionSetDescription('fs0', 100.0, 200.0, self.functionList)
        with pytest.raises(ValueError):
            fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, 200.0, self.functionList)

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
                        "FUNCTION Gaussian   # LABEL blob\n",
                         "PA\t\t0.0\t\tfixed\n",
                         "ell\t\t0.5\t\t0.1,0.8\n",
                         "I_0\t\t100.0\t\t10.0,1000.0\n",
                         "sigma\t\t10.0\t\t5.0,20.0\n"]
        outputLines = fsetdesc1.getStringDescription()
        assert outputLines == lines_correct

        # test the str() representation:
        str_correct = "".join(lines_correct)
        output_str = str(fsetdesc1)
        assert output_str == str_correct

    def test_FunctionSetDescription_getStrings_no_limits(self):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)

        lines_correct = ["X0\t\t100.0\n",
                         "Y0\t\t200.0\n",
                        "FUNCTION Gaussian   # LABEL blob\n",
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
                         "FUNCTION Gaussian   # LABEL blob\n",
                         "PA\t\t0.0\t\t# +/- 0.0\n",
                         "ell\t\t0.5\t\t# +/- 0.001073\n",
                         "I_0\t\t100.0\t\t# +/- 3.4567\n",
                         "sigma\t\t10.0\t\t# +/- 0.333\n"]
        outputLines = fsetdesc1.getStringDescription(errors=errorValues)
        assert outputLines == lines_correct

    def test_DictToFunctionSet_noParamLimits(self):
        # start out with parameters having no limits
        ref_p1 = ParameterDescription("PA", 0.0)
        ref_p2 = ParameterDescription("ell", 0.5)
        ref_p3 = ParameterDescription("I_0", 100.0)
        ref_p4 = ParameterDescription("sigma", 10.0)
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)
        ref_x0_p = ParameterDescription("X0", 100.0)
        ref_y0_p = ParameterDescription("Y0", 200.0)
        fsetdesc_correct = FunctionSetDescription(None, ref_x0_p, ref_y0_p, [fdesc_correct])

        p = {'PA': 0.0, 'ell': 0.5, 'I_0': 100.0, 'sigma': 10.0}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict = {'X0': 100.0, 'Y0': 200.0, 'function_list': [fDict]}
        fsetdesc_from_dict = FunctionSetDescription.dict_to_FunctionSetDescription(fsetDict)
        assert fsetdesc_from_dict == fsetdesc_correct

    def test_DictToFunctionSet_withParamLimits(self):
        ref_p1 = ParameterDescription("PA", 0.0, fixed=True)
        ref_p2 = ParameterDescription("ell", 0.5, limits=[0.1, 0.8])
        ref_p3 = ParameterDescription("I_0", 100.0, limits=[10.0, 1e3])
        ref_p4 = ParameterDescription("sigma", 10.0, limits=[5.0, 20.0])
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)
        ref_x0_p = ParameterDescription("X0", 100.0, limits=[90.0,110.0])
        ref_y0_p = ParameterDescription("Y0", 200.0, limits=[180.0,220.0])
        fsetdesc_correct = FunctionSetDescription(None, ref_x0_p, ref_y0_p, [fdesc_correct])

        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1,0.8], 'I_0': [100.0, 10.0,1e3],
             'sigma': [10.0, 5.0,20.0]}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict = {'X0': [100.0, 90.0,110.0], 'Y0': [200.0, 180.0,220.0], 'function_list': [fDict]}
        fsetdesc_from_dict = FunctionSetDescription.dict_to_FunctionSetDescription(fsetDict)
        assert fsetdesc_from_dict == fsetdesc_correct

    def test_getFuncSetAsDict(self):
        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1,0.8], 'I_0': [100.0, 10.0,1e3],
             'sigma': [10.0, 5.0,20.0]}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict_correct = {'X0': [100.0, 90.0,110.0], 'Y0': [200.0, 180.0,220.0], 'function_list': [fDict]}
        # round-trip test
        fsetdesc_from_dict = FunctionSetDescription.dict_to_FunctionSetDescription(fsetDict_correct)
        fsetDict = fsetdesc_from_dict.getFuncSetAsDict()
        assert fsetDict == fsetDict_correct



class TestModelDescription(object):

    def setup_method( self ):
        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, [180.0, 220.0])
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, [0.1,0.8])
        self.p3 = ParameterDescription("I_0", 100.0, [10.0, 1e3])
        self.p4 = ParameterDescription("sigma", 10.0, [5.0,20.0])
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fullParamDescList = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]
        self.fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        self.fsetList = [self.fsetdesc1]
        self.N_PARAMS_CORRECT = 4 + 2

        self.p1_2 = ParameterDescription("I_tot", 1000.0, [0,1e5])
        self.paramDescList2 = [self.p1_2]
        self.fullParamDescList2 = [self.x0_p, self.y0_p, self.p1_2]
        self.fdesc2 = FunctionDescription('PointSource', "nsc", self.paramDescList2)
        self.functionList2 = [self.fdesc2]
        self.fsetdesc2 = FunctionSetDescription('fs1', self.x0_p, self.y0_p, self.functionList2)
        self.fsetList2 = [self.fsetdesc2]
        self.N_PARAMS_CORRECT2 = 1 + 2

    def test_ModelDescription_simple( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        assert modeldesc1.functionSetIndices() == [0]
        assert modeldesc1.functionNameList() == ['Gaussian']
        assert modeldesc1.parameterList() == self.fullParamDescList
        assert modeldesc1.nParameters == self.N_PARAMS_CORRECT
        assert modeldesc1.hasPointSources is False

    def test_ModelDescription_simple_hasPointSources( self ):
        modeldesc2 = ModelDescription(self.fsetList2)
        assert modeldesc2.functionSetIndices() == [0]
        assert modeldesc2.functionNameList() == ['PointSource']
        assert modeldesc2.parameterList() == self.fullParamDescList2
        assert modeldesc2.nParameters == self.N_PARAMS_CORRECT2
        assert modeldesc2.hasPointSources is True

    def test_ModelDescription_getParamLimits( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        pLimits = modeldesc1.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

    def testModelDescription_updateOptions( self ):
        modeldesc1 = ModelDescription(self.fsetList)

        assert {} == modeldesc1.optionsDict
        optionsDict1 = {"GAIN": 10.5, "READNOISE": 0.9, "ORIGINAL_SKY": 45.01}
        modeldesc1.updateOptions(optionsDict1)
        assert optionsDict1 == modeldesc1.optionsDict
        # the following should *replace* the old GAIN and READNOISE values,
        # but leave the ORIGINAL_SKY value unchanged
        optionsDict2 = {"GAIN": 4.5, "READNOISE": 0.95}
        modeldesc1.updateOptions(optionsDict2)
        refDict = {"GAIN": 4.5, "READNOISE": 0.95, "ORIGINAL_SKY": 45.01}
        assert refDict == modeldesc1.optionsDict

    def testModelDescription_replaceOtions( self ):
        modeldesc1 = ModelDescription(self.fsetList)

        assert {} == modeldesc1.optionsDict
        optionsDict1 = {"GAIN": 10.5, "READNOISE": 0.9, "ORIGINAL_SKY": 45.01}
        modeldesc1.replaceOptions(optionsDict1)
        assert optionsDict1 == modeldesc1.optionsDict
        optionsDict2 = {"GAIN": 4.5, "READNOISE": 0.9}
        modeldesc1.replaceOptions(optionsDict2)
        assert optionsDict2 == modeldesc1.optionsDict

    def test_ModelDescription_load_from_file( self ):
        x0_p = ParameterDescription("X0", 129.0, [125,135])
        y0_p = ParameterDescription("Y0", 129.0, [125,135])
        p1 = ParameterDescription("PA", 18.0, [0,90])
        p2 = ParameterDescription("ell", 0.2, [0.0,1])
        p3 = ParameterDescription("I_0", 100.0, [0, 500])
        p4 = ParameterDescription("h", 25, [0,100])
        paramDescList = [p1, p2, p3, p4]
        fullParamDescList = [x0_p, y0_p, p1, p2, p3, p4]
        fdesc = FunctionDescription('Exponential', "blob", paramDescList)

        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        assert modeldesc.functionSetIndices() == [0]
        assert modeldesc.functionNameList() == ['Exponential']
        assert modeldesc.functionSetNameList() == [['Exponential']]
        assert modeldesc.parameterList() == fullParamDescList
        assert modeldesc.hasPointSources is False

        input_params_correct = np.array([129.0,129.0, 18.0,0.2,100.0,25.0])
        assert_allclose(modeldesc.getRawParameters(), input_params_correct)

    def test_ModelDescription_load_from_file_2sets( self ):
        modeldesc2sets = ModelDescription.load(CONFIG_EXAMPLE_2SETS)
        assert modeldesc2sets.functionNameList() == ['Gaussian', 'Gaussian', 'FlatSky']
        assert modeldesc2sets.functionSetNameList() == [['Gaussian'], ['Gaussian', 'FlatSky']]
        assert modeldesc2sets.hasPointSources is False

    def test_ModelDescription_getStrings(self):
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)

        x0_p = ParameterDescription("X0", 129.0, [125,135])
        y0_p = ParameterDescription("Y0", 129.0, [125,135])
        p1 = ParameterDescription("PA", 18.0, [0,90])
        p2 = ParameterDescription("ell", 0.2, [0.0,1])
        p3 = ParameterDescription("I_0", 100.0, [0, 500])
        p4 = ParameterDescription("h", 25, [0,100])
        paramDescList = [p1, p2, p3, p4]
        fullParamDescList = [x0_p, y0_p, p1, p2, p3, p4]

        assert modeldesc.functionSetIndices() == [0]
        assert modeldesc.functionNameList() == ['Exponential']
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
        # use OrderedDict to ensure options are entered in same order across different versions of Python
        optionsDict = OrderedDict()
        optionsDict["GAIN"] = 4.5
        optionsDict["READNOISE"] = 0.9
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

    def test_ModelDescription_numberedParameterNames(self):
        modeldesc1 = ModelDescription.load(CONFIG_EXAMPLE_EXPONENTIAL)
        paramNames_correct = ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "h_1"]
        assert modeldesc1.numberedParameterNames == paramNames_correct

        modeldesc2 = ModelDescription.load(CONFIG_EXAMPLE_2SETS)
        paramNames_correct2 = ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "sigma_1",
                              "X0_2", "Y0_2", "PA_2", "ell_2", "I_0_2", "sigma_2", "I_0_3"]
        assert modeldesc2.numberedParameterNames == paramNames_correct2

    def test_DictToModel_noParamLimits(self):
        # start out with parameters having no limits
        ref_p1 = ParameterDescription("PA", 0.0)
        ref_p2 = ParameterDescription("ell", 0.5)
        ref_p3 = ParameterDescription("I_0", 100.0)
        ref_p4 = ParameterDescription("sigma", 10.0)
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)
        ref_x0_p = ParameterDescription("X0", 100.0)
        ref_y0_p = ParameterDescription("Y0", 200.0)
        fsetdesc_correct = FunctionSetDescription(None, ref_x0_p, ref_y0_p, [fdesc_correct])
        fsetList = [fsetdesc_correct]
        modeldesc_no_options_correct = ModelDescription(fsetList)
        optionsDict = {"GAIN": 2.0, "ORIGINAL_SKY": 101.0}
        modeldesc_with_options_correct = ModelDescription(fsetList, optionsDict)

        p = {'PA': 0.0, 'ell': 0.5, 'I_0': 100.0, 'sigma': 10.0}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict = {'X0': 100.0, 'Y0': 200.0, 'function_list': [fDict]}
        modelDict = {"function_sets": [fsetDict]}
        modeldesc_from_dict = ModelDescription.dict_to_ModelDescription(modelDict)
        assert modeldesc_from_dict == modeldesc_no_options_correct
        modelDict_with_options = {"function_sets": [fsetDict], "options": optionsDict}
        modeldesc_from_dict = ModelDescription.dict_to_ModelDescription(modelDict_with_options)
        assert modeldesc_from_dict == modeldesc_with_options_correct

    def test_DictToModel_withParamLimits(self):
        ref_p1 = ParameterDescription("PA", 0.0, fixed=True)
        ref_p2 = ParameterDescription("ell", 0.5, limits=[0.1, 0.8])
        ref_p3 = ParameterDescription("I_0", 100.0, limits=[10.0, 1e3])
        ref_p4 = ParameterDescription("sigma", 10.0, limits=[5.0, 20.0])
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)
        ref_x0_p = ParameterDescription("X0", 100.0, limits=[90.0,110.0])
        ref_y0_p = ParameterDescription("Y0", 200.0, limits=[180.0,220.0])
        fsetdesc_correct = FunctionSetDescription("", ref_x0_p, ref_y0_p, [fdesc_correct])
        fsetList = [fsetdesc_correct]
        modeldesc_no_options_correct = ModelDescription(fsetList)
        optionsDict = {"GAIN": 2.0, "ORIGINAL_SKY": 101.0}
        modeldesc_with_options_correct = ModelDescription(fsetList, optionsDict)

        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1,0.8], 'I_0': [100.0, 10.0,1e3],
             'sigma': [10.0, 5.0,20.0]}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict = {'X0': [100.0, 90.0,110.0], 'Y0': [200.0, 180.0,220.0], 'function_list': [fDict]}
        modelDict = {"function_sets": [fsetDict]}
        modeldesc_from_dict = ModelDescription.dict_to_ModelDescription(modelDict)
        assert modeldesc_from_dict == modeldesc_no_options_correct
        modelDict_with_options = {"function_sets": [fsetDict], "options": optionsDict}
        modeldesc_from_dict = ModelDescription.dict_to_ModelDescription(modelDict_with_options)
        assert modeldesc_from_dict == modeldesc_with_options_correct

    def test_getModelAsDict(self):
        p = {'PA': [0.0, "fixed"], 'ell': [0.5, 0.1, 0.8], 'I_0': [100.0, 10.0, 1e3],
             'sigma': [10.0, 5.0, 20.0]}
        fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}
        fsetDict = {'X0': [100.0, 90.0, 110.0], 'Y0': [200.0, 180.0, 220.0], 'function_list': [fDict]}
        modelDict_correct = {"function_sets": [fsetDict]}

        ref_p1 = ParameterDescription("PA", 0.0, fixed=True)
        ref_p2 = ParameterDescription("ell", 0.5, limits=[0.1, 0.8])
        ref_p3 = ParameterDescription("I_0", 100.0, limits=[10.0, 1e3])
        ref_p4 = ParameterDescription("sigma", 10.0, limits=[5.0, 20.0])
        ref_paramDescList = [ref_p1, ref_p2, ref_p3, ref_p4]
        fdesc_correct = FunctionDescription('Gaussian', "blob", ref_paramDescList)
        ref_x0_p = ParameterDescription("X0", 100.0, limits=[90.0,110.0])
        ref_y0_p = ParameterDescription("Y0", 200.0, limits=[180.0,220.0])
        fsetdesc_correct = FunctionSetDescription("", ref_x0_p, ref_y0_p, [fdesc_correct])
        fsetList = [fsetdesc_correct]
        modeldesc_no_options_correct = ModelDescription(fsetList)

        modelDict = modeldesc_no_options_correct.getModelAsDict()
        assert modelDict == modelDict_correct

        # round-trip test
        modeldesc_from_dict = ModelDescription.dict_to_ModelDescription(modelDict_correct)
        modelDict = modeldesc_from_dict.getModelAsDict()
        assert modelDict == modelDict_correct

    def test_functionSetIndices(self):
        # test that we generate correct functionSetIndices when there are > 1 function sets, some
        # with multiple functions
        modeldesc = ModelDescription.load(CONFIG_EXAMPLE_MULTISET)
        assert modeldesc.functionSetIndices() == [0, 1, 3, 6]



class TestSimpleModelDescription(object):

    def setup_method( self ):
        self.x0_p = ParameterDescription("X0", 100.0, fixed=True)
        self.y0_p = ParameterDescription("Y0", 200.0, [180.0, 220.0])
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, [0.1,0.8])
        self.p3 = ParameterDescription("I_0", 100.0, [10.0, 1e3])
        self.p4 = ParameterDescription("sigma", 10.0, [5.0,20.0])
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]
        self.fullParamDescList = [self.x0_p, self.y0_p, self.p1, self.p2, self.p3, self.p4]
        self.fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        self.functionList = [self.fdesc1]
        self.fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        self.fsetList = [self.fsetdesc1]
        # bad example: 2 function sets
        self.fsetdesc2 = FunctionSetDescription('fs1', self.x0_p, self.y0_p, self.functionList)
        self.fsetList_bad = [self.fsetdesc1, self.fsetdesc2]
        self.N_PARAMS_CORRECT = 4 + 2

        self.p1_2 = ParameterDescription("I_tot", 1000.0, [0,1e5])
        self.paramDescList2 = [self.p1_2]
        self.fullParamDescList2 = [self.x0_p, self.y0_p, self.p1_2]
        self.fdesc2 = FunctionDescription('PointSource', "nsc", self.paramDescList2)
        self.functionList2 = [self.fdesc2]
        self.fsetdesc2 = FunctionSetDescription('fs1', self.x0_p, self.y0_p, self.functionList2)
        self.fsetList2 = [self.fsetdesc2]
        self.N_PARAMS_CORRECT2 = 1 + 2

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
        assert simplemodeldesc.functionNameList() == ['Gaussian']
        assert simplemodeldesc.hasPointSources is False

        pLimits = simplemodeldesc.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

        assert simplemodeldesc.nParameters == self.N_PARAMS_CORRECT

    def test_SimpleModelDescription_simple_hasPointSources( self ):
        modeldesc2 = SimpleModelDescription(self.fsetList2)
        assert modeldesc2.functionSetIndices() == [0]
        assert modeldesc2.functionNameList() == ['PointSource']
        assert modeldesc2.parameterList() == self.fullParamDescList2
        assert modeldesc2.nParameters == self.N_PARAMS_CORRECT2
        assert modeldesc2.hasPointSources is True

    def test_SimpleModelDescription_from_functionSet( self ):
        simplemodeldesc = SimpleModelDescription(self.fsetdesc1)
        print(dir(simplemodeldesc))
        assert simplemodeldesc.x0 == self.x0_p
        assert simplemodeldesc.y0 == self.y0_p

        assert simplemodeldesc.functionSetIndices() == [0]
        assert simplemodeldesc.functionNameList() == ['Gaussian']

        pLimits = simplemodeldesc.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

        assert simplemodeldesc.nParameters == self.N_PARAMS_CORRECT

    def test_SimpleModelDescription_from_functionNameList( self ):
        simplemodeldesc = SimpleModelDescription(self.fsetList)
        print(dir(simplemodeldesc))
        assert simplemodeldesc.x0 == self.x0_p
        assert simplemodeldesc.y0 == self.y0_p

        assert simplemodeldesc.functionSetIndices() == [0]
        assert simplemodeldesc.functionNameList() == ['Gaussian']

        pLimits = simplemodeldesc.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]

        assert simplemodeldesc.nParameters == self.N_PARAMS_CORRECT

    def testSimpleModelDescription_add_Function_to_Empty(self):
        simplemodeldesc = SimpleModelDescription()
        simplemodeldesc.addFunction(self.fdesc1)
        simplemodeldesc.x0.setValue(100, fixed=True)
        simplemodeldesc.y0.setValue(200, limits=[180, 220])
        assert simplemodeldesc.x0 == self.x0_p
        assert simplemodeldesc.y0 == self.y0_p
        assert simplemodeldesc.functionSetIndices() == [0]
        assert simplemodeldesc.functionNameList() == ['Gaussian']
        pLimits = simplemodeldesc.getParameterLimits()
        assert pLimits == [None,(180.0,220.0), None, (0.1,0.8), (10.0,1e3), (5.0,20.0)]
        assert simplemodeldesc.nParameters == self.N_PARAMS_CORRECT

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



