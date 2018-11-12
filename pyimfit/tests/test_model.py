# Test code for model.py module of pyimfit
# Execute via
#    $ pytest test_model.py

import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose

from ..model import ParameterDescription, FunctionDescription, FunctionSetDescription
from ..model import ModelDescription, SimpleModelDescription

CONFIG_EXAMPLE_EXPONENTIAL = "../data/config_exponential_ic3478_256.dat"



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



class TestFunctionDescription(object):

    def setup_method( self ):
        self.p1 = ParameterDescription("PA", 0.0, fixed=True)
        self.p2 = ParameterDescription("ell", 0.5, 0.1,0.8)
        self.p3 = ParameterDescription("I_0", 100.0, 10.0, 1e3)
        self.p4 = ParameterDescription("sigma", 10.0, 5.0,20.0)
        self.paramDescList = [self.p1, self.p2, self.p3, self.p4]

    def test_FunctionDescription_simple( self ):
        fdesc1 = FunctionDescription('Gaussian', "blob", self.paramDescList)
        assert fdesc1.funcType == "Gaussian"
        assert fdesc1.name == "blob"
        assert fdesc1.PA == self.p1
        assert fdesc1.ell == self.p2
        assert fdesc1.I_0 == self.p3
        assert fdesc1.sigma == self.p4
        plist = fdesc1.parameterList()
        assert plist == self.paramDescList



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

    def test_FunctionSetDescription_catchErrors( self ):
        fsetdesc1 = FunctionSetDescription('fs0', self.x0_p, self.y0_p, self.functionList)
        # do we catch the case of trying to add a non-FunctionDescription object?
        with pytest.raises(ValueError):
            fsetdesc1.addFunction(self.p1)
        # do we catch the case of trying to add a duplicate function?
        with pytest.raises(KeyError):
            fsetdesc1.addFunction(self.fdesc1)



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

    def test_ModelSimpleDescription_simple( self ):
        modeldesc1 = ModelDescription(self.fsetList)
        simplemodeldesc = SimpleModelDescription(modeldesc1)
        print(dir(simplemodeldesc))
        # NOTE: the following does NOT work!
        #assert simplemodeldesc.name == "fs0"
        # properties of SimpleModelDescription
        assert simplemodeldesc.x0 == self.x0_p
        assert simplemodeldesc.y0 == self.y0_p
        # other things
# 		assert simplemodeldesc.PA == self.p1



