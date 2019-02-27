"""
Created on Sep 23, 2013

@author: andre
"""
from pyimfit import FunctionSetDescription, ModelDescription
from pyimfit import make_imfit_function


def example_model_description():
    fs = FunctionSetDescription('example')
    fs.x0.setValue(36.0, [25, 45])
    fs.y0.setValue(32.0, [25, 45])
    
    sersic = make_imfit_function('Sersic')
    sersic.PA.setValue(93.0217, [0, 180])
    sersic.ell.setValue(0.37666, [0, 1])
    sersic.n.setValue(4, fixed=True)
    sersic.I_e.setValue(1, [0, 10])
    sersic.r_e.setValue(25, [0, 100])
    
    exponential = make_imfit_function('Exponential')
    exponential.PA.setValue(93.0217, [0, 180])
    exponential.ell.setValue(0.37666, [0, 1])
    exponential.I_0.setValue(1, [0, 10])
    exponential.h.setValue(25, [0, 100])
    
    fs.addFunction(sersic)
    fs.addFunction(exponential)
    return ModelDescription([fs])
    

def test_model():
    desc = example_model_description()
    print(desc)
    print('I_e = %f' % desc.example.Sersic.I_e.value)
    print('r_e = %f' % desc.example.Sersic.r_e.value)
    print('I_0 = %f' % desc.example.Exponential.I_0.value)
    print('h = %f' % desc.example.Exponential.h.value)
    
    
if __name__ == '__main__':
    test_model()
    
