"""
"""

#Modification of Andre's "model.py" (originally created Sep 2013).

from copy import copy, deepcopy
from collections import OrderedDict

from typing import List, Dict, Sequence, Optional, Union

import numpy as np   # type: ignore


__all__ = ['SimpleModelDescription', 'ModelDescription',
           'ParameterDescription', 'FunctionDescription', 'FunctionSetDescription']



class ParameterDescription(object):
    """
    Holds information for a single parameter of an Imfit image function,
    or the X0 or Y0 parameter of a function block/set
    (corresponding to what is encoded in a single parameter line of an
    Imfit configuration file).

    It holds the name, current value (e.g., a suggested initial value),
    possible lower and upper limits for fitting purposes, and whether or
    not the value is to be held fixed during a fit.

    Attributes
    ----------
        name : str
            label of the parameter (e.g., "X0", "sigma")
        value : float
            current value of the parameter
        limits : 2-element tuple of float
            lower and upper limits for parameter when fitting
        fixed : bool
            whether a parameter should be held fixed during fitting

        _name : str
            label of the parameter (e.g., "X0", "sigma")
        _value : float
            current value of the parameter
        _limits : 2-element tuple of float
            lower and upper limits for parameter when fitting
        _fixed : bool
            whether a parameter should be held fixed during fitting

    Methods
    -------
        setValue(value, limits=None, fixed=False)
            Set the value (and limits, fixed state) of the parameter

        setTolerance(tol)
            Set the limits as +/- fraction of parameter value

        setLimitsRel(i1, i2)
            Set the limits as -i1,+i2 relative to parameter value

        setLimits(v1, v2)
            Set the limits directly

        getStringDescription(noLimits=False, error=none)
            Returns a string containing parameter name and value (and optionally limits or
            1-sigma uncertainty)

    """
    def __init__( self, name: str, value: float, limits: Optional[Sequence[float]]=None, fixed=False ):
        self._name = name  #type: str
        self._limits = None  #type: Optional[Sequence[float]]
        self.setValue(value, limits, fixed)


    @property
    def name(self):
        """
        The label of the parameter. Examples: "x0", "I_e".
        """
        return self._name


    @property
    def value(self):
        """
        The value of the parameter.
        """
        return self._value


    @property
    def limits(self):
        """
        The low and upper limits for the parameter, as a tuple.
        """
        return self._limits


    @property
    def fixed(self):
        """
        Whether or not this parameter is to be held fixed, as a bool.
        """
        return self._fixed


    def setValue( self, value: float, limits: Optional[Sequence[float]]=None, fixed: bool=False ):
        """
        Set the value and (optionally) constraints of the parameter.

        Parameters
        ----------
        value : float
            Value of the parameter.

        limits : 2-element sequence of float, optional
            Lower and upper limits of the parameter.
            Default: ``None`` (= no limits).

        fixed : bool, optional
            Flag the parameter as fixed. Default: ``False``.
        """
        if limits is not None:
            try:
                lower_limit, upper_limit = limits
            except TypeError:
                raise ValueError("limits must be None or two-element iterable.")
            # test for valid limits
            if value < lower_limit:
                lower_limit = value
            elif value > upper_limit:
                upper_limit = value
            if lower_limit >= upper_limit:
                raise ValueError("Lower limit must be < upper limit.")
            self._limits = (float(lower_limit), float(upper_limit))

        self._value = float(value)
        self._fixed = fixed


    def setTolerance( self, tol: float ):
        """
        Set the parameter limits using a fractional "tolerance" value, so that the
        lower limit = (1 - `tol`)*value and the upper limit = (1 + `tol`)*value.
        For example, a tolerance of 0.2 for a property of value 1.0 sets the
        limits to [0.8, 1.2].

        Parameters
        ----------
        tol : float
            Fractional offset for lower and upper limits
            Must lie between ``0.0`` and ``1.0``.
        """
        if tol > 1.0 or tol < 0.0:
            raise ValueError('Tolerance must be between 0.0 and 1.0.')
        self._limits = (self._value * (1 - tol), self._value * (1 + tol))


    def setLimitsRel( self, i1: float, i2: float ):
        """
        Set the parameter limits using relative intervals. The limits
        will be [value - i1, value + i2]

        Parameters
        ----------
        i1 : float
            Lower limit interval.

        i2 : float
            Upper limit interval.
        """
        if i1 < 0.0 or i2 < 0.0:
            raise ValueError('Limit intervals must be positive.')
        self.setLimits(self._value - i1, self._value + i2)


    def setLimits( self, v1: float, v2: float ):
        """
        Set the parameter limits using specified values: [v1, v2]

        Parameters
        ----------
        v1 : float
            Lower limit.

        v2 : float
            Upper limit.
        """
        if v1 >= v2:
            raise ValueError('v2 must be larger than v1.')
        if v1 > self._value:
            v1 = self._value
        elif v2 < self._value:
            v2 = self._value
        self._limits = (v1, v2)


    def getStringDescription( self, noLimits=False, error: Optional[float]=None ):
        """
        Returns a string with parameter name, value, limits, suitable for inclusion in
        an imfit/makeimage config file.

        Parameters
        ----------
        noLimits : bool, optional
            if True, then only parameter values (no limits or "fixed" indicators) are output

        error : float, optional
            error on parameter value (e.g., from Levenberg-Marquardt minimization); if supplied
            then no limit info is output, but "# +/- <error>" is appended

        Returns
        -------
        outputString : string
        """
        outputString = "{0}\t\t{1}".format(self.name, self.value)
        if error is not None:
            outputString += "\t\t# +/- {0}".format(error)
        elif noLimits:
            pass
        else:
            if self._fixed:
                outputString += "\t\tfixed"
            elif self.limits is not None:
                outputString += "\t\t{0},{1}".format(self.limits[0], self.limits[1])
        return outputString


    def __eq__(self, rhs):
        if ((self._name == rhs._name) and (self._value == rhs._value)
                and (self._limits == rhs._limits) and (self._fixed == rhs._fixed)):
            return True
        else:
            return False


    def __str__(self):
        return self.getStringDescription()
        # if self._fixed:
        #     return '{0:s}      {1:f}    fixed'.format(self._name, self._value)
        # elif self.limits is not None:
        #     return '{0:s}      {1:f}    {2:f},{3:f}'.format(self._name, self._value, self._limits[0], self._limits[1])
        # else:
        #     return '{0:s}      {1:f}'.format(self._name, self._value)



class FunctionDescription(object):
    """
    Holds information describing a single Imfit image function and its
    associated parameters, including their values and limits.

    This contains the official Imfit image-function name (e.g., "Gaussian",
    "EdgeOnDisk"), an optional label (e.g., "disk", "outer ring"), and a
    list of ParameterDescription objects which describe the parameter names,
    values, and limits (or fixed status) for each of the image function's
    parameters.

    Attributes
    ----------
        label : str
            optional label for this image function (e.g., "NSC", "Outer disk", etc.)

        _funcName : str
            name of the image function (e.g., "Gaussian", "EdgeOnDisk")
        _label : str
            unique (optional) label for this function (e.g., "disk", "nuclear ring")
        _parameters : list of `ParameterDescription`
            the list of `ParameterDescription` objects for the image-function
            parameters
        nParameters : int
            number of parameters for this function

    Methods
    -------
        addParameter(p)
            Add a `ParameterDescription` instance for one of the function's
            parameter

        parameterList()
            Returns a list of the ParameterDescription objects

    """
    def __init__(self, func_name: str, label: Optional[str]="", parameters=None):
        self._funcName = func_name  #type: str
        self._label = label  #type: Optional[str]
        self._parameters = []  #type: List[ParameterDescription]
        self.nParameters = 0
        if parameters is not None:
            for p in parameters:
                self.addParameter(p)


    @property
    def label(self):
        """
        Custom name/label for the function (e.g., "disk", "nuclear ring").
        """
        return self._label


    def addParameter( self, p: ParameterDescription ):
        if not isinstance(p, ParameterDescription):
            raise ValueError('p is not a ParameterDescription object.')
        self._parameters.append(p)
        # add parameter name as an attribute, so we can do things like
        # function_instance.<param_name>
        setattr(self, p.name, p)
        self.nParameters += 1


    def parameterList( self ):
        """
        A list of the parameters (ParameterDescription objects) of this image function.

        Returns
        -------
        param_list : list of :class:`ParameterDescription`
            List of the parameters.
        """
        return [p for p in self._parameters]


    def parameterNameList( self ):
        """
        A list of the parameters names (list of str) of this image function.

        Returns
        -------
        param_list : list of str
            List of the parameter names.
        """
        return [p.name for p in self._parameters]



    def getStringDescription( self, noLimits=False, errors: Optional[Sequence[float]]=None ):
        """
        Returns a list of strings suitable for inclusion in an imfit/makeimage config file.

        Parameters
        ----------
        noLimits : bool, optional
            if True, then only parameter values (no limits or "fixed" indicators) are output

        errors : sequence float, optional
            errors on parameter values (e.g., from Levenberg-Marquardt minimization)

        Returns
        -------
        outputStrings : list of string
            list of newline-terminated strings, starting with function name and followed
            by one string for each parameter, as output by ParameterDescription.getStringDescription
            If errors is supplied, then parameter strings will contain "# +/- <error>" at end
        """
        funcLine = "FUNCTION {0}".format(self._funcName)
        if (self._label is not None) and (self._label != ""):
            funcLine += "   # LABEL {0}".format(self._label)
        outputLines = [funcLine + "\n"]

        for i in range(self.nParameters):
            p = self._parameters[i]
            if errors is None:
                newString = p.getStringDescription(noLimits=noLimits)
            else:
                newString = p.getStringDescription(error=errors[i])
            outputLines.append(newString + "\n")
        return outputLines


    def __eq__( self, rhs ):
        if ((self._funcName == rhs._funcName) and (self._label == rhs._label)
                    and (self._parameters == rhs._parameters)):
            return True
        else:
            return False


    def __str__(self):
        lines = self.getStringDescription()
        return ''.join(lines)
        # lines = []
        # lines.append('FUNCTION {0}   # {1}'.format(self._funcName, self._label))
        # lines.extend(str(p) for p in self._parameters)
        # return '\n'.join(lines)


    def __deepcopy__(self, memo):
        f = FunctionDescription(self._funcName, self._label)
        f._parameters = [copy(p) for p in self._parameters]
        f.nParameters = self.nParameters
        return f

    @classmethod
    def dict_to_FunctionDescription(cls, inputDict):
        """
        This is a convenience method to generate a FunctionDescription object
        from a dict specifying the function.

        Parameters
        ----------
        inputDict : dict
            dict describing the function
            Example:
                p = {'PA': 0.0, 'ell': 0.5, 'I_0': 100.0, 'sigma': 10.0}
                fDict = {'name': "Gaussian", 'label': "blob", 'parameters': p}

        Returns
        -------
        fset : :class:`FunctionDescription`
            The function description.
        """
        funcName = inputDict['name']
        try:
            funcLabel = inputDict['label']
        except KeyError:
            funcLabel = None
        # FIXME: construct list of ParameterDescription objects from inputDict['parameters']
        # entry (itself a dict)
        try:
            paramsObjList = [ParameterDescription(pname, pval)
                             for pname,pval in inputDict['parameters'].items()]
        except KeyError:
            paramsObjList = None
        return FunctionDescription(funcName, funcLabel, paramsObjList)


class FunctionSetDescription(object):
    """
    Holds information describing an image-function block/set: one or more
    Imfit image functions sharing a common (X0,Y0) position on the image.

    This contains the X0 and Y0 coordinates, a list of FunctionDescription
    objects, and name or label for the function set (e.g., "fs0", "star 1",
    "galaxy 5", "offset nucleus", etc.)

    Attributes
    ----------

        name : str
            name for the function set

        _name : str
            name for the function set
        x0 : ParameterDescription
            x-coordinate of the function block/set's center
        y0 : ParameterDescription
            y-coordinate of the function block/set's center
        _functions : list of `FunctionDescription`
            the FunctionDescription objects, one for each image function
        nFunctions : int
            number of functions in the function set
        nParameters : int
            total number of parameters for this function set, including X0 and Y0

    Class methods
    -------------
        dict_to_FunctionSetDescription(dict)
            Returns a new instance of this class based on a dict

    Methods
    -------
        addFunction(f)
            Add a FunctionDescription instance

        functionList()
            Returns a list of the FunctionDescription objects in the function set

        functionNameList()
            Returns a list of names for the image-functions in the function set

        functionLabelList()
            Returns a list of labels for the image-functions in the function set

        parameterList()
            Returns a list of ParameterDescription objects corresponding to
            the function block/set (including X0,Y0)

    """
    def __init__( self, name: str, x0param: Optional[ParameterDescription]=None,
                  y0param: Optional[ParameterDescription]=None,
                  functionList: Optional[List[FunctionDescription]]=None ):
        self._name = name
        if x0param is None:
            self.x0 = ParameterDescription('X0', 0.0)
        else:
            if not isinstance(x0param, ParameterDescription):
                msg = "x0param should be instance of ParameterDescription"
                raise ValueError(msg)
            self.x0 = x0param
        if y0param is None:
            self.y0 = ParameterDescription('Y0', 0.0)
        else:
            if not isinstance(y0param, ParameterDescription):
                msg = "y0param should be instance of ParameterDescription"
                raise ValueError(msg)
            self.y0 = y0param
        self._functions = []  #type: List[FunctionDescription]
        self.nFunctions = 0
        self.nParameters = 2   # X0,Y0
        if functionList is not None:
            for f in functionList:
                self.addFunction(f)
            self.nFunctions = len(functionList)


    @classmethod
    def dict_to_FunctionSetDescription(cls, inputDict: dict):
        """
        This is a convenience method to generate a ModelDescription object
        from a standard Imfit configuration file.

        Parameters
        ----------
        inputDict : dict
            dict describing the function set

        Returns
        -------
        fset : :class:`FunctionSetDescription`
            The function-set description.
        """

        # extract name
        try:
            fsName = inputDict['name']
        except KeyError:
            fsName = 'fs0'
        # extract x0,y0
        x0_p = ParameterDescription("X0", float(inputDict['X0']))
        y0_p = ParameterDescription("Y0", float(inputDict['Y0']))
        # generate list of FunctionDescription objects
        nFuncs = len(inputDict['function_list'])
        funcList = [FunctionDescription.dict_to_FunctionDescription(fdict) for fdict in
                    inputDict['function_list']]
        return FunctionSetDescription(fsName, x0_p, y0_p, funcList)


    @property
    def name(self):
        """
        Custom name/label for the function set.
        """
        return self._name


    def addFunction(self, f: FunctionDescription):
        """
        Add an Imfit image function created using :func:`make_image_function`.

        Parameters
        ----------
        f : :class:`FunctionDescription`.
            Function description to be added to the function set.
        """
        if not isinstance(f, FunctionDescription):
            raise ValueError('func is not a Function object.')
        if (f.label is not None) and (f.label != "") and (self._contains(f.label)):
            raise KeyError('Function with label \"%s\" already exists.' % f.label)
        self._functions.append(f)
        # add function labels as attributes, so we can do things like
        # function_set_instance.<func_name>
        if f._label is not None:
            setattr(self, f._label, f)
        self.nFunctions += 1
        self.nParameters += f.nParameters


    def _contains(self, label: str):
        for f in self._functions:
            if f.label == label:
                return True
        return False


    def functionList(self):
        """
        A list of the FunctionDescription objects making up this function set.

        Returns
        -------
        function_list : list of FunctionDescription
            List of the functions.
        """
        return self._functions


    def functionNameList(self):
        """
        A list of the Imfit image-function names making up this function set.

        Returns
        -------
        function_list : list of str
            List of the function types.
        """
        return [f._funcName for f in self._functions]


    def functionLabelList(self):
        """
        A list of labels for the Imfit image-functions making up this function set.

        Returns
        -------
        function_list : list of str
            List of the function types.
        """
        # FIXME: Get function label
        return [f.label for f in self._functions]


    def parameterList(self):
        """
        A list of all the parameters corresponding to this function set
        (including the X0,Y0 position).

        Returns
        -------
        param_list : list of :class:`ParameterDescription`
            List of the parameters.
        """
        params = []
        params.append(self.x0)
        params.append(self.y0)
        for f in self._functions:
            params.extend(f.parameterList())
        return params


    def getStringDescription( self, noLimits=False, errors: Optional[Sequence[float]]=None ):
        """
        Returns a list of strings suitable for inclusion in an imfit/makeimage config file.

        Parameters
        ----------
        noLimits : bool, optional
            if True, then only parameter values (no limits or "fixed" indicators) are output

        errors : sequence float, optional
            errors on parameter values (e.g., from Levenberg-Marquardt minimization)

        Returns
        -------
        outputStrings : list of string
            list of newline-terminated strings describing the function set.
            If errors is supplied, then parameter strings will contain "# +/- <error>" at end
        """
        # x0,y0
        if errors is not None:
            x0Line = self.x0.getStringDescription(error=errors[0])
            y0Line = self.y0.getStringDescription(error=errors[1])
        else:
            x0Line = self.x0.getStringDescription(noLimits=noLimits)
            y0Line = self.y0.getStringDescription(noLimits=noLimits)
        outputLines = [x0Line + "\n", y0Line + "\n"]
        for i in range(self.nFunctions):
            if errors is not None:
                functionLines = self._functions[i].getStringDescription(errors=errors[2:])
            else:
                functionLines = self._functions[i].getStringDescription(noLimits=noLimits)
            outputLines.extend(functionLines)

        return outputLines


    def __eq__(self, rhs):
        if ((self._name == rhs._name) and (self.x0 == rhs.x0) and (self.y0 == rhs.y0)
                    and (self._functions == rhs._functions)):
            return True
        else:
            return False


    def __str__(self):
        lines = []
        lines.append(str(self.x0))
        lines.append(str(self.y0))
        lines.extend(str(f) for f in self._functions)
        return '\n'.join(lines)


    def __deepcopy__(self, memo):
        fs = FunctionSetDescription(self._name)
        fs.x0 = copy(self.x0)
        fs.y0 = copy(self.y0)
        fs._functions = deepcopy(self._functions, memo)
        fs.nFunctions = self.nFunctions
        return fs



class ModelDescription(object):
    """
    Holds information describing an Imfit model, including image-description
    data.

    The main components are a dict containing image-descriptions parameters
    and their values (e.g., {"GAIN": 4.5, "ORIGINAL_SKY": 325.39} and a list
    of FunctionSetDescription objects, corresponding to the image-function
    sets/blocks making up the model proper.

    Attributes
    ----------
        optionsDict : dict of {str: float}
            dict mapping image-description parameters (e.g., "GAIN") to
            their corresponding values

        options : dict of {str: float}
            dict mapping image-description parameters (e.g., "GAIN") to
            their corresponding values [this is the internal name for the
            property optionsDict]
        _functionSets : list of `FunctionSetDescription`
            the individual image-function blocks/sets making up the model
        nFunctionSets : int
        nParameters : int
            total number of model parameters

    Class methods
    -------------
        load(fname)
            Returns a new instance of this class based on a standard Imfit
            configuration file (`fname`).

    Methods
    -------
        addFunctionSet(fs)
            Add a function block/set to the model description

        addOptions(optionDict)
            Add image-description options via a dict

        functionSetIndices()
            Returns a list of ``int`` specifying the function-set start indices

        functionList()
            Retuns a list of FunctionDescription instances for all the
            image functions in the model

        functionNameList()
            Returns a list of names for the image-functions in the function set

        functionLabelList()
            Returns a list of labels for the image-functions in the function set

        parameterList()
            Returns a list of ParameterDescription instances corresponding
            to all the parameters in the model

    """

    def __init__( self, functionSetsList: Optional[List[FunctionSetDescription]]=None,
                  options: Optional[Dict[str,float]]=None ):
        self.options = OrderedDict()  #type: Dict[str,float]
        if options is not None:
            self.options.update(options)
        self._functionSets = []  #type: List[FunctionSetDescription]
        self.nFunctionSets = 0
        self.nParameters = 0
        if functionSetsList is not None:
            for fs in functionSetsList:
                # note that addFunctionSet will increment nFunctionSets, so we don't need to
                # do that here
                self.addFunctionSet(fs)


    @classmethod
    def load(cls, fileName: str):
        """
        This is a convenience method to generate a ModelDescription object
        from a standard Imfit configuration file.

        Parameters
        ----------
        fileName : string
            Path to the Imfit configuration file.

        Returns
        -------
        model : :class:`ModelDescription`
            The model description.

        See also
        --------
        parse_config_file
        """
        # note that we need to put the "import parse_config_file" here, rather than
        # at the top of the module, to prevent circular-import errors (since config.py
        # depends on definitions in *this* file)
        from .config import parse_config_file

        return parse_config_file(fileName)


    @classmethod
    def dict_to_ModelDescription(cls, inputDict: dict):
        """
        This is a convenience method to generate a ModelDescription object
        from a standard Imfit configuration file.

        Parameters
        ----------
        inputDict : dict
            dict describing the model, with one required entry -- "function_sets" --
            and one optional entry -- "options"
                "function_set_list" : list of dict, each one specifying a function set,
                suitable as input to FunctionSetDescription.dict_to_FunctionSetDescription()

                "options" : dict of {str: float} specifying image-description options
                ("GAIN", "ORIGINAL_SKY", etc.)

        Returns
        -------
        mdesc : :class:`ModelDescription`
            The model description.
        """

        # extract function sets

        fsetDictList = inputDict["function_sets"]
        try:
            optionsDict = inputDict["options"]
        except KeyError:
            optionsDict = None
        fsetList = [ FunctionSetDescription.dict_to_FunctionSetDescription(fsetDict)
                     for fsetDcit in fsetDictList ]
        return ModelDescription(functionSetsList=fsetList, options=optionsDict)


    @property
    def optionsDict(self):
        """
        Image-description options, as a dict
        E.g., {"GAIN": 2.75, "READNOISE": 103.43}
        """
        return self.options


    @property
    def numberedParameterNames(self):
        """
        List of parameter names for the current model, annotated by function number.
        E.g., ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "h_1", ...]
        """
        outputList = []
        nCurrentFunc = 0
        for i_set in range(self.nFunctionSets):
            outputList.append("X0_{0:d}".format(i_set + 1))
            outputList.append("Y0_{0:d}".format(i_set + 1))
            funcSet = self._functionSets[i_set]
            currentFuncList = funcSet.functionList()
            nFuncs = len(currentFuncList)
            for i in range(nFuncs):
                nCurrentFunc += 1
                thisFunc = currentFuncList[i]
                currentFuncParamNames = thisFunc.parameterNameList()
                for name in currentFuncParamNames:
                    outputList.append("{0}_{1:d}".format(name, nCurrentFunc))
        return outputList


    def addFunctionSet(self, fs: FunctionSetDescription):
        """
        Add a function set to the model description.

        Parameters
        ----------
        fs : :class:`FunctionSetDescription`
            Function set description instance.
        """
        if not isinstance(fs, FunctionSetDescription):
            raise ValueError('fs is not a FunctionSet object.')
        if self._contains(fs.name):
            raise KeyError('FunctionSet named %s already exists.' % fs.name)
        self._functionSets.append(fs)
        setattr(self, fs.name, fs)
        self.nFunctionSets += 1
        self.nParameters += fs.nParameters


    def updateOptions( self, optionsDict: Dict[str,float] ):
        """
        Updates the internal image-descriptions dict, replacing current values for keys
        already in the dict and added key-value pairs for keys not already present.

        Parameters
        ----------
        optionDict : dict
        """
        self.options.update(optionsDict)


    def replaceOptions( self, optionsDict: Dict[str,float] ):
        """
        Replaces the current image-descriptions dict. This differs from updateOptions()
        in that it will completely replace the current image-descriptions dict,
        discarding any key-value pairs with keys not in the replacement dict.


        Parameters
        ----------
        optionDict : dict
        """
        self.options = optionsDict


    def _contains(self, name: str):
        for fs in self._functionSets:
            if fs.name == name:
                return True
        return False


    def functionSetIndices(self):
        """
        Returns the indices in the full parameters list corresponding
        to the starts of individual function sets/blocks.
        """
        indices = [0]
        for i in range(self.nFunctionSets - 1):
            functionsThisSet = self._functionSets[i].functionList()
            indices.append(len(functionsThisSet))
        return indices


    def functionList(self):
        """
        List of the FunctionDescription objects making up this model.

        Returns
        -------
        func_list : list of FunctionDescription objects
        """
        functions = []
        for function_set in self._functionSets:
            functions.extend(function_set.functionList())
        return functions


    def functionNameList(self):
        """
        List of names of the image functions making up this model.

        Returns
        -------
        func_list : list of str
            List of the function names.
        """
        functionNames = []
        for function_set in self._functionSets:
            functionNames.extend(function_set.functionNameList())
        return functionNames


    def functionLabelList(self):
        """
        List of labels for the image functions making up this model.

        Returns
        -------
        func_list : list of str
            List of the function labels.
        """
        functionLabels = []
        for function_set in self._functionSets:
            functionLabels.extend(function_set.functionLabelList())
        return functionLabels


    def functionSetNameList(self):
        """
        List of the function sets composing this model, as strings.

        Returns
        -------
        func_set_list : list of list of string
            List of the function sets: [[functions_in_set1], [functions_in_set2], ...]
        """
        functionSetList = []
        for function_set in self._functionSets:
            thisFunctionNameList = function_set.functionNameList()
            functionSetList.append(thisFunctionNameList)
        return functionSetList


    def parameterList(self):
        """
        A list of the parameters (ParameterDescription objects) making up this model.

        Returns
        -------
        param_list : list of :class:`ParameterDescription`
            List of the parameters.
        """
        params = []
        for function_set in self._functionSets:
            params.extend(function_set.parameterList())
        return params


    def getRawParameters(self):
        """
        Returns a Numpy array of the ModelDescription's current parameter values

        Returns
        -------
        paramValues : ndarray of float
        """
        paramsList = self.parameterList()
        return np.array([p.value for p in paramsList])


    def getParameterLimits(self):
        """
        Returns a list containing lower and upper limits for all parameters in the model.

        Returns
        -------
        parameterLimits : list of 2-element tuples of float
            [(lower_limit, upper_limit)_1, (lower_limit, upper_limit)_2, ...]
        """
        paramsList = self.parameterList()
        return [p.limits for p in paramsList]


    def getStringDescription( self, noLimits=False, errors: Optional[Sequence[float]]=None, saveOptions=False ):
        """
        Returns a list of strings suitable for inclusion in an imfit/makeimage config file.

        Parameters
        ----------
        noLimits : bool, optional
            if True, then only parameter values (no limits or "fixed" indicators) are output

        errors : sequence float, optional
            errors on parameter values (e.g., from Levenberg-Marquardt minimization)

        saveOptions : bool, optional
            if False, then image-description options (GAIN, READNOISE, etc.) are *not* output

        Returns
        -------
        outputStrings : list of string
            list of newline-terminated strings describing the model.
            If errors is supplied, then parameter strings will contain "# +/- <error>" at end
        """
        outputLines = []
        # image-description parameters
        if saveOptions and len(self.options) > 0:
            for key,value in self.options.items():
                newLine = "{0}\t\t{1}\n".format(key, value)
                outputLines.append(newLine)

        # function sets
        fblockIndices = self.functionSetIndices()
        for i in range(self.nFunctionSets):
            outputLines.extend("\n")
            fblock = self._functionSets[i]
            fblockStartIndex = fblockIndices[i]
            if errors is not None:
                newLines = fblock.getStringDescription(noLimits=noLimits, errors=errors[fblockStartIndex:])
            else:
                newLines = fblock.getStringDescription(noLimits=noLimits)
            outputLines.extend(newLines)
        return outputLines


    def __eq__(self, rhs):
        if ((self.options == rhs.options) and (self._functionSets == rhs._functionSets)):
            return True
        else:
            return False


    def __str__(self):
        lines = []
        for k, v in list(self.options.items()):
            lines.append('%s	%f' % (k, v))
        lines.extend(str(fs) for fs in self._functionSets)
        return '\n'.join(lines)


    def __deepcopy__(self, memo):
        model = type(self)()
        model._functionSets = deepcopy(self._functionSets, memo)
        model.options = copy(self.options)
        model.nFunctionSets = self.nFunctionSets
        return model




class SimpleModelDescription(ModelDescription):
    """
    Simple version of ModelDescription with only one function set.

    Attributes
    ----------
        x0 : ParameterDescription
            ParameterDescription object for the x-coordinate of the model center

        y0 : ParameterDescription
            ParameterDescription object for the y-coordinate of the model center
    """

    def __init__(self, inst=None):
        """
        inst can be:
            ModelDescription instance
            FunctionSetDescription instance
            list or tuple containing a single FunctionSetDescription instance

        if inst is None, then a minimal object is returned (with a bare-bones
        FunctionSetDescription), to which one can later add functions
        """
        super().__init__()
        if isinstance(inst, ModelDescription):
            if len(inst._functionSets) != 1:
                raise ValueError('Original model must have only one function set.')
            self.addFunctionSet(copy(inst._functionSets[0]))
        elif isinstance(inst, FunctionSetDescription):
            self.addFunctionSet(copy(inst))
        elif isinstance(inst, (list,tuple)):
            if len(inst) == 1:
                if isinstance(inst[0], FunctionSetDescription):
                    self.addFunctionSet(copy(inst[0]))
                else:
                    raise ValueError('Invalid type: %s' % type(inst[0]))
            else:
                raise ValueError('List or tuple argument must have length = 1')
        elif inst is None:
            self.addFunctionSet(FunctionSetDescription('fs'))
        else:
            raise ValueError('Invalid type: %s' % type(inst))


    @property
    def x0(self):
        """
        X coordinate of the center of the model.
        Instance of :class:`ParameterDescription`.
        """
        return self._functionSets[0].x0


    @property
    def y0(self):
        """
        Y coordinate of the center of the model.
        Instance of :class:`ParameterDescription`.
        """
        return self._functionSets[0].y0


    def addFunction(self, f: FunctionDescription):
        """
        Add a function created using :func:`function_description`.

        Parameters
        ----------
        f : :class:`FunctionDescription`.
            Function description to be added to the model.
        """
        self._functionSets[0].addFunction(f)
        self.nParameters += f.nParameters



# 	return [attr] from the first (only, really) FunctionSetDescription
    #    _name; name
    #    x0; y0
    #
    # so simple_model_desc.x0 should --> simple_model_desc.functionSets[0].x0
    def __getattr__(self, attr):
        if len(self._functionSets) == 0:
            return None
        else:
            return self._functionSets[0][attr]
