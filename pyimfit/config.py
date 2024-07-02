"""
Functions for parsing Imfit configuration files, returning instances of the classes
in descriptions.py

The main useful function is parse_config_file, which returns an instance of the
ModelDescription class.
"""

#Modification of Andre's "config.py" (originally created 19 Sep 2013).

from collections import OrderedDict

from typing import List

from .descriptions import ParameterDescription, FunctionDescription, FunctionSetDescription, ModelDescription

__all__ = ['parse_config_file', 'parse_config']


commentChar = '#'
x0_str = 'X0'
y0_str = 'Y0'
function_str = 'FUNCTION'
fixed_str = 'fixed'

# The following are the currently recognized config-file image options (see
# imfit_main.cpp) and the functions for transforming string values from a
# config file:
# The first is a dict mapping string keywords to Python *functions* -- note that
# mypy will erroneously interpret these as type annotations when run under Python 3.5
# (this is not an actual problem)
recognizedOptions = {"GAIN": float, "READNOISE": float, "EXPTIME": float, "NCOMBINED": int,
                     "ORIGINAL_SKY": float, "NCOLS": int, "NROWS": int}
recognizedOptionNames = list(recognizedOptions.keys())



#FIXME: (PE) Need to handle case of optional image-function parameters


def parse_config_file( fileName: str ) -> ModelDescription:
    """
    Read and parse an Imfit model-configuration file.

    Parameters
    ----------
    fileName : str
        Path to the model configuration file.

    Returns
    -------
    model : :class:`~imfit.ModelDescription`
        A model description object.

    See Imfit documentation for details on the format of configuration files.
    """
    with open(fileName) as fd:
        return parse_config(fd.readlines())




def parse_config( lines: List[str] ) -> ModelDescription:
    """
    Parse an Imfit model configuration from a list of strings.

    Parameters
    ----------
    lines : list of str
        String representantion of Imfit model configuration.

    Returns
    -------
    model : :class:`~imfit.ModelDescription`
        A model description object.

    See also
    --------
    parse_config_file
    """
    lines = clean_lines(lines)

    model = ModelDescription()

    block_start = 0
    functionBlock_id = 0   # number of current function set
    for i in range(block_start, len(lines)):
        if lines[i].startswith(x0_str):
            if block_start == 0:
                options = read_options(lines[block_start:i])
                model.options.update(options)
            else:
                # possible auto-label-generation code
                # funcSetLabel = "fs{0:d}".format(functionBlock_id)
                funcSetLabel = ""
                model.addFunctionSet(read_function_set(funcSetLabel, lines[block_start:i]))
                functionBlock_id += 1
            block_start = i
    # funcSetLabel = "fs{0:d}".format(functionBlock_id)
    funcSetLabel = ""
    model.addFunctionSet(read_function_set(funcSetLabel, lines[block_start:i + 1]))
    return model




def clean_lines( lines: List[str] ) -> List[str]:
    """
    Returns a list of lines = input list of lines, with comments and empty lines
    stripped out (blank lines and lines beginning with '#' are removed; lines
    ending in comments have the comments removed).

    Parameters
    ----------
    lines : list of str

    Returns
    -------
    cleaned_lines : list of str
    """
    cleaned_lines = []
    for line in lines:
        # Clean the comments.
        line = line.split(commentChar, 1)[0]
        # Remove leading and trailing whitespace.
        line = line.strip()
        # Skip the empty lines.
        if line == '':
            continue
        cleaned_lines.append(line)
    return cleaned_lines




def read_options( lines: List[str] ) -> OrderedDict:
    """
    Parse the lines from an Imfit configuration file which contain image-description
    parameters (GAIN, READ_NOISE, etc.).

    Parameters
    ----------
    lines : list of str
        String representantion of Imfit model configuration.

    Returns
    -------
    config : dict
        maps parameter names to numerical values
        e.g, {"GAIN": 4.56, "ORIGINAL_SKY": 233.87}
    """
    config = OrderedDict()  #type: OrderedDict
    for line in lines:
        # Options are key-value pairs.
        pieces = line.split()
        if len(pieces) < 2:
            msg = "Expected image-description parameter and value"
            raise ValueError(msg)
        k, val = pieces[0], pieces[1]
        if k in [x0_str, y0_str, function_str]:
            msg = "Expected image-description parameter name, but got {0:s} instead.".format(k)
            raise ValueError(msg)
        if k in recognizedOptionNames:
            val = recognizedOptions[k](val)
            config[k] = val
        else:
            print("Ignoring unrecognized image-description parameter \"{0}\"".format(k))

    return config




def read_function_set( label: str, lines: List[str] ) -> FunctionSetDescription:
    """
    Reads in lines of text corresponding to a function block (or 'set') containing
    X0,Y0 coords and one or more image functions with associated parameter settings.

    Parameters
    ----------
    label : str
        optional label for the function set (for "no particular label", use "")
    
    lines : list of str
        lines from configuration file
    
    Returns
    -------
    fs : :class:`~imfit.`FunctionSetDescription`
        Contains extracted information about function block, functions, and their parameters
    """
    # A function set starts with X0 and Y0 parameters.
    x0 = read_parameter(lines[0])
    y0 = read_parameter(lines[1])
    if x0.name != x0_str or y0.name != y0_str:
        raise ValueError('A function set must begin with the parameters X0 and Y0.')
    fs = FunctionSetDescription(label)
    fs.x0 = x0
    fs.y0 = y0
    block_start = 2
    for i in range(block_start, len(lines)):
        # Functions for a given set start with FUNCTION.
        if i == block_start or not lines[i].startswith(function_str):
            continue
        fs.addFunction(read_function(lines[block_start:i]))
        block_start = i
    # Add the last function in the set.
    fs.addFunction(read_function(lines[block_start:i + 1]))
    return fs




def read_function( lines: List[str] ) -> FunctionDescription:
    """
    Reads in lines of text corresponding to a function declaration and
    initial values, ranges, etc. for its parameters.

    Parameters
    ----------
    lines : list of str
        lines from configuration file; initial line is of form 'FUNCTION <func-name>'
        with subsequent lines (assumed to be in correct order) describing
        parameter values and (optionally) 'fixed' or lower,upper limits
    
    Returns
    -------
    func : :class:`~imfit.`FunctionDescription`
        Contains extracted information about function and its parameters
    """

    def GetFunctionLabel( theLine ):
        labelText = ""
        if theLine.find("LABEL") > 0:
            labelText = theLine.split("LABEL")[1].strip()
        return labelText

    # First line contains the function name, and optionally the label
    pieces = lines[0].split()
    test_function_str = pieces[0]
    if test_function_str != function_str:
        raise ValueError('Function definition must begin with FUNCTION.')
    if len(pieces) <= 1:
        raise ValueError('No function name was supplied.')
    name  = pieces[1].strip()
    label = GetFunctionLabel(lines[0])
    func = FunctionDescription(name, label)
    # Read the function parameters.
    for i in range(1, len(lines)):
        func.addParameter(read_parameter(lines[i]))

    # FIXME: check function and parameters.
    return func




def read_parameter( line: str ) -> ParameterDescription:
    """
    Reads in a single text line containing parameter info, parses it, and
    returns a ParameterDescription object with the parameter info.

    Parameters
    ----------
    line : str
        line from configuration file specifying parameter name, initial value,
        and (optionally) 'fixed' or lower,upper limits
    
    Returns
    -------
    :class:`~imfit.ParameterDescription` instance
        Contains extracted information about parameter
        (name, value, possible limits or fixed state)
    """
    limits = None
    fixed = False

    # Format:
    # PAR_NAME	  VALUE	  [ "fixed" | LLIMIT,ULIMIT ] [# comments]
    goodLine = line.split("#")[0]
    pieces = goodLine.split()
    if len(pieces) <= 1:
        raise ValueError("Line must have at least two elements (parameter name, value).")
    name = pieces[0]
    value = float(pieces[1])
    if len(pieces) > 2:
        predicate = pieces[2]  #type: str
        if predicate == fixed_str:
            fixed = True

        elif ',' in predicate:
            llimit_str, ulimit_str = predicate.split(',')
            llimit = float(llimit_str)
            ulimit = float(ulimit_str)
            if llimit > ulimit:
                raise ValueError('lower limit ({0:f}) is larger than upper limit ({1:f})'.format(llimit, ulimit))
            limits = [llimit, ulimit]
        else:
            raise ValueError("Malformed limits on parameter line.")

    return ParameterDescription(name, value, limits, fixed)
