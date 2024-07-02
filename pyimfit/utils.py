# Code for reading in and analyzing outputs of imfit

from __future__ import division

import glob
from typing import List

import numpy as np   # type: ignore

from . import imfit_funcs as imfuncs

# If pandas is available, use its (much faster) file-reading code
usingPandas = False
try:
    import pandas as pd   # type: ignore
    usingPandas = True
    def GetDataColumns( filename, usecols=None ):
        df = pd.read_csv(filename, delim_whitespace=True, comment='#', dtype=np.float64,
                            usecols=usecols, header=None)
        return df.values
except ImportError:
    def GetDataColumns( filename, usecols=None ):
        return np.loadtxt(filename, usecols=usecols)


# dictionary mapping imfit function short names (as found in the config/parameter file) to
# corresponding 1-D Python functions in imfit_funcs.py, along with some useful information:
#    "function" = corresponding imfit_funcs.py function, if one exists
#    "nSkip" = the number of 2D-related  parameters to skip (e.g., PA, ellipticity),
#    "ell" = index for ellipticity parameter, if it exists,
#    "a" = index or indices for semi-major-axis parameters (r_e, h, sigma, etc.)
# THIS IS NOT MEANT TO BE A COMPLETE LIST
imfitFunctionMap = {
                    "Exponential": {"function": imfuncs.Exponential, "nSkip": 2, "ell": 1, "a": [3]},
                    "Exponential_GenEllipse": {"function": imfuncs.Exponential, "nSkip": 3, "ell": 1, "a": [4]},
                    "Sersic": {"function": imfuncs.Sersic, "nSkip": 2, "ell": 1, "a": [4]},
                    "Sersic_GenEllipse": {"function": imfuncs.Sersic, "nSkip": 3, "ell": 1, "a": [5]},
                    "Gaussian": {"function": imfuncs.Gauss, "nSkip": 2, "ell": 1, "a": [3]},
                    "GaussianRing": {"function": imfuncs.GaussRing, "nSkip": 2, "ell": 1, "a": [3,4]},
                    "GaussianRing2Side": {"function": imfuncs.GaussRing2Side, "nSkip": 2, "ell": 1, "a": [3,4,5]},
                    "Moffat": {"function": imfuncs.Moffat, "nSkip": 2, "ell": 1, "a": [3]},
                    "BrokenExponential": {"function": imfuncs.BrokenExp, "nSkip": 2, "ell": 1, "a": [3,4,5]}
}  #type: Dict[str, Dict]



def ChopComments( theLine: str ) -> str:
    return theLine.split("#")[0]


def GetFunctionImageNames( baseName: str, funcNameList: List[str] ) -> List[str]:
    """Generate a list of FITS filenames as would be created by makeimage in "--output-functions"
    mode.

    Parameters
    ----------
    baseName : str
        root name of output files

    funcNameList : list of str
        list containing function names (e.g., ["Exponential", "Sersic", "Sersic"]

    Returns
    -------
    imageNameList : list of str
        list of output filenames
    """

    nImages = len(funcNameList)
    imageNameList = [ "%s%d_%s.fits" % (baseName, i + 1, funcNameList[i]) for i in range(nImages) ]
    return imageNameList


def ReadImfitConfigFile( fileName: str, minorAxis=False, pix=1.0, getNames=False, X0=0.0 ):
    """Function to read and parse an imfit-generated parameter file (or input config file)
    and return a tuple consisting of: (list of 1-D imfit_funcs functions, list of lists of parameters).

    pix = scale in arcsec/pixel, if desired for plotting vs radii in arcsec.

    We assume that all functions have a center at x = 0; this can be changed by setting
    X0.

    Returns tuple of (functionList, trimmedParameterList)
    If getNames == True:
        Returns tuple of (functionNameList, functionList, trimmedParameterList)

    Parameters
    ----------
    fileName : str
        Imfit configuration or best-fit parameter file

    minorAxis : bool, optional

    pix : float, optional
        scale in arcsec/pixel, if desired for plotting radii in arcsec

    getNames : bool, optional
        if True, output is tuple of (functionNameList, functionList, trimmedParameterList)

    X0 : float, optional
        value for function centers

    Returns
    -------
    (functionList, trimmedParameterList) : (list of image-functions, list of float)
    OR
    (functionNameList, functionList, trimmedParameterList) : (list of str, list of image-functions, list of float)
    """

    dlines = [ line for line in open(fileName) if len(line.strip()) > 0 and line[0] != "#" ]

    funcNameList = []
    paramMetaList = []
    currentParamList = []  #type: List[float]
    for line in dlines:
        trimmedLine = ChopComments(line)
        if trimmedLine.find("X0") == 0:
            continue
        if trimmedLine.find("Y0") == 0:
            continue
        if trimmedLine.find("FUNCTION") == 0:
            # if this isn't the first function, store the previous set of parameters
            if len(currentParamList) > 0:
                paramMetaList.append(currentParamList)
            # make a new parameter list for the new function
            currentParamList = [X0]
            pp = trimmedLine.split()
            fname = pp[1].strip()
            funcNameList.append(fname)
            continue
        else:
            pp = trimmedLine.split()
            newValue = float(pp[1])
            currentParamList.append(newValue)

    # ensure that final set of parameters get stored:
    paramMetaList.append(currentParamList)

    # process function list to remove unneeded parameters (and convert size measures
    # from major-axis to minor-axis, if requested)
    funcList = [ imfitFunctionMap[fname]["function"] for fname in funcNameList ]
    trimmedParamList = []
    nFuncs = len(funcList)
    for i in range(nFuncs):
        fname = funcNameList[i]
        nSkipParams = imfitFunctionMap[fname]["nSkip"]
        fullParams = paramMetaList[i]
        # calculate scaling factor for minor-axis values, if needed
        if minorAxis:
            ellIndex = imfitFunctionMap[fname]["ell"]  #type: int
            ell = fullParams[ellIndex + 1]
            q = 1.0 - ell
        else:
            q = 1.0
        smaIndices = imfitFunctionMap[fname]["a"]
        # convert length values to arcsec and/or minor-axis, if needed,
        for smaIndex in smaIndices:
            # +1 to account for X0 value at beginning of parameter list
            fullParams[smaIndex + 1] = pix*q*fullParams[smaIndex + 1]
        # construct the final 1-D parameter set for this function: X0 value, followed
        # by post-2D-shape parameters
        trimmedParams = [fullParams[0]]
        trimmedParams.extend(fullParams[nSkipParams + 1:])
        trimmedParamList.append(trimmedParams)


    if getNames:
        return (funcNameList, funcList, trimmedParamList)
    else:
        return (funcList, trimmedParamList)




# Code for reading output of bootstrap resampling and MCMC chains

def GetBootstrapOutput( filename: str ):
    """Reads imfit's bootstrap-resampling output when saved using the
    --save-bootstrap command-line option.

    Parameters
    ----------
    filename : str
        name of file with bootstrap-resampling output

    Returns
    -------
    (column_names, data_array) : tuple of (list, np.ndarray)
        column_names = list of column names (strings)
        data_array = numpy array of parameter values
            with shape = (n_iterations, n_parameters)
    """

    # get first 100 lines (or all lines in file, whichever is shorter)
    firstLines = []
    with open(filename) as theFile:
        try:
            for i in range(100):
                firstLines.append(next(theFile))
        except StopIteration:
            pass

    # find header line with column names and extract column names
    for i in range(len(firstLines)):
        if firstLines[i].find("# Bootstrap resampling output") >= 0:
            columnNamesIndex = i + 1
            break
    columnNames = firstLines[columnNamesIndex][1:].split()
    for i in range(len(columnNames)):
        if columnNames[i] == "likelihood":
            break

    # get the data
    d = GetDataColumns(filename)

    return (columnNames, d)



def GetSingleChain( filename: str, getAllColumns=False ):
    """Reads a single MCMC chain output file and returns a tuple of column names
    and a numpy array with the data.

    Parameters
    ----------
    filename : str
        name of file with MCMC output chain

    getAllColumns: bool, optional
        if False [default], only model parameter-value columns are retrieved;
        if True, all output columns (including MCMC diagnostics) are retrieved

    Returns
    -------
    (column_names, data_array) : tuple of (list, np.ndarray)
        column_names = list of column names (strings)
        data_array = numpy array of parameter values
            with shape = (n_iterations, n_parameters)
    """

    # get first 100 lines
    # FIXME: file *could* be shorter than 100 lines; really complicated
    # model could have > 100 lines of header...
    with open(filename) as theFile:
        firstLines = [next(theFile) for x in range(100)]

    # find header line with column names and extract column names
    for i in range(len(firstLines)):
        if firstLines[i].find("# Column Headers") >= 0:
            columnNamesIndex = i + 1
            break
    columnNames = firstLines[columnNamesIndex][1:].split()
    for i in range(len(columnNames)):
        if columnNames[i] == "likelihood":
            nParamColumns = i
            break

    # get data for all columns, or just the model parameters?
    whichCols = None
    if not getAllColumns:
        whichCols = list(range(nParamColumns))
        outputColumnNames = columnNames[:nParamColumns]
    else:
        whichCols = None
        outputColumnNames = columnNames

    # get the data
    d = GetDataColumns(filename, usecols=whichCols)

    return (outputColumnNames, d)


def MergeChains( fname_root: str, maxChains=None, getAllColumns=False, start=10000, last=None,
                    secondHalf=False  ):
    """
    Reads and concatenates all MCMC output chains with filenames = fname_root.*.txt,
    using data from t=start onwards. By default, all generations from each chain
    are extracted; this can be modified with the start, last, or secondHalf keywords.


    Parameters
    ----------
    fname_root : str
        root name of output chain files (e.g., "mcmc_out")

    maxChains : int or None, optional
        maximum number of chain files to read [default = None = read all files]

    getAllColumns : bool, optional
        if False [default], only model parameter-value columns are retrieved;
        if True, all output columns (including MCMC diagnostics) are retrieved

    start : int, optional
        extract samples from each chain beginning with time = start
        ignored if "secondHalf" is True or if "last" is not None

    last : int or None, optional
        extract last N samples from each chain
        ignored if "secondHalf" is True

    secondHalf : bool, optional
        if True, only the second half of each chain is extracted
        if False [default],

    Returns
    -------
    (column_names, data_array) : tuple of (list, np.ndarray)
        column_names = list of column names (strings)
        data_array = numpy array of parameter values
            with shape = (n_samples, n_parameters)
    """

    # construct list of filenames
    if maxChains is None:
        globPattern = "{0}.*.txt".format(fname_root)
        filenames = glob.glob(globPattern)
    else:
        filenames = ["{0}.{1}.txt".format(fname_root, n) for n in range(maxChains)]
    nFiles = len(filenames)

    if (nFiles < 1):
        print("ERROR: No MCMC output files found using pattern \"{0}\"".format(globPattern))
        return None

    # get the first chain so we can tell how long the chains are
    (colNames, dd) = GetSingleChain(filenames[0], getAllColumns=getAllColumns)
    nGenerations = dd.shape[0]

    # figure out what part of full chain to extract
    if secondHalf:
        startTime = int(np.floor(nGenerations / 2))
    elif last is not None:
        startTime = -last   #pylint: disable=invalid-unary-operand-type
    else:
        startTime = start

    # get first chain and column names; figure out if we get all columns or just
    # model parameters
    if (startTime >= nGenerations):
        txt = "WARNING: # generations in MCMC chain file {0} ({1:d}) is <= ".format(filenames[0],
                                                                                nGenerations)
        txt += "requested start time ({0:d})!\n".format(startTime)
        print(txt)
        return None
    dd_final = dd[startTime:,:]
    if getAllColumns is False:
        nParamColumns = len(colNames)
        whichCols = list(range(nParamColumns))  #type: Optional[List[int]]
    else:
        whichCols = None

    # get and append rest of chains if more than 1 chain-file was requested
    if nFiles > 1:
        for i in range(1, nFiles):
            dd_next = GetDataColumns(filenames[i], usecols=whichCols)
            dd_final = np.concatenate((dd_final, dd_next[startTime:,:]))

    return (colNames, dd_final)

