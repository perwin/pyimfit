# Cython implementation file for wrapping Imfit code, by PE; based on code
# by Andre Luis de Amorim.
# Copyright André Luiz de Amorim, 2013; Peter Erwin, 2018-2023.

# Note that we are using "typed memoryviews" to translate numpy arrays into
# C-style double * arrays; this apparently the preferred (newer, more flexible)
# Cython approach (versus the older np.ndarray[np.float64_t, ndim=2] syntax that
# Andre's code uses).
# http://docs.cython.org/en/latest/src/userguide/memoryviews.html

# cython: language_level=3

# the following is so we can use Cython decorators
cimport cython

cimport imfit_lib
from .imfit_lib cimport AIC_corrected, BIC
from .imfit_lib cimport AddFunctions, GetFunctionNames, mp_par, mp_result
from .imfit_lib cimport Convolver, ModelObject, SolverResults, DispatchToSolver
from .imfit_lib cimport GetFunctionParameterNames
from .imfit_lib cimport BootstrapErrorsArrayOnly
from .imfit_lib cimport PsfOversamplingInfo
from .imfit_lib cimport MASK_ZERO_IS_GOOD, MASK_ZERO_IS_BAD
from .imfit_lib cimport WEIGHTS_ARE_SIGMAS, WEIGHTS_ARE_VARIANCES, WEIGHTS_ARE_WEIGHTS
from .imfit_lib cimport NO_FITTING, MPFIT_SOLVER, DIFF_EVOLN_SOLVER, NMSIMPLEX_SOLVER

# from local pure-Python module
from .descriptions import ModelDescription, FunctionDescription, ParameterDescription

import sys
import numpy as np
cimport numpy as np
from copy import deepcopy

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy


# code snippet to check for system byte order
sys_byteorder = ('>', '<')[sys.byteorder == 'little']

# convert an ndarray to local system byte order, if it's not already
def FixByteOrder( array ):
    """
    Converts an ndarray to local system byte order, if it's not already

    Parameeters
    -----------
    array : numpy ndarray
        Input numpy array

    Returns
    -------
    array : numpy ndarray
        The input array, with bytes in system byte order
    """
    if array.dtype.byteorder not in ('=', sys_byteorder):
        array = array.byteswap().newbyteorder(sys_byteorder)
    return array

def FixImage( array ):
    """
    Checks an input numpy array and, if necessary, converts it to
    double-precision floating point, little-endian byte order, with
    contiguous layout, to enable use with Imfit.

    Parameters
    ----------
    array : numpy ndarray

    Returns
    -------
    array : numpy ndarray
        The input array, suitably converted
    """
    if (array.dtype != np.float64):
        array = array.astype(np.float64)
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return FixByteOrder(array)



# FIXME: add generic-NLOpt-solver
solverID_dict = {'LM': MPFIT_SOLVER, 'NM': NMSIMPLEX_SOLVER, 'DE': DIFF_EVOLN_SOLVER}


#   int fixed;        /* 1 = fixed; 0 = free */
#   int limited[2];   /* 1 = low/upper limit; 0 = no limit */
#   double limits[2]; /* lower/upper limit boundary value */

# the following does not work! (produces seg-fault on return)
def NewParamInfo( ):
    cdef mp_par newParamInfo
    newParamInfo.fixed = False
    newParamInfo.limited[0] = 0
    newParamInfo.limited[1] = 0
    newParamInfo.limits[0] = 0.0
    newParamInfo.limits[1] = 0.0
    return newParamInfo



def get_function_list( ):
    """
    Returns a list of Imfit image-function names (e.g., ["Gaussian", "Exponential", etc.])

    Returns
    -------
    function_list : list of str
        list of Imfit image-function names
    """
    cdef vector[string] funcNameVect
    GetFunctionNames(funcNameVect)
    return [funcName.decode() for funcName in funcNameVect]


def get_function_dict( ):
    """
    Returns a dict mapping each Imfit image-function names to a list of its corresponding
    parameter names.

    Returns
    -------
    function_dict : dict
        dict where image-function names are keys and items are list of function
        parameter names
    """
    cdef int status
    cdef vector[string] parameters
    funcNameList = get_function_list()
    theDict = {}
    for funcName in funcNameList:
        parameters.clear()
        status = GetFunctionParameterNames(funcName.encode(), parameters)
        theDict[funcName] = [paramName.decode() for paramName in parameters]
    return theDict



def make_imfit_function(func_type, label=""):
    """
    Given a string specifying the official name of an Imfit image function
    (e.g., "Sersic", "Sersic_GenEllipse", "ExponentialDisk3D"), this
    returns an instance of FunctionDescription describing that function
    and its parameters (with values all set to 0).

    Parameters
    ----------
    func_type : string
        The function type; must be one of the recognized Imfit image-function names
        (E.g., "Sersic", "BrokenExponential", etc. Use "imfit --list-functions" on
        the command line to get the full list, or FunctionNames in this module.)

    label : string, optional
        Custom identifying label for this instance of this function.
        Example: "disk", "bulge".
        Default: "" (which means no label).

    Returns
    -------
    func_desc : :class:`FunctionDescription`
        Instance of :class:`FunctionDescription`.

    """
    cdef int status
    cdef vector[string] parameters
    # convert string to byte form for use by C++
    status = GetFunctionParameterNames(func_type.encode(), parameters)
    if status < 0:
        msg = 'Function name \"{0}\" is not a recognized Imfit image function.'.format(func_type)
        raise ValueError(msg)
    func_desc = FunctionDescription(func_type, label)
    for paramName in parameters:
        # convert parameter names to Unicode strings
        param_desc = ParameterDescription(paramName.decode('UTF-8'), value=0.0)
        func_desc.addParameter(param_desc)
    return func_desc


def convolve_image( np.ndarray[np.double_t, ndim=2] image not None,
                   np.ndarray[np.double_t, ndim=2] psf not None,
                   int nproc=0, verbose=False, normalizePSF=True ):
    '''
    Convolve an image with a given PSF image.

    Parameters
    ----------
    image : array
        Image to be convolved.

    psf : array
        PSF to apply.

    nproc : int, optional
        Number of threads to use. If ```nproc <= 0``, use all available cores.
        Default: ``0``, use all cores.

    verbose : bool, optional
        Print diagnostic messages.
        Default: ``False`` ,be quiet.

    normalizePSF : bool, optional
        Specifies whether input PSF image should be normalized.

    Returns
    -------
    convolved_image : array
        An array of same shape as ``image`` containing
        the convolved image.

    '''

    cdef Convolver *convolver = new Convolver()

    if not psf.flags['C_CONTIGUOUS']:
        psf = np.ascontiguousarray(psf)	  # Makes a contiguous copy of the numpy array.
    # Cython typed memoryview, pointing to flattened (1D) copy of PSF data
    cdef double[::1] psf_data = psf.flatten()

    convolver.SetupPSF(&psf_data[0], psf.shape[1], psf.shape[0], normalizePSF)

    if nproc >= 0:
        convolver.SetMaxThreads(nproc)

    cdef int image_x, image_y
    image_x, image_y = image.shape[1], image.shape[0]
    convolver.SetupImage(image_x, image_y)
    cdef int debug_level
    if verbose:
        debug_level = 1
    else:
        debug_level = -1
    convolver.DoFullSetup(debug_level, False)

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(psf)	# Makes a contiguous copy of the numpy array.
    # create 1D image vector (copy of input image data)
    # FIXME: check to see if we really need to flatten the image!
    image_1d = image.flatten()
    # Cython typed memoryview, pointing to flattened version of image data
    cdef double[::1] image_data = image_1d

    convolver.ConvolveImage(&image_data[0])

    # restore 1D image vector to numpy 2D format
    convolved_image = image_1d.reshape((image_y, image_x))

    del convolver

    return convolved_image



# cdef class to hold PSF oversampling info, mainly by storing it in a locally
# instantiated PsfOversamplingInfo object.
# WARNING: input psfImage must be processed by FixImage *prior* to using it to instantiate
# an object of this class!
cdef class PsfOversampling( object ):
    cdef PsfOversamplingInfo * _psfOversamplingInfo_ptr
    cdef double[::1] _imageData
    cdef int _nRows, _nCols

    # __cinit__ so we can allocate memory for the pointer to PsfOversamplingInfo
    def __cinit__(self, np.ndarray[np.double_t, ndim=2, mode='c'] psfImage not None,
                 int scale, str regionString, int xOffset, int yOffset,
                 bool doNormalization ):
        print("PsfOversampling: starting initialization...", flush=True)
        self._imageData = psfImage.flatten()
        self._nRows = psfImage.shape[0]
        self._nCols = psfImage.shape[1]
        self._psfOversamplingInfo_ptr = new PsfOversamplingInfo()
        # Note a difference from the C++ implementation: we always specify isUnique = False,
        # to avoid problems de-allocating the Numpy PSF image array
        self._psfOversamplingInfo_ptr.AddPsfPixels(&self._imageData[0], self._nCols,
                                               self._nRows, False)
        self._psfOversamplingInfo_ptr.AddRegionString(regionString.encode())
        self._psfOversamplingInfo_ptr.AddOversamplingScale(scale)
        self._psfOversamplingInfo_ptr.AddImageOffset(xOffset, yOffset)

        print(self._psfOversamplingInfo_ptr.GetNColumns())
        print("PsfOversampling: done.", flush=True)

    def __dealloc__(self):
        print("PsfOversampling: starting __dealloc__.", flush=True)
        if self._psfOversamplingInfo_ptr != NULL:
            del self._psfOversamplingInfo_ptr
        print("PsfOversampling: done with __dealloc__.", flush=True)




cdef class ModelObjectWrapper( object ):

    cdef ModelObject *_model
    cdef vector[mp_par] _paramInfo
    cdef double *_paramVect
    cdef double *_fitErrorsVect
    cdef double * _modelFluxes
    cdef bool _paramLimitsExist
    cdef int _nParams
    cdef int _nFreeParams
    cdef object _modelDescr
    cdef object _parameterList
    cdef int _nPixels, _nRows, _nCols
    cdef SolverResults *_solverResults
    cdef mp_result *_fitResult
    cdef int _fitStatus
    cdef int _nFuncEvals

    cdef double[::1] _imageData
    cdef double[::1] _errorData
    cdef double[::1] _maskData
    cdef double[::1] _psfData
    cdef bool _inputDataLoaded
    cdef bool _finalSetupDone
    cdef bool _modelImageComputed
    cdef bool _fitted
    cdef object _fitMode
    cdef bool _freed


    def __init__( self, object model_descr, int debug_level=0, int verbose_level=-1,
                  bool subsampling=True, np.ndarray[np.double_t, ndim=2, mode='c'] psf=None,
                  bool normalizePSF=True ):
        self._paramLimitsExist = False
        self._paramVect = NULL
        self._fitErrorsVect = NULL
        self._modelFluxes = NULL
        self._solverResults = NULL
        self._model = NULL
        self._fitResult = NULL

        self._inputDataLoaded = False
        self._nRows = self._nCols = self._nPixels = 0
        self._finalSetupDone = False   # have we called ModelObject::FinalSetupForFitting
        self._fitted = False
        self._modelImageComputed = False
        self._fitMode = None
        self._freed = False
        self._fitStatus = 0
        self._nFuncEvals = 0

        if not isinstance(model_descr, ModelDescription):
            raise ValueError('model_descr must be a ModelDescription object.')
        self._modelDescr = model_descr

        self._solverResults = new SolverResults()
        if self._solverResults == NULL:
            raise MemoryError('Could not allocate SolverResults.')

        self._model = new ModelObject()
        if self._model == NULL:
            raise MemoryError('Could not allocate ModelObject.')
        self._model.SetDebugLevel(debug_level)
        self._model.SetVerboseLevel(verbose_level)
        if psf is not None:
            self._psfData = psf.flatten()
            n_rows_psf = psf.shape[0]
            n_cols_psf = psf.shape[1]
            self._model.AddPSFVector(n_cols_psf * n_rows_psf, n_cols_psf, n_rows_psf, &self._psfData[0],
                                     normalizePSF)
        # self._addFunctions(self._modelDescr, subsampling=subsampling, verbose=debug_level>0)
        self._addFunctions(self._modelDescr, subsampling=subsampling, verbose=verbose_level)
        self._paramSetup(self._modelDescr)


    def setMaxThreads(self, int nproc):
        """
        Specifies maximum number of OpenMP threads to use in image computation.

        Parameters
        ----------
        nproc : int
        """
        self._model.SetMaxThreads(nproc)


    def setChunkSize(self, int chunk_size):
        """
        Specifies the chunk size for OpenMP parallel computations (mainly useful for tests).

        Parameters
        ----------
        chunk_size : int
            8 or 10 is probably good value (default internal value is 10)
        """
        self._model.SetOMPChunkSize(chunk_size)


    def _paramSetup( self, object model_descr ):
        """
        Sets up parameter-related info: _nParams, _nFreeParams, _paramVect, parameter-limits info,
        populating them with values from model_descr
        """
        cdef mp_par newParamInfo
        self._parameterList = model_descr.parameterList()
        self._nParams = self._nFreeParams = self._model.GetNParams()
        if self._nParams != len(self._parameterList):
            msg = "Number of input parameters ({0:d}) does not equal ".format(self._parameterList)
            msg += "required number of parameters for specified functions "
            msg += "({0:d}).".format(self._nParams)
            raise Exception(msg)
        self._paramVect = <double *> calloc(self._nParams, sizeof(double))
        if self._paramVect == NULL:
            raise MemoryError('Could not allocate parameter initial values.')
        self._fitErrorsVect = <double *> calloc(self._nParams, sizeof(double))
        if self._fitErrorsVect == NULL:
            raise MemoryError('Could not allocate space for best-fit parameter errors.')

        # Fill parameter info and initial value.
        for i, param in enumerate(self._parameterList):
            if param.fixed:
                newParamInfo.fixed = True
                self._nFreeParams -= 1
            else:
                newParamInfo.fixed = False
            if param.limits is not None:
                newParamInfo.limited[0] = True
                newParamInfo.limited[1] = True
                newParamInfo.limits[0] = param.limits[0]
                newParamInfo.limits[1] = param.limits[1]
                self._paramLimitsExist = True
            else:
                newParamInfo.limited[0] = False
                newParamInfo.limited[1] = False
                newParamInfo.limits[0] = 0.0
                newParamInfo.limits[1] = 0.0
            self._paramVect[i] = param.value
            self._paramInfo.push_back(newParamInfo)


    cdef _addFunctions(self, object model_descr, bool subsampling, int verbose=0):
        cdef int status = 0
        functionNameList = [funcName.encode() for funcName in model_descr.functionNameList()]
        functionLabelList = [funcName.encode() for funcName in model_descr.functionLabelList()]
        status = AddFunctions(self._model, functionNameList, functionLabelList,
                              model_descr.functionSetIndices(), subsampling, verbose)
        if status < 0:
            raise RuntimeError('Failed to add the functions.')


#     def setPSF(self, np.ndarray[np.double_t, ndim=2, mode='c'] psf, bool normalizePSF=True):
#         cdef int n_rows_psf, n_cols_psf
#
#         # FIXME: check that PSF data has correct type, byteorder
#         # Maybe this was called before.
# # 		if self._psfData != NULL:
# # 			free(self._psfData)
# # 		self._psfData = alloc_copy_from_ndarray(psf)
#         # Cython typed memoryview, pointing to flattened (1D) copy of PSF data
#         self._psfData = psf.flatten()
#
#         n_rows_psf = psf.shape[0]
#         n_cols_psf = psf.shape[1]
#         self._model.AddPSFVector(n_cols_psf * n_rows_psf, n_cols_psf, n_rows_psf, &self._psfData[0],
#                                  normalizePSF)


    # The following needs to be cdef so we can directly access the C++ pointer inside the
    # PsfOversamplingInfo object, even if it's just to pass it on to the ModelObject pointer
    cdef addOversamplingInfo(self, PsfOversampling oversamplingInfo):
        self._model.AddOversampledPsfInfo(oversamplingInfo._psfOversamplingInfo_ptr)


    def loadData(self, np.ndarray[np.double_t, ndim=2, mode='c'] image not None,
                 np.ndarray[np.double_t, ndim=2, mode='c'] error,
                 np.ndarray[np.double_t, ndim=2, mode='c'] mask, **kwargs):
        """
        This is where we supply the data image that will be fit. Since the "data model"
        includes things like error model and masking, this is also where we supply
        any error and/or mask images, along with any specifications of which fit statistic
        to use. Finally, this is also where we supply image-description parameter values
        (gain, read noise, original sky, etc.).

        Parameters
        ----------
        image : 2-D ndarray of double
            Image to be fitted.

        error : 2-D ndarray of double, optional
            error/weight image, same shape as ``image``. If not set,
            errors are generated from ``image``. See also the keyword args
            ``use_poisson_mlr``, ``use_cash_statistics``, and ``use_model_for_errors``.

        mask : 2-D ndarray of double, optional
            Array containing the masked pixels; must have the same shape as ``image``.

        Keyword arguments
        -----------------
        n_combined : integer
            Number of images which were averaged to make final image (if counts are average
            or median).
            Default: 1

        exp_time : float
            Exposure time in seconds (only if image is in ADU/sec).
            Default: 1.0

        gain : float
            Image gain (e-/ADU).
            Default: 1.0

        read_noise : float
            Image read noise (Gaussian sigma, in e-).
            Default: 0.0

        original_sky : float
            Original sky background (ADUs) which has already been subtracted from image.
            Default: 0.0

        error_type : string
            Values in ``error`` image should be interpreted as:
                * ``'sigma'`` (default).
                * ``'weight'``.
                * ``'variance'``.

        mask_format : string
            Values in ``mask`` should be interpreted as:
                * ``'zero_is_good'`` (default).
                * ``'zero_is_bad'``.

        psf_oversampling_list : list of PsfOversampling
            List of PsfOversampling objects, describing oversampling regions, PSFs,
            and oversampling scales.

        use_poisson_mlr : boolean
            Use Poisson MLR (maximum-likelihood-ratio) statistic instead of
            chi^2. Takes precedence over ``error``, ``use_model_for_errors`,
            and ``use_cash_statistics``.
            Default: ``False``

        use_cash_statistics : boolean
            Use Cash statistic instead of chi^2 or Poisson MLR. Takes precedence
            over ``error`` and ``use_model_for_errors``.
            Default: ``False``

        use_model_for_errors : boolean
            Use model values (instead of data) to estimate errors for
            chi^2 computation. Takes precedence over ``error``.
            Default: ``False``
        """

        cdef int n_rows, n_cols, n_rows_err, n_cols_err
        cdef int n_pixels

        # Maybe this was called before.
        if self._inputDataLoaded:
            raise RuntimeError('Data already loaded.')
        if self._freed:
            raise RuntimeError('Objects already freed.')

        # kwargs
        cdef int n_combined
        cdef double exp_time
        cdef double gain
        cdef double read_noise
        cdef double original_sky
        cdef int error_type
        cdef int mask_format
        cdef bool use_cash_statistics
        cdef bool use_poisson_MLR
        cdef bool use_model_for_errors
        cdef int i, n

        if 'n_combined' in kwargs:
            n_combined = kwargs['n_combined']
        else:
            n_combined = 1

        if 'exp_time' in kwargs:
            exp_time = kwargs['exp_time']
        else:
            exp_time = 1.0

        if 'gain' in kwargs:
            gain = kwargs['gain']
        else:
            gain = 1.0

        if 'read_noise' in kwargs:
            read_noise = kwargs['read_noise']
        else:
            read_noise = 0.0

        if 'original_sky' in kwargs:
            original_sky = kwargs['original_sky']
        else:
            original_sky = 0.0

        if 'error_type' in kwargs:
            if kwargs['error_type'] == 'sigma':
                error_type = WEIGHTS_ARE_SIGMAS
            elif kwargs['error_type'] == 'variance':
                error_type = WEIGHTS_ARE_VARIANCES
            elif kwargs['error_type'] == 'weight':
                error_type = WEIGHTS_ARE_WEIGHTS
            else:
                raise Exception('Unknown error type: %s' % kwargs['error_type'])
        else:
            error_type = WEIGHTS_ARE_SIGMAS

        if 'mask_format' in kwargs:
            if kwargs['mask_format'] == 'zero_is_good':
                mask_format = MASK_ZERO_IS_GOOD
            elif kwargs['mask_format'] == 'zero_is_bad':
                mask_format = MASK_ZERO_IS_BAD
            else:
                raise Exception('Unknown mask format: %s' % kwargs['mask_format'])
        else:
            mask_format = MASK_ZERO_IS_GOOD

        # select alternate fit statistic
        if 'use_cash_statistics' in kwargs:
            use_cash_statistics = kwargs['use_cash_statistics']
        else:
            use_cash_statistics = False

        if 'use_poisson_mlr' in kwargs:
            use_poisson_MLR = kwargs['use_poisson_mlr']
            use_cash_statistics = False
        else:
            use_poisson_MLR = False

        if 'use_model_for_errors' in kwargs:
            use_model_for_errors = kwargs['use_model_for_errors']
        else:
            use_model_for_errors = False

        # copy the input image data in 1D form
        self._imageData = image.flatten()
        self._nRows = image.shape[0]
        self._nCols = image.shape[1]
        self._nPixels = self._nRows * self._nCols

        self._model.AddImageDataVector(&self._imageData[0], self._nCols, self._nRows)
        self._model.AddImageCharacteristics(gain, read_noise, exp_time, n_combined, original_sky)

        # reminder: PsfOversamplingInfo objects must be added *after* data image!
        if 'psf_oversampling_list' in kwargs:
            psfOversamplingInfoList = kwargs['psf_oversampling_list']
            n = len(psfOversamplingInfoList)
            for i in range(n):
                self.addOversamplingInfo(psfOversamplingInfoList[i])

        if use_poisson_MLR:
            self._model.UsePoissonMLR()
        elif use_cash_statistics:
            self._model.UseCashStatistic()
        else:
            if error is not None:
                # copy the input error image in 1D form
                self._errorData = error.flatten()
                self._model.AddErrorVector(self._nPixels, self._nCols, self._nRows,
                                            &self._errorData[0], error_type)
            elif use_model_for_errors:
                self._model.UseModelErrors()

        if mask is not None:
            # copy the input mask image in 1D form
            self._maskData = mask.flatten()
            success = self._model.AddMaskVector(self._nPixels, self._nCols, self._nRows,
                                                &self._maskData[0], mask_format)
            if success != 0:
                raise Exception('Error adding mask vector, unknown mask format.')

        self._inputDataLoaded = True
        self.doFinalSetup()
        self._finalSetupDone = True


    def setupModelImage(self, shape):
        if self._inputDataLoaded:
            raise Exception('Input data already loaded.')
        self._nRows = shape[0]
        self._nCols = shape[1]
        self._nPixels = self._nRows * self._nCols
        self._model.SetupModelImage(self._nCols, self._nRows)
        print("ModelObjectWrapper: about to call _model.CreateModelImage()...")
        self._model.CreateModelImage(self._paramVect)
        self._inputDataLoaded = True


    def _testCreateModelImage(self, int count=1):
        for _ from 0 <= _ < count:
            self._model.CreateModelImage(self._paramVect)
        self._modelImageComputed = True


    def doFinalSetup(self):
        cdef int status
        status = self._model.FinalSetupForFitting()
        if status < 0:
            raise Exception('Failure in ModelObject::FinalSetupForFitting().')
        self._finalSetupDone = True


    def computeFitStatistic(self, np.ndarray[np.double_t, ndim=1, mode='c'] newParameters ):
        """
        Computes and returns the fit statistic corresponding to the input parameter vector.

        Parameters
        ----------
        newParameters : Numpy ndarray of float64
            the vector of parameter values
        """
        if len(newParameters) != self._nParams:
            msg = "Length of newParameters ({0:d}) is not ".format(len(newParameters))
            msg += "equal to number of parameters in model ({0:d})!".format(self._nParams)
            raise ValueError(msg)
        return self._model.GetFitStatistic(&newParameters[0])


    def fit( self, double ftol=1e-8, int verbose=-1, mode='LM', seed=0 ):
        cdef int solverID
        cdef string nloptSolverName
        # status = self._model.FinalSetupForFitting()
        # if status < 0:
        #     raise Exception('Failure in ModelObject::FinalSetupForFitting().')

        if not self._finalSetupDone:
            self.doFinalSetup()
        solverID = solverID_dict[mode]
        nloptSolverName = b""
        self._fitStatus = DispatchToSolver(solverID, self._nParams, self._nFreeParams,
                                            self._nPixels, self._paramVect, self._paramInfo,
                                            self._model, ftol, self._paramLimitsExist,
                                            verbose, self._solverResults, nloptSolverName, seed)
        if mode == 'LM':
            self._fitResult = self._solverResults.GetMPResults()
            if self._solverResults.ErrorsPresent():
                self._solverResults.GetErrors(self._fitErrorsVect)

        self._fitMode = mode
        self._fitted = True
        self._modelImageComputed = True
        self._nFuncEvals = self._solverResults.GetNFunctionEvals()


    def doBootstrapIterations( self, int nIters, double ftol=1e-8, bool verboseFlag=False,
                               unsigned long seed=0 ):
        # define outputParams array [double **]
        # outputParamArray[i][nSuccessfulIters] = paramsVect[i];

        # Note that BootstrapErrorsArrayOnly expects a pre-allocated array for outputParamArray,
        # so we should pass it a typed memoryview to a numpy array we create here

        cdef int whichFitStatistic
#        cdef bool verboseFlag = False
        shape = (nIters, self._nParams)
        bootstrappedParamsArray = np.zeros(shape, dtype='float64')
        if not bootstrappedParamsArray.flags['C_CONTIGUOUS']:
            bootstrappedParamsArray = np.ascontiguousarray(bootstrappedParamsArray)
        # create 1D version
        bootstrappedParamsArray_1d = bootstrappedParamsArray.flatten()
        # typed memoryview pointing to bootstrappedParamsArray_1d
        cdef double[::1] outputParams = bootstrappedParamsArray_1d

        whichFitStatistic = self._model.WhichFitStatistic()
        nSuccessfulIterations = BootstrapErrorsArrayOnly(self._paramVect, self._paramInfo,
                                    self._paramLimitsExist, self._model, ftol, nIters,
                                    self._nFreeParams, whichFitStatistic, &outputParams[0], seed,
                                    verboseFlag)

        bootstrappedParamsArray = bootstrappedParamsArray_1d.reshape(shape)
        return bootstrappedParamsArray


    def getModelDescription(self):
        model_descr = deepcopy(self._modelDescr)
        for i, p in enumerate(model_descr.parameterList()):
            p.setValue(self._paramVect[i])
        return model_descr


    def getRawParameters(self):
        vals = []
        for i in range(self._nParams):
            vals.append(self._paramVect[i])
        return vals


    def getParameterErrors(self):
        errorVals = []
        for i in range(self._nParams):
            errorVals.append(self._fitErrorsVect[i])
        return errorVals


    def getFitStatistic( self, mode='none' ):
        cdef double fitstat
        if self.fittedLM:
            fitstat = self._fitResult.bestnorm
        else:
            fitstat = self._model.GetFitStatistic(self._paramVect)
            self._modelImageComputed = True
        cdef int n_valid_pix = self._model.GetNValidPixels()
        cdef int deg_free = n_valid_pix - self._nFreeParams

        if mode == 'none':
            return fitstat
        elif mode == 'reduced':
            return fitstat / deg_free
        elif mode == 'AIC':
            return AIC_corrected(fitstat, self._nFreeParams, n_valid_pix, 1)
        elif mode == 'BIC':
            return BIC(fitstat, self._nFreeParams, n_valid_pix, 1)
        else:
            raise Exception('Unknown statistic mode: %s' % mode)


    # FIXME: possibly change this to use typed memoryview?
    # note that we *do* need to *copy* the data pointed to by model_image,
    # since we want to return a self-contained numpy array, and we want it to
    # survive beyond the point where the ModelObject's destructor is called
    # (which will delete the original data pointed to by model_image)
    def getModelImage( self, newParameters=None ):
        cdef double *model_image
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] output_array
        cdef int imsize = self._nPixels * sizeof(double)
        cdef double *parameterArray

        # FIXME: if newParameters is None -- can we be sure ModelObject has parameter values if fit hasn't been done?
        if newParameters is not None:
            parameterArray = <double *> calloc(self._nParams, sizeof(double))
            for i in range(self._nParams):
                parameterArray[i] = newParameters[i]
            self._model.CreateModelImage(parameterArray)
            free(parameterArray)
        model_image = self._model.GetModelImageVector()
        if model_image is NULL:
            raise Exception('Error: model image has not yet been computed.')
        output_array = np.empty((self._nRows, self._nCols), dtype='float64')
        memcpy(&output_array[0,0], model_image, imsize)

        return output_array


    def getModelFluxes( self, newParameters=None, estimationImageSize=5000 ):
        """
        Computes and returns total and individual-function fluxes for the current model
        and current parameter values.

        Parameters
        ----------
        estimationImageSize : int, optional
            width and height of model image for flux estimation

        Returns
        -------
        (totalFlux, individualFluxes) : tuple of (float, ndarray of float)
            totalFlux = total flux of model
            individualFluxes = numpy ndarray of fluxes for each image-function in the
            model
        """
        cdef double totalFlux
        cdef int nFunctions = self._model.GetNFunctions()
        cdef double *parameterArray

        self._modelFluxes = <double *> calloc(nFunctions, sizeof(double))
        if self._modelFluxes is NULL:
            raise Exception('Error: unable to allocate memory for modelFluxes in getModelFluxes')
        if newParameters is not None:
            parameterArray = <double *> calloc(self._nParams, sizeof(double))
            for i in range(self._nParams):
                parameterArray[i] = newParameters[i]
        else:
            parameterArray = self._paramVect
        totalFlux = self._model.FindTotalFluxes(parameterArray, estimationImageSize,
                                                estimationImageSize, self._modelFluxes)
        functionFluxes = [self._modelFluxes[i] for i in range(nFunctions)]
        if newParameters is not None:
            free(parameterArray)
        if self._modelFluxes != NULL:
            free(self._modelFluxes)
        return (totalFlux, np.array(functionFluxes))



    @property
    def imageSizeSet(self):
        return self._nPixels > 0


    @property
    def fittedLM(self):
        return self._fitted and (self._fitMode == 'LM')


    @property
    def nPegged(self):
        if self.fittedLM:
            return self._fitResult.npegged
        else:
            return -1


    @property
    def nIter(self):
        if self.fittedLM:
            return self._fitResult.niter
        else:
            return -1


    @property
    def nFuncEvals(self):
        # this is initialized to 0, and then set to result from doing fit
        return self._nFuncEvals


    @property
    def nFev(self):
        if self.fittedLM:
            return self._fitResult.nfev
        else:
            return 0


    @property
    def nValidPixels(self):
        return self._model.GetNValidPixels()


    @property
    def validPixelFraction(self):
        return self._model.GetNValidPixels() / self._nPixels


    @property
    def fitConverged(self):
        if not self._fitted:
            raise Exception('Not fitted yet.')
        return (self._fitStatus > 0) and (self._fitStatus < 5)


    @property
    def fitError(self):
        if not self._fitted:
            raise Exception('Not fitted yet.')
        return self._fitStatus <= 0


    @property
    def fitTerminated(self):
        if not self._fitted:
            raise Exception('Not fitted yet.')
        # See Imfit/src/mpfit.cpp for magic numbers.
        return self._fitStatus >= 5


    def close(self):
        if self._model != NULL:
            del self._model
        if self._paramVect != NULL:
            free(self._paramVect)
        if self._fitErrorsVect != NULL:
            free(self._fitErrorsVect)
        if self._solverResults != NULL:
            free(self._solverResults)
            # note that this should automatically free self._fitResult, if that
            # was pointing to something allocated by SolverResults
        self._freed = True

