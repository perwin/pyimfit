# Cython implementation file for wrapping Imfit code.

# Note that we are using "typed memoryviews" to translate numpy arrays into
# C-style double * arrays; this apparently the preferred (newer, more flexible)
# Cython approach (versus the older np.ndarray[np.float64_t, ndim=2] syntax that
# Andre's code uses).
# http://docs.cython.org/en/latest/src/userguide/memoryviews.html

from __future__ import print_function

# the following is so we can use Cython decorators
cimport cython

cimport imfit_lib
from .imfit_lib cimport Mean, StandardDeviation, AIC_corrected, BIC
from .imfit_lib cimport AddFunctions, GetFunctionNames, mp_par, mp_result
from .imfit_lib cimport Convolver, ModelObject, SolverResults, DispatchToSolver
from .imfit_lib cimport GetFunctionParameterNames
from .imfit_lib cimport MASK_ZERO_IS_GOOD, MASK_ZERO_IS_BAD
from .imfit_lib cimport WEIGHTS_ARE_SIGMAS, WEIGHTS_ARE_VARIANCES, WEIGHTS_ARE_WEIGHTS
from .imfit_lib cimport NO_FITTING, MPFIT_SOLVER, DIFF_EVOLN_SOLVER, NMSIMPLEX_SOLVER

# from local pure-Python module
from .model import ModelDescription, FunctionDescription, ParameterDescription

import sys
import numpy as np
cimport numpy as np
from copy import deepcopy

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp cimport map

from libc.stdlib cimport calloc, free
from libc.string cimport memcpy


# code snippet to check for system byte order
sys_byteorder = ('>', '<')[sys.byteorder == 'little']

# convert an ndarray to local system byte order, if it's not already
def FixByteOrder( array ):
	if array.dtype.byteorder not in ('=', sys_byteorder):
		array = array.byteswap().newbyteorder(sys_byteorder)
	return array

def FixImage( array ):
	"""Checks an input numpy array; if necessary, converts array to
	double-precision floating point, little-endian byte order, and
	contiguous layout.
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
	


def FunctionNames():
	"""
	List the available Imfit image functions.
	
	Returns
	-------
	func_names : list of string
		list containing the image-function names
	"""
	cdef vector[string] function_names
	GetFunctionNames(function_names)
	# return function names as Unicode strings
	return [func_name.decode('UTF-8') for func_name in function_names]




def function_description( func_type, name=None ):
	'''
	Given a string specifying the name of an Imfit image function,
	returns an instance of FunctionDescription describing the function
	and its parameters.
	
	Parameters
	----------
	func_type : string
		The function type; must be one of the recognized Imfit image-function names
		(E.g., "Sersic", "BrokenExponential", etc. Use "imfit --list-functions" on
		the command line to get the full list, or FunctionNames in this module.)
		
	name : string, optional
		Custom identifying name for this instance of this function. 
		Example: "disk", "bulge".
		Default: None.
		
	Returns
	-------
	func_desc : :class:`FunctionDescription`
		Instance of :class:`FunctionDescription`.
		
	'''
	cdef int status
	cdef vector[string] parameters
	# convert string to byte form for use by C++
	status = GetFunctionParameterNames(func_type.encode(), parameters)
	if status < 0:
		msg = 'Function name \"{0}\" is not a recognized Imfit image function.'.format(func_type)
		raise ValueError(msg)
	func_desc = FunctionDescription(func_type, name)
	for paramName in parameters:
		# convert parameter names to Unicode strings
		param_desc = ParameterDescription(paramName.decode('UTF-8'), value=0.0)
		func_desc.addParameter(param_desc)
	return func_desc




def convolve_image( np.ndarray[np.double_t, ndim=2] image not None,
				   np.ndarray[np.double_t, ndim=2] psf not None,
				   int nproc=0, verbose=False ):
	'''
	Convolve an image with a given PSF.
	
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
	
	convolver.SetupPSF(&psf_data[0], psf.shape[1], psf.shape[0])
	
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



cdef class ModelObjectWrapper( object ):

	cdef ModelObject *_model 
	cdef vector[mp_par] _paramInfo
	cdef double *_paramVect
	cdef bool _paramLimitsExist
	cdef int _nParams
	cdef int _nFreeParams
	cdef object _modelDescr
	cdef object _parameterList
	cdef int _nPixels, _nRows, _nCols
	cdef SolverResults *_solverResults
	cdef mp_result *_fitResult
	cdef int _fitStatus
	
	cdef double[::1] _imageData
	cdef double[::1] _errorData
	cdef double[::1] _maskData
	cdef double[::1] _psfData
	cdef bool _inputDataLoaded
	cdef bool _fitted
	cdef object _fitMode
	cdef bool _freed
	

	def __init__( self, object model_descr, int debug_level=0, int verbose_level=-1, 
					bool subsampling=True ):
		self._paramLimitsExist = False
		self._paramVect = NULL
		self._solverResults = NULL
		self._model = NULL
		self._fitResult = NULL

		self._inputDataLoaded = False
		self._fitted = False
		self._fitMode = None
		self._freed = False
		self._fitStatus = 0
		
		if not isinstance(model_descr, ModelDescription):
			raise ValueError('model_descr must be a ModelDescription object.')
		self._modelDescr = model_descr
		
		self._solverResults = new SolverResults()
		if self._solverResults == NULL:
			raise MemoryError('Could not allocate SolverResults.')

		self._model = new ModelObject()
		self._model.SetDebugLevel(debug_level)
		# FIXME: maybe add back in SetVerboseLevel() method to ModelObject?
#		self._model.SetVerboseLevel(verbose_level)
		if self._model == NULL:
			raise MemoryError('Could not allocate ModelObject.')
		self._addFunctions(self._modelDescr, subsampling=subsampling, verbose=debug_level>0)
		self._paramSetup(self._modelDescr)
		
		
	def setMaxThreads(self, int nproc):
		self._model.SetMaxThreads(nproc)
		
		
	def setChunkSize(self, int chunk_size):
		self._model.SetOMPChunkSize(chunk_size)
		

	def _paramSetup(self, object model_descr):
		cdef mp_par newParamInfo
		self._parameterList = model_descr.parameterList()
		self._nParams = self._nFreeParams = self._model.GetNParams()
		if self._nParams != len(self._parameterList):
			raise Exception('Number of input parameters (%d) does not equal required number of parameters for specified functions (%d).' % (len(self._parameterList), self._nParams))
		self._paramVect = <double *> calloc(self._nParams, sizeof(double))
		if self._paramVect == NULL:
			raise MemoryError('Could not allocate parameter initial values.')
	
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


	cdef _addFunctions(self, object model_descr, bool subsampling, bool verbose=False):
		cdef int status = 0
		functionNameList = [funcName.encode() for funcName in model_descr.functionList()]
		status = AddFunctions(self._model, functionNameList, model_descr.functionSetIndices(), 
								subsampling, verbose)
		if status < 0:
			raise RuntimeError('Failed to add the functions.')

	
	def setPSF(self, np.ndarray[np.double_t, ndim=2, mode='c'] psf):
		cdef int n_rows_psf, n_cols_psf

		# FIXME: check that PSF data has correct type, byteorder
		# Maybe this was called before.
# 		if self._psfData != NULL:
# 			free(self._psfData)
# 		self._psfData = alloc_copy_from_ndarray(psf)
		# Cython typed memoryview, pointing to flattened (1D) copy of PSF data
		self._psfData = psf.flatten()

		n_rows_psf = psf.shape[0]
		n_cols_psf = psf.shape[1]
		self._model.AddPSFVector(n_cols_psf * n_rows_psf, n_cols_psf, n_rows_psf, &self._psfData[0])
		

	def loadData(self,
				 np.ndarray[np.double_t, ndim=2, mode='c'] image not None,
				 np.ndarray[np.double_t, ndim=2, mode='c'] error,
				 np.ndarray[np.double_t, ndim=2, mode='c'] mask,
				 **kwargs):
		
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


	def setupModelImage(self, shape):
		if self._inputDataLoaded:
			raise Exception('Input data already loaded.')
		self._nRows = shape[0]
		self._nCols = shape[1]
		self._nPixels = self._nRows * self._nCols
		self._model.SetupModelImage(self._nCols, self._nRows)
		self._model.CreateModelImage(self._paramVect)
		self._inputDataLoaded = True
		
		
	def _testCreateModelImage(self, int count=1):
		for _ from 0 <= _ < count:
			self._model.CreateModelImage(self._paramVect)
		
		
	def fit( self, double ftol=1e-8, int verbose=-1, mode='LM' ):
		cdef int solverID
		cdef string solverName
		status = self._model.FinalSetupForFitting()
		if status < 0:
			raise Exception('Failure in ModelObject::FinalSetupForFitting().')

		solverID = solverID_dict[mode]
		solverName = ""
		self._fitStatus = DispatchToSolver(solverID, self._nParams, self._nFreeParams,
											self._nPixels, self._paramVect, self._paramInfo,
											self._model, ftol, self._paramLimitsExist,
											verbose, self._solverResults, solverName)
		if mode == 'LM':
			self._fitResult = self._solverResults.GetMPResults()

		self._fitMode = mode
		self._fitted = True
	
	
	def getModelDescription(self):
		model_descr = deepcopy(self._modelDescr)
		for i, p in enumerate(model_descr.parameterList()):
			p.setValue(self._paramVect[i])
		return model_descr
	
		
	def getRawParameters(self):
		vals = []
		for i in xrange(self._nParams):
			vals.append(self._paramVect[i])
		return vals
			
	
	# FIXME: possibly change this to use typed memoryview?
	# (note that we *do* need to *copy* the data pointed to by model_image,
	# since we want to return a self-contained numpy array
	def getModelImage( self ):
		cdef double *model_image
		cdef np.ndarray[np.double_t, ndim=2, mode='c'] output_array
		cdef int imsize = self._nPixels * sizeof(double)

		model_image = self._model.GetModelImageVector()
		if model_image is NULL:
			raise Exception('Error: model image has not yet been computed.')
		output_array = np.empty((self._nRows, self._nCols), dtype='float64')
		memcpy(&output_array[0,0], model_image, imsize)

		return output_array
		
		
	def getFitStatistic( self, mode='none' ):
		cdef double fitstat
		if self.fittedLM:
			fitstat = self._fitResult.bestnorm
		else:
			fitstat = self._model.GetFitStatistic(self._paramVect)
		cdef int n_valid_pix = self._model.GetNValidPixels()
		cdef int deg_free = n_valid_pix - self._nFreeParams

		if mode == 'none':
			return fitstat
		elif mode == 'reduced':
			return fitstat / deg_free
		elif mode == 'AIC':
			return AIC_corrected(fitstat, self._nFreeParams, n_valid_pix, 1)
		elif mode == 'BIC':
			return BIC(fitstat, self._nFreeParams, n_valid_pix, 1);
		else:
			raise Exception('Unknown statistic mode: %s' % mode)


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
	def nFev(self):
		if self.fittedLM:
			return self._fitResult.nfev
		else:
			return -1
	
	
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
		if self._solverResults != NULL:
			free(self._solverResults)
			# note that this should automatically free self._fitResult, if that
			# was pointing to something allocated by SolverResults			
		self._freed = True
		
