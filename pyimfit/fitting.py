"""

"""
# Created on Sep 20, 2013
#
# [original author: Andre de Luiz Amorim; modifications by Peter Erwin]

import copy
from typing import Union

import numpy as np   # type: ignore

from .descriptions import ModelDescription
from .pyimfit_lib import FixImage, PsfOversampling, ModelObjectWrapper  # type: ignore

__all__ = ['MakePsfOversampler', 'FitResult', 'Imfit']


# These map PyImfit option names (e.g., "gain", "read_noise") to Imfit config-file option
# names (e.g., "GAIN", "READNOISE"), or vice-versa
imageOptionNameDict = {'n_combined': "NCOMBINED", 'exptime': "EXPTIME", 'gain': "GAIN",
                  'read_noise': "READNOISE", 'original_sky': "ORIGINAL_SKY"}
imageOptionNameDict_reverse = {"NCOMBINED": 'n_combined', "EXPTIME": 'exptime', "GAIN": 'gain',
                  "READNOISE": 'read_noise', "ORIGINAL_SKY": 'original_sky' }


class FitError(Exception):
    pass



class FitResult( dict ):
    """
    Represents the result of fitting an image.
    Constructed by Imfit.getFitResult method.

    Attributes
    ----------
    solverName : str
        Which solver was used (e.g., "LM", "NM", "DE")
    fitConverged : bool
        Whether fit converged or not
    nIter : int
        Number of iterations performed during fit
    fitStat : float
        Final fit-statistic value
    fitStatReduced : float
        Final reduced fit-statistic value
    aic : float
        Final Akaike Information Criterion value
    bic : float
        Final Bayesian Information Criterion value
    params : ndarray
        The best-fit vector of parameter values.
    paramErrs : ndarray
        Cooresponding uncertainties on best-fit parameter values, if L-M
        fit was done.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v) for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"


    def __dir__(self):
        return list(self.keys())



def _composemask( arr, mask, mask_zero_is_bad: bool ):
    """
    Helper function to properly compose masks.

    If ``arr`` (e.g., data image) is a numpy MaskedArray, then:
        If ``mask`` does not exist, the mask part of ``arr`` is returned as the final mask
        If ``mask`` *does* exist, then the final mask is the union of it and the mask part
            of ``arr`` (that is, any pixel that is masked in either of the input masks is
            masked in the output final mask)
    If ``arr`` is *not* a MaskedArray, then ``mask`` is returned as the final mask

    Parameters
    ----------
    arr : 2-D numpy array
        Image to be fitted. Can be a masked array.

    mask : 2-D numpy array or None
        Array containing the masked pixels; must have the same shape as ``image``.
        Pixels set to ``True`` are bad by default (see the kwarg ``mask_format``
        for other options). If not set and ``image`` is a masked array, then its
        mask is used. If both masks are present, the final mask is composed by masking
        any pixel that is masked in either of the input masks.

    mask_zero_is_bad : bool

    Returns
    -------
    mask : 2D numpy array or None
        Final output mask
    """
    if isinstance(arr, np.ma.MaskedArray):
        if mask is None:
            if mask_zero_is_bad:
                mask = ~arr.mask
            else:
                mask = arr.mask
        else:
            if mask_zero_is_bad:
                mask = ~arr.mask & mask
            else:
                mask = arr.mask | mask
    return mask



def MakePsfOversampler( psfImage, oversampleScale, regionSpec, psfNormalization=True ):
    """
    Helper function to generate PsfOversampling objects (corrects input psfImage, sets
    up region string, etc.).

    Parameters
    ----------
    psfImage : 2-D numpy array
        the oversampled PSF image

    oversampleScale : int
        oversampling scale of psfImage relative to data image: number of PSF-image
         pixels per data-image pixel in one dimension (must be >= 1)

    regionSpec : sequence of int
        specifies inclusive boundaries of image region to be oversampled
        [x1,x2,y1,y2]

    psfNormalization : bool, optional
        Normalize the PSF image before using.
        Default: ``True``.

    Returns
    -------
    oversampleInfo : instance of PsfOversampling class
    """

    psfImage = FixImage(psfImage)
    regionString = "{0:d}:{1:d},{2:d}:{3:d}".format(regionSpec[0],regionSpec[1],regionSpec[2],regionSpec[3])
    return PsfOversampling(psfImage, oversampleScale, regionString, 0, 0, psfNormalization)



class Imfit( object ):
    """
    The main class for fitting models to images using Imfit.
    Can also be used to create images based on models.

    Due to some library limitations, this object can only fit the model
    specified in the constructor. If you want to fit several models,
    you need to create one instance of :class:`Imfit` for each model.
    On the other hand, one instance can be used to fit the same model
    to any number of images, or to fit and then create the model image.

    Attributes
    ----------
    AIC
    BIC
    fitConverged
    fitError
    fitStatistic
    fitTerminated
    nIter
    nPegged
    nValidPixles
    numberedParameterNames
    parameterErrors
    reducedFitStatistic

    See also
    --------
    parse_config_file
    """

    def __init__( self, model_descr: Union[ModelDescription, dict], psf=None, psfNormalization=True,
                  quiet=True, maxThreads=0, chunk_size=10, subsampling=True, zeroPoint=None ):
        """
        Parameters
        ----------
        model_descr : :class:`ModelDescription` OR dict
            Model to be fitted to the data; can be either
                an instance of :class:`ModelDescription`
                OR: a dict (suitable for use by ModelDescription.dict_to_ModelDescription)

        psf : 2-D Numpy array, optional
            Point Spread Function image to be convolved to the images.
            Note that if the model_descr contains one or more PointSource-type functions,
            then a PSF *must* be supplied.
            Default: ``None`` (no convolution).

        psfNormalization : bool, optional
            Normalize the PSF image before using.
            Default: ``True``.

        quiet : bool, optional
            Suppress output, only error messages will be printed.
            Default: ``True``.

        maxThreads : int, optional
            Number of threads to use when fitting. If `0``, use all available threads.
            Default: ``0`` (use all available threads).

        chunk_size : int, optional
            Chunk size for OpenMP processing

        subsampling : bool, optional
            Use pixel subsampling near centers of image functions.
            Default: ``True``.

        zeroPoint : float, optional
            photometric zero point for data image (used only for outputting
            model and component magnitudes) via getModelMagnitudes
        """
        if (type(model_descr) != dict) and not (isinstance(model_descr, ModelDescription)):
            raise ValueError('model_descr must be a ModelDescription object or a dict.')
        self._modelObjectWrapper = None
        # if model_descr is an actual ModelDescription object (rather than a dict), copy it
        # (we don't want any links to the input object, in case the latter gets updated later
        # somewhere else). Don't use copy.deepcopy, since this zeros some attribute values
        # (e.g., model_descr.nParameters)
        if type(model_descr) is dict:
            self._modelDescr = ModelDescription.dict_to_ModelDescription(model_descr)
        else:
            self._modelDescr = copy.deepcopy(model_descr)
        self.nParameters = self._modelDescr.nParameters
        if psf is not None:
            self._psf = FixImage(psf)
        else:
            self._psf = None
            # [ ] check to see if model_descr contains a PointSource-type function; if so,
            # raise an error
            if self._modelDescr.hasPointSources:
                msg = "Model description includes at least one PointSource-type function, "
                msg += "but no PSF image was supplied in call to Imfit()"
                raise TypeError(msg)
        self._normalizePSF = psfNormalization
        self._mask = None
        self._maxThreads = maxThreads
        self._chunkSize = chunk_size
        if quiet:
            self._debugLevel = 0
            self._verboseLevel = -1
        else:
            self._debugLevel = 1
            self._verboseLevel = 1
        self._subsampling = subsampling
        self._dataSet = False
        self._finalSetupDone = False
        self._fitDone = False
        self._lastSolverUsed = None
        self._fitStatComputed = False
        self._zeroPoint = zeroPoint


    def getModelDescription(self):
        """
        Returns
        -------
        model : :class:`ModelDescription`
            A copy of the currently fitted model, or a copy of
            the template model if no fitting has taken place yet.
        """
        if self._modelObjectWrapper is not None:
            return self._modelObjectWrapper.getModelDescription()
        else:
            return copy.copy(self._modelDescr)


    def _updateModelDescription( self, kwargs ):
        """Updates the internal options dict ("GAIN", etc.) in self._modelDesc

        Parameters
        ----------
        kwargs : dict
            dictionary of keyword arguments (e.g., input parameters for loadData)
        """
        if len(kwargs) > 0:
            options = {}
            # add key-value pairs to options, but only for valid image-option keys
            for kw in kwargs.keys():
                try:
                    optionName = imageOptionNameDict[kw]
                    options[optionName] = kwargs[kw]
                except KeyError:
                    pass
            self._modelDescr.updateOptions(options)


    def saveCurrentModelToFile( self, filename: str, includeImageOptions=False ):
        """
        Saves the current model description and parameter values to a text file in
        Imfit-configuration-file format.

        Parameters
        ----------
        filename : str
            Name for the output file

        includeImageOptions : bool, optional
            if True, then image-description options ("GAIN", etc.) are also written
            to the output file
        """
        # use getModelDescription to get the current (e.g., updated with best-fit parameters)
        # ModelDescription object from self._modelObjectWrapper
        modelDesc = self.getModelDescription()
        # self.parameterErrors will be None if the fit hasn't been performed yet,
        # or if the fit did not produce parameter uncertainty estimates
        outputLines = modelDesc.getStringDescription(errors=self.parameterErrors, saveOptions=includeImageOptions)
        with open(filename, 'w') as outf:
            for line in outputLines:
                outf.write(line)


    def getModelAsDict(self):
        """
        Returns current model (including parameter values) as a dict suitable for use
        with ModelDescription.dict_to_ModelDescription class method.

        Returns
        -------
        model_dict : dict
        """
        model_desc = self.getModelDescription()
        return model_desc.getModelAsDict()


    def getRawParameters(self):
        """
        Returns current model parameter values.

        Returns
        -------
        raw_params : ndarray of float
            A 1D array containing all the model parameter values.
        """
        return np.array(self._modelObjectWrapper.getRawParameters())


    def getParameterErrors(self):
        """
        Returns current best-fit model parameter uncertainties (from L-M minimization).

        Returns
        -------
        errors : ndarray of float
            A 1D array containing the Levenberg-Marquardt parameter uncertainties.
        """
        return np.array(self._modelObjectWrapper.getParameterErrors())


    def getParameterLimits(self):
        """
        Returns a list containing lower and upper limits for all parameters in the model.

        Returns
        -------
        parameterLimits : list of 2-element tuples of float
            [(lower_limit, upper_limit)_1, (lower_limit, upper_limit)_2, ...]
         """
        return self._modelDescr.getParameterLimits()


    def _setupModel(self):
        """
        Creates the internal ModelObjectWrapper instance (which in turn creates the actual
        C++ ModelObject instance), including model description (and PSF if it exists).
        """
        if self._modelObjectWrapper is not None:
            # FIXME: Find a better way to free cython resources.
            self._modelObjectWrapper.close()
        self._modelObjectWrapper = ModelObjectWrapper(self._modelDescr, self._debugLevel,
                                                      self._verboseLevel, self._subsampling,
                                                      self._psf, self._normalizePSF)
        # if self._psf is not None:
        #     self._modelObjectWrapper.setPSF(np.asarray(self._psf), self._normalizePSF)
        if self._maxThreads > 0:
            self._modelObjectWrapper.setMaxThreads(self._maxThreads)
        if self._chunkSize > 0:
            self._modelObjectWrapper.setChunkSize(self._chunkSize)


    def loadData( self, image, error=None, mask=None, **kwargs ):
        """
        Supply the underlying ModelObject instance with data image and error model,
        optionally including error and/or mask images.

        Parameters
        ----------
        image : 2-D numpy array (ndarray or MaskedArray)
            Image to be fitted. Can be a masked array.

        error : 2-D numpy array, optional
            error/weight image, same shape as ``image``. If not set,
            errors are generated from ``image``. See also the keyword args
            ``use_poisson_mlr``, ``use_cash_statistics``, and ``use_model_for_errors``.

        mask : 2-D numpy array, optional
            Array containing the masked pixels; must have the same shape as ``image``.
            Pixels set to ``True`` are bad by default (see the kwarg ``mask_format``
            for other options). If not set and ``image`` is a masked array, then its
            mask is used. If both masks are present, the final mask is composed by masking
            any pixel that is masked in either of the input masks.

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
            Values in ``error`` should be interpreted as:
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

        all_kw = ['n_combined', 'exp_time', 'gain', 'read_noise', 'original_sky',
                  'error_type', 'mask_format', 'psf_oversampling_list', 'use_poisson_mlr',
                  'use_cash_statistics', 'use_model_for_errors']
        supplied_kw = list(kwargs.keys())
        for kw in supplied_kw:
            if kw not in all_kw:
                raise ValueError('Unknown kwarg: %s' % kw)
        mask_zero_is_bad = 'mask_format' in kwargs and kwargs['mask_format'] == 'zero_is_bad'

        # create the ModelObjectWrapper instance
        self._setupModel()

        # check to make sure we don't skip ModelDescription options that were already
        # set, if they're not overridden by kwargs
        if len(self._modelDescr.optionsDict) > 0:
            for key,value in self._modelDescr.optionsDict.items():
                # convert "GAIN" --> "gain", etc.
                key = imageOptionNameDict_reverse[key]
                if key not in supplied_kw:
                    kwargs[key] = value

        # update the ModelDescription instance with keyword values
        if len(kwargs) > 0:
            self._updateModelDescription(kwargs)

        # Check for mismatches in image size/shape and convert to correct byte order
        # and double-precision floating point
        if mask is not None:
            if mask.shape != image.shape:
                msg = "Mask image (%d,%d) and " % mask.shape
                msg += "data image (%d,%d) shapes do not match." % image.shape
                raise ValueError(msg)
            mask = FixImage(mask)
        if error is not None:
            if error.shape != image.shape:
                msg = "Error image (%d,%d) and " % error.shape
                msg += "data image (%d,%d) shapes do not match." % image.shape
                raise ValueError(msg)
            error = FixImage(error)
        image = FixImage(image)

        # PE: this generates a "composed" mask image, which can be None, the same as mask
        # (if image is *not* a MaskedArray), the embedded mask in image if it *is* a
        # MaskedArray, or the composition of mask and the embedded mask of image.
        mask = _composemask(image, mask, mask_zero_is_bad)
        # ModelObjectWrapper.loadData expects mask to be either None or numpy.double
        if mask is not None:
            mask = mask.astype(np.double)
        # if image is MaskedArray, work with a normal ndarray copy instead
        if isinstance(image, np.ma.MaskedArray):
            image = image.filled(fill_value=0.0)

        if error is not None:
            # if error is MaskedArray, work with a normal ndarray copy instead
            if isinstance(error, np.ma.MaskedArray):
                error = error.filled(fill_value=error.max())

        # Note that the following will call _modelObjectWrapper.doFinalSetup()
        self._modelObjectWrapper.loadData(image, error, mask, **kwargs)
        self._dataSet = True


    def doFit( self, solver='LM', ftol=1e-8, verbose=None ):
        """
        Fit the model to previously supplied data image.

        Parameters
        ----------
        solver : string, optional
            One of the following solvers (optimization algorithms) to be used for the fit:
                * ``'LM'`` : Levenberg-Marquardt.
                * ``'NM'`` : Nelder-Mead Simplex.
                * ``'DE'`` : Differential Evolution.

        ftol : float, optional
            fractional tolerance in fit statistic for determining fit convergence

        verbose : int or None, optional
            set this to an integer to specify a feedback level for the fit (this overrides
            the Imfit object's internal verbosity setting)

        Returns
        -------
        result : FitResult object

        Examples
        --------
        TODO: Examples of doFit().

        """
        if not self._dataSet:
            raise Exception('No data for fit! (Supply it via loadData() or fit() methods)')
        if solver not in ['LM', 'NM', 'DE']:
            raise ValueError('Invalid solver name: {0}'.format(solver))

        if not self._finalSetupDone:
            self._modelObjectWrapper.doFinalSetup()
        self._finalSetupDone = True
        if verbose is not None:
            verboseLevel = verbose
        else:
            verboseLevel = self._verboseLevel
        self._modelObjectWrapper.fit(verbose=verboseLevel, mode=solver, ftol=ftol)
        self._lastSolverUsed = solver
        if not self.fitError:
            self._fitDone = True
            self._fitStatComputed = True
        return self.getFitResult()


    def fit( self, image, error=None, mask=None, solver='LM', ftol=1e-8, verbose=None, **kwargs ):
        """
        Supply data image (and optionally mask and/or error images) and image info, then
        fit the model to the data.

        Parameters
        ----------
        image : 2-D numpy array (ndarray or MaskedArray)
            Image to be fitted. Can be a masked array.

        error : 2-D numpy array, optional
            error/weight image, same shape as ``image``. If not set,
            errors are generated from ``image``. See also the keyword args
            ``use_poisson_mlr``, ``use_cash_statistics``, and ``use_model_for_errors``.

        mask : 2-D numpy array, optional
            Array containing the masked pixels; must have the same shape as ``image``.
            Pixels set to ``True`` are bad by default (see the kwarg ``mask_format``
            for other options). If not set and ``image`` is a masked array, then its
            mask is used. If both masks are present, the final mask is composed by masking
            any pixel that is masked in either of the input masks.

        solver : string, optional
            One of the following solvers (optimization algorithms) to be used for the fit:
                * ``'LM'`` : Levenberg-Marquardt.
                * ``'NM'`` : Nelder-Mead Simplex.
                * ``'DE'`` : Differential Evolution.

        ftol : float, optional
            fractional tolerance in fit statistic for determining fit convergence

        verbose : int or None, optional
            set this to an integer to specify a feedback level for the fit (this overrides
            the Imfit object's internal verbosity setting)

        See loadData() for list of allowed extra keywords.

        Returns
        -------
        result : FitResult object
        """
        self.loadData(image, error, mask, **kwargs)
        return self.doFit(solver=solver, ftol=ftol, verbose=verbose)


    def getFitResult( self ):
        """
        Returns a summary of the fitting process.

        Returns
        -------
        result : FitResult object
        """
        if not self._fitDone:
            raise FitError()
        result = FitResult()
        if self._fitDone and not self.fitError:
            result.solverName = self._lastSolverUsed
            result.fitConverged = self.fitConverged
            result.nIter = self.nIter
            result.nFuncEvals = self.nFuncEvals
            result.fitStat = self.fitStatistic
            result.fitStatReduced = self.reducedFitStatistic
            result.aic = self.AIC
            result.bic = self.BIC
            result.params = self.getRawParameters()
            result.paramErrs = self.parameterErrors
        return result


    def computeFitStatistic( self, newParameters ):
        """
        Returns the fit-statistic value for the specified parameter vector.
        (Which fit statistic will calculated is set by the loadData() or fit() methods.)

        Parameters
        ----------
        newParameters : ndarray of float

        Returns
        -------
        fitStatistic : float
        """
        if len(newParameters) != self.nParameters:
            msg = "Number of input parameters (%d) " % len(newParameters)
            msg += "does not equal number of model parameters (%d)!" % self.nParameters
            raise ValueError(msg)
        if not isinstance(newParameters, np.ndarray):
            newParams = np.array(newParameters).astype(np.float64)
        else:
            newParams = newParameters.astype(np.float64)
        return self._modelObjectWrapper.computeFitStatistic(newParams)


    def runBootstrap( self, nIterations, ftol=1e-8, verboseFlag=False, seed=0, getColumnNames=False ):
        """
        Do bootstrap resampling for a model.

        Parameters
        ----------
        nIterations : int
            How many bootstrap samples to generate and fit

        ftol : float, optional
            fractional tolerance in fit statistic for determining fit convergence

        verboseFlag : bool, optional
            if True, a progress bar is printed during the boostrap iterations

        seed : int, optional
            random number seed (default is to use system clock)

        getColumnNames : bool, optional
            if True, then column (parameter) names are returned as well

        Returns
        -------
        bootstrapOutput : 2-D ndarray of float
        OR
        (columnNames, bootstrapOutput) : tuple of (list of str, 2-D ndarray of float)
        """
        if not self._dataSet:
            raise Exception('No data supplied for model -- cannot run bootstrap resampling')

        bootstrapOutput = self._modelObjectWrapper.doBootstrapIterations(nIterations, ftol=ftol,
                                                                         verboseFlag=verboseFlag, seed=seed)
        if getColumnNames:
            parameterNames = self.numberedParameterNames
            return (parameterNames, bootstrapOutput)
        else:
            return bootstrapOutput


    @property
    def zeroPoint(self):
        """
        float: photometric zero point (used by getModelMagnitudes method).
        """
        return self._zeroPoint

    @zeroPoint.setter
    def zeroPoint(self, value):
        self._zeroPoint = value


    @property
    def fitConverged(self):
        """
        bool: indicates whether fit converged.
        """
        return self._modelObjectWrapper.fitConverged


    @property
    def fitError(self):
        return self._modelObjectWrapper.fitError


    @property
    def fitTerminated(self):
        """
        bool: indicates whether fit terminated for any reason.
        """
        return self._modelObjectWrapper.fitTerminated


    @property
    def nIter(self):
        """
        int: number of solver iterations during fit.
        """
        return self._modelObjectWrapper.nIter


    @property
    def nFuncEvals(self):
        """
        int: number of 'function evaluations' (model-image computations) during fit.
        """
        return self._modelObjectWrapper.nFuncEvals


    @property
    def parameterErrors(self):
        """
        ndarray of float or None: estimated parameter errors from fit (L-M solver only)
        """
        if (self._modelObjectWrapper is not None and self.fitConverged and
                self._modelObjectWrapper.fittedLM):
            return self.getParameterErrors()
        else:
            return None


    @property
    def nPegged(self):
        """
        int: number of parameters pegged against limits at end of fit.
        """
        return self._modelObjectWrapper.nPegged


    @property
    def nValidPixels(self):
        """
        int: number of non-masked pixels in data image.
        """
        return self._modelObjectWrapper.nValidPixels


    @property
    def fitStatistic(self):
        """
        float: the :math:`\\chi^2`, Poisson MLR, or Cash statistic of the fit.
        """
        return self._modelObjectWrapper.getFitStatistic(mode='none')


    @property
    def reducedFitStatistic(self):
        """
        float: the "reduced" :math:`\\chi^2` or Poisson MLR of the fit.
        """
        return self._modelObjectWrapper.getFitStatistic(mode='reduced')


    @property
    def AIC(self):
        """
        float: bias-corrected Akaike Information Criterion for the fit.
        """
        return self._modelObjectWrapper.getFitStatistic(mode='AIC')


    @property
    def BIC(self):
        """
        float: Bayesian Information Criterion for the fit.
        """
        return self._modelObjectWrapper.getFitStatistic(mode='BIC')


    @property
    def numberedParameterNames(self):
        """
        list of str: List of parameter names for the current model, annotated by function number.
        E.g., ["X0_1", "Y0_1", "PA_1", "ell_1", "I_0_1", "h_1", ...]
        """
        return self._modelDescr.numberedParameterNames


    def getModelImage( self, shape=None, newParameters=None, includeMask=False ):
        """
        Computes and returns the image described by the currently fitted model,
        the input model if no fit was performed, or user-supplied parameters.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the image in (Y, X) = (nRows, nColumns) format.
            Do NOT supply this if Imfit object's image shape has already been defined
            via loadData() or fit() method!

        newParameters : 1-D numpy array of float, optional
            vector of parameter values to use in computing model
            (default is to use current parameter values, e.g., from fit)

        includeMask : bool, optional
            Specifies whether output should be numpy masked array, if there
            is a mask image associated with the data image.

        Returns
        -------
        image : 2-D numpy array
            Image computed from the current model. If a mask is associated with the
            original data image, then the returned image is a numpy masked array
        """
        if self._modelObjectWrapper is None:
            self._setupModel()
        if shape is not None:
            if self._modelObjectWrapper.imageSizeSet:
                msg = "Model image size has already been set!"
                raise ValueError(msg)
            # OK, PROBLEM IS HERE!
            self._modelObjectWrapper.setupModelImage(shape)
        if (newParameters is not None) and (len(newParameters) != self.nParameters):
            msg = "Number of input parameters (%d) " % len(newParameters)
            msg += "does not equal number of model parameters (%d)!" % self.nParameters
            raise ValueError(msg)

        image = self._modelObjectWrapper.getModelImage(newParameters=newParameters)
        if self._mask is not None and includeMask:
            return np.ma.array(image, mask=self._mask)
        else:
            return image


    def getModelFluxes( self, newParameters=None ):
        """
        Computes and returns total and individual-function fluxes for the current model
        and current (or user-supplied) parameter values.

        Parameters
        ----------
        newParameters : 1-D numpy array of float, optional
            vector of parameter values to use in computing model
            (default is to use current parameter values, e.g., from most rencent fit, instead)

        Returns
        -------
        (totalFlux, individualFluxes) : tuple of (float, ndarray of float)
            totalFlux = total flux (or magnitude) of model
            individualFluxes = numpy ndarray of fluxes/magnitudes for each image-function in the
            model
        """
        if (newParameters is not None) and (len(newParameters) != self.nParameters):
            msg = "Number of input parameters (%d) " % len(newParameters)
            msg += "does not equal number of model parameters (%d)!" % self.nParameters
            raise ValueError(msg)
        totalFlux, functionFluxes = self._modelObjectWrapper.getModelFluxes(newParameters=newParameters)
        return(totalFlux, functionFluxes)


    def getModelMagnitudes( self, newParameters=None, zeroPoint=None ):
        """
        Computes and returns total and individual-function magnitudes for the current model
        and current (or user-supplied) parameter values.

        Parameters
        ----------
        newParameters : 1-D numpy array of float, optional
            vector of parameter values to use in computing model
            (default is to use current parameter values, e.g., from most rencent fit, instead)

        zeroPoint : float, optional
            If present, returned values are magnitudes, computed as
                zeroPoint - 2.5*log10(flux)
            (default is to use value of object's zeroPoint property)

        Returns
        -------
        (totalMag, individualMags) : tuple of (float, ndarray of float)
            totalFlux = total flux (or magnitude) of model
            individualFluxes = numpy ndarray of fluxes/magnitudes for each image-function in the
            model

        """
        if (newParameters is not None) and (len(newParameters) != self.nParameters):
            msg = "Number of input parameters (%d) " % len(newParameters)
            msg += "does not equal number of model parameters (%d)!" % self.nParameters
            raise ValueError(msg)
        totalFlux, functionFluxes = self._modelObjectWrapper.getModelFluxes(newParameters=newParameters)
        if zeroPoint is not None:
            ZP = zeroPoint
        else:
            ZP = self.zeroPoint
        #FIXME: Handle case of zeroPoint = None! (i.e., user forget to set it...)
        return (ZP - 2.5*np.log10(totalFlux), ZP - 2.5*np.log10(functionFluxes))


    def __del__(self):
        try:
            if self._modelObjectWrapper is not None:
                # FIXME: Find a better way to free Cython resources.
                self._modelObjectWrapper.close()
        except AttributeError:
            # sometimes this gets called when no _modelObjectWrapper attribute
            # was defined; in this case, there's no memory to worry about freeing
            pass
