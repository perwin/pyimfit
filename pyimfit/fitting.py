"""
Created on Sep 20, 2013

@author: andre
"""
from .model import ModelDescription
from .pyimfit_lib import ModelObjectWrapper
import numpy as np
from copy import deepcopy

__all__ = ['Imfit']


def _composemask(arr, mask, mask_zero_is_bad):
    """
    Helper function to properly compose masks.
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



class Imfit(object):
    """
    A class for fitting models to images using Imfit.
    Can also be used to create images based on models.

    Due to some library limitations, this object can only fit the model
    specified in the constructor. If you want to fit several models,
    you need to create one instance of :class:`Imfit` for each model.
    On the other hand, one instance can be used to fit the same model
    to any number of images, or to fit and then create the model image.

    Parameters
    ----------
    model_descr : :class:`ModelDescription`
        Template model to be fitted, an instance of :class:`ModelDescription`.
        It will be the template model to every subsequent fitting in this instance.

    psf : 2-D array
        Point Spread Function image to be convolved to the images.
        Default: ``None`` (no convolution).

    quiet : bool, optional
        Suppress output, only error messages will be printed.
        Default: ``True``.

    nproc : int, optional
        Number of processors to use when fitting. If `0``,
        use all available processors.
        Default: ``0`` (use all processors).

    subsampling : bool, optional
        Use pixel subsampling near center.
        Default: ``True``.

    See also
    --------
    parse_config_file, fit
    """

    def __init__( self, model_descr, psf=None, quiet=True, nproc=0, chunk_size=8, subsampling=True):
        if not isinstance(model_descr, ModelDescription ):
            raise ValueError('model_descr must be a ModelDescription object.')
        self._modelDescr = model_descr
        self._psf = psf
        self._mask = None
        self._modelObject = None
        self._nproc = nproc
        self._chunkSize = chunk_size
        if quiet:
            self._debugLevel = 0
            self._verboseLevel = -1
        else:
            self._debugLevel = 1
            self._verboseLevel = 1
        self._subsampling = subsampling

    def getModelDescription(self):
        """
        Returns
        -------
        model : :class:`ModelDescription`
            A copy of the currently fitted model, or a copy of
            the template model if no fitting has taken place yet.
        """
        if self._modelObject is not None:
            return self._modelObject.getModelDescription()
        else:
            # FIXME: get rid of deepcopy
            return deepcopy(self._modelDescr)


    def getRawParameters(self):
        """
        Model parameters for debugging purposes.

        Returns
        -------
        raw_params : array of floats
            A 1D array containing all the model parameter values.
        """
        return np.array(self._modelObject.getRawParameters())


    def _setupModel(self):
        if self._modelObject is not None:
            # FIXME: Find a better way to free cython resources.
            self._modelObject.close()
        self._modelObject = ModelObjectWrapper(self._modelDescr, self._debugLevel,
                                               self._verboseLevel, self._subsampling)
        if self._psf is not None:
            self._modelObject.setPSF(np.asarray(self._psf))
        if self._nproc > 0:
            self._modelObject.setMaxThreads(self._nproc)
        if self._chunkSize > 0:
            self._modelObject.setChunkSize(self._chunkSize)


    def fit( self, image, error=None, mask=None, mode='LM', **kwargs ):
        """
        Fit the model to ``image``, optionally specifying that Gaussian per-pixel errors
        should be derived from the ``error`` image (by default, this treats the pixel
        values in ``error`` as Gaussian sigmas) and also optionally masking some pixels.

        Parameters
        ----------
        image : 2-D array
            Image to be fitted. Can be a masked array.

        error : 2-D array, optional
            error/weight image, same shape as ``image``. If not set,
            errors are generated from ``image``. See also the keyword args
            ``use_poisson_mlr``, ``use_cash_statistics``, and ``use_model_for_errors``.

        mask : 2-D array, optional
            Array containing the masked pixels; must have the same shape as ``image``.
            Pixels set to ``True`` are bad by default (see the kwarg ``mask_format``
            for other options). If not set and ``image`` is a masked array, then its
            mask is used. If both masks are present, the final mask is composed by masking
            any pixel that is masked in either of the input masks.

        mode : string
            One of the following optimization algorithms to be used for the fit:
                * ``'LM'`` : Levenberg-Marquardt.
                * ``'NM'`` : Nelder-Mead Simplex.
                * ``'DE'`` : Differential Evolution.

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

        Examples
        --------
        TODO: Examples of fit().

        """
        if mode not in ['LM', 'NM', 'DE']:
            raise Exception('Invalid fit mode: %s' % mode)
        all_kw = ['n_combined', 'exp_time', 'gain', 'read_noise', 'original_sky',
                  'error_type', 'mask_format', 'use_poisson_mlr', 'use_cash_statistics',
                  'use_model_for_errors']
        for kw in list(kwargs.keys()):
            if kw not in all_kw:
                raise Exception('Unknown kwarg: %s' % kw)
        mask_zero_is_bad = 'mask_format' in kwargs and kwargs['mask_format'] == 'zero_is_bad'

        self._setupModel()

        mask = _composemask(image, mask, mask_zero_is_bad)
        if isinstance(image, np.ma.MaskedArray):
            image = image.filled(fill_value=0.0)
        image = image.astype('float64')

        if error is not None:
            if image.shape != error.shape:
                raise Exception('Error and image shapes do not match.')
            mask = _composemask(image, mask, mask_zero_is_bad)
            if isinstance(error, np.ma.MaskedArray):
                error = error.filled(fill_value=error.max())
            error = error.astype('float64')

        if mask is not None:
            if image.shape != mask.shape:
                raise Exception('Mask and image shapes do not match.')
            mask = mask.astype('float64')

        self._modelObject.loadData(image, error, mask, **kwargs)
        self._modelObject.fit(verbose=self._verboseLevel, mode=mode)


    @property
    def fitConverged(self):
        return self._modelObject.fitConverged


    @property
    def fitError(self):
        return self._modelObject.fitError


    @property
    def fitTerminated(self):
        return self._modelObject.fitTerminated


    @property
    def nIter(self):
        return self._modelObject.nIter

    @property
    def nPegged(self):
        return self._modelObject.nPegged


    @property
    def nValidPixels(self):
        return self._modelObject.nValidPixels


    @property
    def fitStatistic(self):
        """
        The :math:`\\chi^2`, Poisson MLR, or Cash statistic of the fit.
        """
        return self._modelObject.getFitStatistic(mode='none')


    @property
    def reducedFitStatistic(self):
        """
        The "reduced" :math:`\\chi^2` or Poisson MLR of the fit.
        """
        return self._modelObject.getFitStatistic(mode='reduced')


    @property
    def AIC(self):
        """
        Bias-corrected Akaike Information Criterion for the fit.
        """
        return self._modelObject.getFitStatistic(mode='AIC')


    @property
    def BIC(self):
        """
        Bayesian Information Criterion for the fit.
        """
        return self._modelObject.getFitStatistic(mode='BIC')


    def getModelImage(self, shape=None):
        """
        Computes and returns the image described by the currently fitted model.
        If not fitted, use the template model.

        Parameters
        ----------
        shape : tuple
            Shape of the image in (Y, X) format.

        Returns
        -------
        image : 2-D array
            Image computed from the current model.
        """
        if self._modelObject is None:
            self._setupModel()
        if shape is not None:
            self._modelObject.setupModelImage(shape)

        image = self._modelObject.getModelImage()
        if self._mask is None:
            return image
        else:
            return np.ma.array(image, mask=self._mask)


    def __del__(self):
        if self._modelObject is not None:
            # FIXME: Find a better way to free Cython resources.
            self._modelObject.close()
################################################################################
