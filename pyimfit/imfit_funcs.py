# Code for astronomically useful functions -- especially functions dealing
# with surface-brightness or similar (e.g., spectroscopic) profiles.
#
# For consistency, all of the main functions should have the following
# signature:
#
#    functionName( rValuesVector, parameterList, mag=True, magOutput=True )
#
# where rValuesVector can be a scalar, a Python list, or a numpy array;
# (the output will be a numpy array if the input is either a list or an
# array); parameterList is a Python list of parameter values; mag is a
# boolean indicating whether or not the input parameter values are in
# mag arcsec^-2 or not; and magOutput indicates whether the output should
# be in mag arcsec^-2 or not (mag must = True as well in this case!)
#
# By default, all functions have mag=True and magOutput=True.
#
# By default, *all* functions take a positional value (x0) as their first
# parameter, even if they ignore it.

from __future__ import division

import math
import scipy.special   # type: ignore
import scipy.optimize   # type: ignore
import numpy as np   # type: ignore

from typing import List


# Notes on the different "incomplete gamma functions"
# http://mpmath.org/doc/current/functions/expintegrals.html?highlight=gammainc#incomplete-gamma-functions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html
def gammainc_lower_scipy( z, b ):
    return scipy.special.gamma(z) * scipy.special.gammainc(z, b)

try:
    from mpmath import besselk as BesselK   # type: ignore
    from mpmath import gamma as Gamma
    from mpmath import gammainc as GammaInc
    mpmathPresent = True
except ImportError:
    from scipy.special import kv as BesselK
    from scipy.special import gamma as Gamma
    GammaInc = gammainc_lower_scipy
    mpmathPresent = False


# auxiliary functions used by other functions

def b_n_exact( n ):
    """Exact calculation of the Sersic derived parameter b_n, via solution
    of the function
            Gamma(2n) = 2 gamma_inc(2n, b_n)
    where Gamma = Gamma function and gamma_inc = lower incomplete gamma function.

    If n is a list or Numpy array, the return value is a 1-d Numpy array
    """
    def myfunc(bn, n):
        return abs(float(2*GammaInc(2*n, 0, bn) - Gamma(2*n)))
    if np.iterable(n):
        b = [scipy.optimize.brent(myfunc, (nn,)) for nn in n]
        b = np.array(b)
    else:
        b = scipy.optimize.brent(myfunc, (n,))
    return b



# Here begins the main set of imfit-compatible functions

def Moffat( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r for a Moffat profile, given the specified
    vector of parameters:
        params[0] = r0 = center of profile
        params[1] = mu_0 or I_0 [depending on whether mag=True or not]
        params[2] = fwhm
        params[3] = beta
    If mag=True, then the second parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input mu_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """
    r0 = params[0]
    if mag is True:
        mu_0 = params[1]
        I_0 = 10**(-0.4*mu_0)
    else:
        I_0 = params[1]
    fwhm = params[2]
    beta = params[3]
    exponent = math.pow(2.0, 1.0/beta)
    alpha = 0.5*fwhm/math.sqrt(exponent - 1.0)
    scaledR = np.abs(r - r0) / alpha
    denominator = (1.0 + scaledR*scaledR)**beta
    I = (I_0 / denominator)
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I



def Sersic( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r for a Sersic profile, given the specified
    vector of parameters:
        params[0] = r0 = center of (symmetric) profile
        params[1] = n
        params[2] = I_e
        params[3] = r_e
    If mag=True, then the second parameter value is mu_e, not I_e, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_e in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    n = params[1]
    if mag is True:
        mu_e = params[2]
        I_e = 10**(-0.4*mu_e)
    else:
        I_e = params[2]
    r_e = params[3]
    I = I_e * np.exp( -b_n_exact(n)*(pow(R/r_e, 1.0/n) - 1.0) )
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I


def Exponential( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r for an exponential profile, given the specified
    vector of parameters:
        params[0] = r0 = center of profile
        params[1] = I_0
        params[2] = h
    If mag=True, then the first parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    if mag is True:
        mu_0 = params[1]
        I_0 = 10**(-0.4*mu_0)
    else:
        I_0 = params[1]
    h = params[2]
    I = I_0*np.exp(-R/h)
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I


def BrokenExp( r, params: List[float], mag=True, magOutput=True ):
    """Calculate the value of a broken exponential function at r, given a
    vector of parameters:
        params[0] = r0 = center of profile
        params[1] = I_0
        params[2] = h1   [aka gamma]
        params[3] = h2   [aka beta]
        params[4] = r_b
        params[5] = alpha
    If mag=True, then the first parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.

    FIXME: Need to handle case of alpha = 0
    FIXME: Need to handle "if ( alpha*(r - Rb) > 100.0)" in case of numpy
    array instead of scalar value of r.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    if mag is True:
        mu_0 = params[1]
        I_0 = 10.0**(-0.4*mu_0)
    else:
        I_0 = params[1]
    h1 = params[2]
    h2 = params[3]
    Rb = params[4]
    alpha = params[5]

    exponent = (1.0/alpha) * (1.0/h1 - 1.0/h2)
    S = (1.0 + np.exp(-alpha*Rb))**(-exponent)

    # check for possible overflow in exponentiatino if r >> Rb
    if type(R) is np.ndarray:
        # OK, we're dealing with a numpy array
        # note that we're assuming that r is monotonically increasing!
        scaledR = alpha*(R - Rb)
        if scaledR[0] > 100.0:
            # all r are beyond crossover point
            I = I = I_0 * S * np.exp(Rb/h2 - Rb/h1 - R/h2)
        elif scaledR[-1] < 100.0:
            # no r are beyond crossover point
            I = I_0 * S * np.exp(-R/h1) * (1.0 + np.exp(alpha*(R - Rb)))**exponent
        else:
            # OK, some r are < crossover point, some are > crossover point
            goodInd = [ i for i in range(len(R)) if scaledR[i] < 100.0 ]
            crossoverInd = goodInd[-1]
            I = np.zeros(len(r))
            I[0:crossoverInd] = I_0 * S * np.exp(-R[0:crossoverInd]/h1) * (1.0 +
                                np.exp(alpha*(R[0:crossoverInd] - Rb)))**exponent
            I[crossoverInd:] = I_0 * S * np.exp(Rb/h2 - Rb/h1 - R[crossoverInd:]/h2)
    elif ( alpha*(R - Rb) > 100.0):
        # scalar value of r, with r > crossover point
        # approximate form for outer exponential:
        I = I_0 * S * np.exp(Rb/h2 - Rb/h1 - R/h2)
    else:
        # scalar value of r, with r < crossover point
        # fully correct calculation:
        I = I_0 * S * np.exp(-R/h1) * (1.0 + np.exp(alpha*(R - Rb)))**exponent

    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I


def Sech( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r for a sech profile, given the specified
    vector of parameters:
        params[0] = r0 = center of profile
        params[1] = I_0
        params[2] = h
    If mag=True, then the first parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    if mag is True:
        mu_0 = params[1]
        I_0 = 10**(-0.4*mu_0)
    else:
        I_0 = params[1]
    h = params[2]
    sech = (1.0/np.cosh(R/h))
    I = I_0*sech
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I


def Sech2( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r for a sech^2 profile, given the specified
    vector of parameters:
        params[0] = r0 = center of profile
        params[1] = I_0
        params[2] = h
    If mag=True, then the first parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    if mag is True:
        mu_0 = params[1]
        I_0 = 10**(-0.4*mu_0)
    else:
        I_0 = params[1]
    h = params[2]
    sech2 = (1.0/np.cosh(R/h))**2
    I = I_0*sech2
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I



def vdKSech( r: float, params: List[float], mag=True, magOutput=True ):
    """Compute intensity at radius r [= vertical height z in the case of off-plane
    profiles] for a profile following van der Kruit's (1988)
    generalized sech function, given the specified vector of parameters:
        params[0] = r0 = center of profile
        params[1] = I_0
        params[2] = z_0 = scale length (or height) of profile
        params[3] = alpha   [= n/2 in van der Kruit's formulation]
    If mag=True, then the first parameter value is mu_0, not I_0, and
    the value will be calculated in magnitudes, not intensities.

    To have input I_0 in magnitudes but *output* in intensity, set mag=True
    and magOutput=False.
    """

    r0 = params[0]
    R = np.abs(r - r0)
    if mag is True:
        mu_0 = params[1]
        I_0 = 10**(-0.4*mu_0)
    else:
        I_0 = params[1]
    z_0 = params[2]
    alpha = params[3]
    sech_alpha = (1.0/np.cosh(R/(alpha*z_0)))**alpha
    # Note that the following scaling (the 2**(-alpha) part) ensures that profiles
    # which differ only in alpha will converge to the same quasi-exponential profile at
    # large radii, but will *differ* as r --> 0.
    I = I_0 * (2.0**(-alpha)) * sech_alpha
    if (mag is True) and (magOutput is True):
        return -2.5 * np.log10(I)
    else:
        return I



def Gauss( x: float, params: List[float], mag=True, magOutput=True ):
    """Compute surface brightness for a profile consisting of a Gaussian,
    given input parameters in vector params:
        params[0] = x-value of Gaussian center.
        params[1] = A_gauss_mag [= magnitudes/sq.arcsec if mag=True]
        params[2] = sigma
    """

    x0 = params[0]
    if mag is True:
        A_gauss_mag = params[1]
        A = 10.0**(-0.4*A_gauss_mag)
    else:
        A = params[1]
    sigma = params[2]

    scaledX = np.abs(x - x0)
    I_gauss = A * np.exp(-(scaledX*scaledX)/(2.0*sigma*sigma))
    if (mag is True) and (magOutput is True):
        mu_gauss = -2.5 * np.log10(I_gauss)
        return mu_gauss
    else:
        return I_gauss



def GaussRing( x: float, params: List[float], mag=True, magOutput=True ):
    """Compute surface brightness for a profile consisting of a Gaussian,
    given input parameters in vector params:
        params[0] = ignored.
        params[1] = A_gauss_mag [= magnitudes/sq.arcsec if mag=True]
        params[2] = x-value of Gaussian center (i.e., ring radius)
        params[3] = sigma

    This is the Gaussian for a *ring* with ring (major-axis) radius = params[2]
    """

    x0 = params[2]
    if mag is True:
        A_gauss_mag = params[1]
        A = 10.0**(-0.4*A_gauss_mag)
    else:
        A = params[1]
    sigma = params[3]

    scaledX = np.abs(x - x0)
    I_gauss = A * np.exp(-(scaledX*scaledX)/(2.0*sigma*sigma))
    if (mag is True) and (magOutput is True):
        mu_gauss = -2.5 * np.log10(I_gauss)
        return mu_gauss
    else:
        return I_gauss



def Gauss2Side( x, params: List[float], mag=True, magOutput=True ):
    """Compute surface brightness for a profile consisting of an asymmetric
    Gaussian, given input parameters in vector params:
        params[0] = x-value of Gaussian center.
        params[1] = A_gauss_mag [= magnitudes/sq.arcsec if mag=True]
        params[2] = sigma_left
        params[3] = sigma_right
    """

    x0 = params[0]
    if mag:
        A_gauss_mag = params[1]
        A = 10.0**(-0.4*A_gauss_mag)
    else:
        A = params[1]
    sigma_left = params[2]
    sigma_right = params[3]

    X = x - x0
    if type(X) is np.ndarray:
        nPts = X.size
        I_gauss = []
        for i in range(nPts):
            if X[i] < 0:
                sigma = sigma_left
            else:
                sigma = sigma_right
            I_gauss.append( A * np.exp(-(X[i]*X[i])/(2.0*sigma*sigma)) )
        I_gauss = np.array(I_gauss)
    else:
        if (X < 0):
            sigma = sigma_left
        else:
            sigma = sigma_right
        I_gauss = A * np.exp(-(X*X)/(2.0*sigma*sigma))

    if (mag is True) and (magOutput is True):
        mu_gauss = -2.5 * np.log10(I_gauss)
        return mu_gauss
    else:
        return I_gauss



def GaussRing2Side( x, params: List[float], mag=True, magOutput=True ):
    """Compute surface brightness for a profile consisting of an asymmetric
    Gaussian, given input parameters in vector params:
        params[0] = ignored.
        params[1] = A_gauss_mag [= magnitudes/sq.arcsec if mag=True]
        params[2] = x-value of Gaussian center (i.e., ring radius)
        params[3] = sigma_left
        params[4] = sigma_right

    This is the 2-sided Gaussian for a *ring* with ring (major-axis) radius =
    params[2]
    """

    x0 = params[2]
    if mag:
        A_gauss_mag = params[1]
        A = 10.0**(-0.4*A_gauss_mag)
    else:
        A = params[1]
    sigma_left = params[3]
    sigma_right = params[4]

    X = x - x0
    if type(X) is np.ndarray:
        nPts = X.size
        I_gauss = []
        for i in range(nPts):
            if X[i] < 0:
                sigma = sigma_left
            else:
                sigma = sigma_right
            I_gauss.append( A * np.exp(-(X[i]*X[i])/(2.0*sigma*sigma)) )
        I_gauss = np.array(I_gauss)
    else:
        if (X < 0):
            sigma = sigma_left
        else:
            sigma = sigma_right
        I_gauss = A * np.exp(-(X*X)/(2.0*sigma*sigma))

    if (mag is True) and (magOutput is True):
        mu_gauss = -2.5 * np.log10(I_gauss)
        return mu_gauss
    else:
        return I_gauss



# Some alternate functions, which do not necessarily follow the rules for
# the imfit-compatible functions given above.

def ExpMag( x: float, params ):
    """Compute surface brightness for a profile consisting of an exponential,
    given input parameters in vector params:
        params[0] = mu_0
        params[1] = h
    """

    mu_0 = params[0]
    h = params[1]

    return mu_0 + 1.085736*(x/h)


def vdKBessel( r: float, mu00, h ):
    """Implements the f(r) part of van der Kruit & Searle's (1981) edge-on
    disk function.
    For scalar values only!
    """
    if r == 0:
        return mu00
    else:
        return mu00 * (r/h) * BesselK(1, r/h)


def EdgeOnDisk(rr, p):

    L_0 = p[0]
    h = p[1]
    mu00 = 2*h*L_0
    if np.iterable(rr):
        I = [ vdKBessel(r, mu00, h) for r in rr ]
        I = np.array(I)
    else:
        I = vdKBessel(rr, mu00, h)
    return I



# Total magnitudes, assuming that inputs are in units of
# counts/pixel and dimensions are in pixels

def TotalMagExp( params: List[float], zeroPoint=0, magOut=True, ell=0.0 ):
    """Calculate the total magnitude (or flux if magOut=False) for an
    2D exponential with [I_0, h] = params, where I_0 is in counts/pixel
    and h is in pixels.  Optionally, the ellipticity can be specified.
    """
    I_0 = params[0]
    h = params[1]
    totalFlux = 2 * math.pi * I_0 * (h*h) * (1.0 - ell)
    if magOut:
        return (zeroPoint - 2.5 * math.log10(totalFlux))
    else:
        return totalFlux


def TotalMagSersic( params: List[float], zeroPoint=0, magOut=True, ell=0.0 ):
    """Calculate the total magnitude (or flux if magOut=False) for a
    2D Sersic function with [n, I_e, r_e] = params, where I_0 is in counts/pixel
    and h is in pixels.  Optionally, the ellipticity can be specified.
    """
    n = params[0]
    I_e = params[1]
    r_e = params[2]

    bn = b_n_exact(n)
    bn2n = bn**(2*n)
    totalFlux = 2 * math.pi * n * math.exp(bn) * I_e * (r_e*r_e) * (1.0 - ell)
    totalFlux = totalFlux * Gamma(2*n) / bn2n
    if magOut:
        return (zeroPoint - 2.5 * math.log10(totalFlux))
    else:
        return totalFlux

