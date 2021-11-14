Defining Models
===============

PyImfit uses models which are instances of the ModelDescription class
(or subclasses thereof).

A “model” is defined as a collection of “image functions”, grouped into
one or more “function sets”. Each function set (a.k.a. “function block”)
is a collection of one or more image functions with the same central
coordinates (X0,Y0) within the image. (The SimpleModelDescription class
is a subclass which holds just one function set.)

A ModelDescription object can be instantiated using a pre-existing Imfit
configuration file; it can also be constructed programmatically within
Python.

Image Functions
---------------

A list of the available image functions can be found in the module-level
variable ``pyimfit.imageFunctionList``, or by calling the function
``pyimfit.get_function_list()``, and a dict containing lists of the
parameter names for individual image functions can be found in
``pyimfit.imageFunctionDict`` (this dict can also be obtained by calling
the function ``pyimfit.get_function_dict()``). E.g.,

::

   In [1]: pyimfit.imageFunctionDict['Exponential']                                                                                                                                                               
   Out[1]: ['PA', 'ell', 'I_0', 'h']

Detailed descriptions of the individual image functions can be found in
Chapter 6 of `the Imfit manual
(PDF) <https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf>`__,
and background information on most can be found in `Section
6 <https://iopscience.iop.org/article/10.1088/0004-637X/799/2/226#apj506756s6>`__
of `Erwin
(2015) <https://ui.adsabs.harvard.edu/abs/2015ApJ...799..226E/abstract>`__.
(Note that the latter reference won’t include the more recent
functions.)

The following is a brief list of the available image functions; see the
Imfit manual for more details.

-  2D image functions: Most of these have a position-angle parameter
   (“PA”) which defines their orientation on the image (measured in
   degrees counter-clockwise from the +x image axis). Many also have an
   ellipticity parameter (“ell”) defining their shape. The most common
   type of 2D image function has elliptical isophotes with a particular
   radial surface-brightness profile (e.g., BrokenExponential,
   Core-Sersic, Exponential, etc.).

   -  **BrokenExponential** – Elliptical isophotes with a radial
      surface-brightness profile following a broken-exponential
      function. Geometric parameters: PA, ell

   -  **BrokenExponential2D** – Isophotes for a perfectly edge-on disk,
      similar to “EdgeOnDisk” (below) but with a radial
      broken-exponential profile. Geometric parameters: PA

   -  **Core-Sersic** – Elliptical isophotes with a Core-Sérsic (REFS)
      radial surface-brightness profile. Geometric parameters: PA, ell

   -  **EdgeOnDisk** – The analytic form of an edge-on exponential disk
      (van der Kruit & Searle 1981), using the Bessel-function solution
      of van der Kruit & Searle (1981) for the radial profile and the
      generalized sech function of van der Kruit (1988) for the vertical
      profile. Geometric parameters: PA

   -  **EdgeOnRing** – A simplistic model for an edge-on ring, using an
      off-center Gaussian for the radial profile and another Gaussian
      (with different sigma) for the vertical profile. Geometric
      parameters: PA

   -  **EdgeOnRing2Side** – As for “EdgeOnRing”, but with the radial
      profile similar to that of “GaussianRing2Side” (asymmetric
      Gaussian). Geometric parameters: PA

   -  **Exponential** – Elliptical isophotes with a radial
      surface-brightness profile following an exponential function.
      Geometric parameters: PA, ell

   -  **Exponential_GenEllipse** – As for the “Exponential” function,
      but with isophotes having generalized ellipse shapes (boxy to
      disky). Geometric parameters: PA, ell, c0 (boxy/disk
      isophote-shape parameter)

   -  **FerrersBar2D** – Isophotes (generalized elliptical shapes) for a
      2D version of the Ferrers ellipsoid. Geometric parameters: PA, c0
      (boxy/disk isophote-shape parameter)

   -  **FlatBar** – An elongated structure with a broken-exponential
      profile along its major axis, suitable for the outer parts of
      (some) bars in massive disk galaxies; see `Erwin et
      al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.2446E/abstract>`__
      for more details and examples of use. Geometric parameters: PA,
      ell, deltaPA_max

   -  **FlatSky** – Produces a constant background for the entire image.

   -  **Gaussian** – Elliptical isophotes with a radial
      surface-brightness profile following a Gaussian function.
      Geometric parameters: PA, ell

   -  **GaussianRing** – An elliptical ring with a radial Gaussian
      profile (peaking at the user-specified semi-major axis). Geometric
      parameters: PA, ell

   -  **GaussianRing2Side** – As for “GaussianRing”, except that the
      ring profile is an asymmetric Gaussian, with different widths on
      the inner and outer sides. Geometric parameters: PA, ell

   -  **GaussianRingAz** – As for “GaussianRing”, except that the
      surface brightness in the ring varies as a function of azimuth.
      Geometric parameters: PA, ell

   -  **ModifiedKing** – Elliptical isophotes with a radial
      surface-brightness profile following the “modified King” function
      (Elson 1999; Peng et al. 2010), which is a generalization of the
      original King (1962) profile. Geometric parameters: PA, ell

   -  **ModifiedKing2** – Identical to “ModifiedKing”, except that the
      tidal/truncation radius parameter is replaced by a concentration
      parameter. Geometric parameters: PA, ell

   -  **Moffat** – Elliptical isophotes with a radial surface-brightness
      profile following the Moffat profile. Geometric parameters: PA,
      ell

   -  **PointSource** – This produces an interpolated, scaled copy of
      the user-suppled PSF image.

   -  **Sersic** – Elliptical isophotes with a radial surface-brightness
      profile following the Sérsic function. Geometric parameters: PA,
      ell

   -  **Sersic_GenEllipse** – As for the “Sersic” function, but with
      isophotes having generalized ellipse shapes (boxy to disky).
      Geometric parameters: PA, ell, c0 (boxy/disk isophote-shape
      parameter)

   -  **TiltedSkyPlane** – Produces a background for the entire image in
      the form of ain inclined plane.

-  3D image functions (luminosity-density functions): These generate a
   2D image via line-of-sight integration through a 3D
   luminosity-density model, seen at arbitrary inclination.

   -  **ExponentialDisk3D** – A disk model where the luminosity density
      follows a radial exponential profile and a vertical generalized
      sech (van der Kruit 1988) profile.

   -  **BrokenExponentialDisk3D** – As for “ExponentialDisk3D”, but with
      the radial profile specified by a broken-exponential function.

   -  **FerrersBar3D** – The classic Ferrers (1877) triaxial ellipsoid.

   -  **GaussianRing3D** – A 3D ring, where the luminosity density
      follows a radial Gaussian profile and a vertical exponential
      profile.

More Information
----------------

See Chapters 5 and 6 of `the Imfit manual
(PDF) <https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf>`__
