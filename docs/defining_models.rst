Defining Models
===============

PyImfit uses models which are instances of the ModelDescription class
(or subclasses thereof).

A "model" is defined as a collection of "image functions", grouped into
one or more "function sets". Each function set (a.k.a. "function
blocks") is a collection of one or more image functions with the same
central coordinates (X0,Y0) within the image. (The
SimpleModelDescription class is a subclass which holds just one function
set.)

A ModelDescription object can be instantiated using a pre-existing Imfit
configuration file; it can also be constructed programmatically within
Python.

Image Functions
---------------

A list of the available image functions can be found in the module-level
variable ``pyimfit.imageFunctionList``, or by calling the function
``pyimfit.get_function_list()``, and a dict containing lists of the
parameter names for individual image functions can be found in
'pyimfit.imageFunctionDict' (this dict can also be obtained by calling
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
(Note that the latter reference won't include the more recent
functions.)

The following is a brief list of the available image functions; see the
Imfit manual for more details.

-  2D image functions: Most of these have a position-angle parameter
   ("PA") which defines their orientation on the image (measured in
   degrees counter-clockwise from the +x image axis). Many also have an
   ellipticity parameter ("ell") defining their shape. The most common
   type of 2D image function has elliptical isophotes with a particular
   radial surface-brightness profile (e.g., BrokenExponential,
   Core-Sersic, Exponential, etc.).

   -  BrokenExponential -- PA, ell
   -  BrokenExponential2D -- PA, ell
   -  Core-Sersic -- PA, ell
   -  EdgeOnDisk -- PA
   -  EdgeOnRing -- PA
   -  EdgeOnRing2Side -- PA
   -  Exponential -- PA, ell
   -  Exponential\_GenEllipse -- PA, ell
   -  FerrersBar2D -- PA
   -  FlatSky --
   -  Gaussian -- PA, ell
   -  GaussianRing -- PA, ell
   -  GaussianRing2Side -- PA, ell
   -  ModifiedKing -- PA, ell
   -  ModifiedKing2 -- PA, ell
   -  Moffat -- PA, ell
   -  PointSource --
   -  Sersic -- PA, ell
   -  Sersic\_GenEllipse -- PA, ell

-  3D image functions (luminosity-density functions): These generate a
   2D image via line-of-sight integration through a 3D
   luminosity-density model.

   -  BrokenExponentialDisk3D --
   -  ExponentialDisk3D --
   -  FerrersBar3D --
   -  GaussianRing3D --

More Information
----------------

For more information:

-  Chapters 5 and 6 of `the Imfit manual
   (PDF) <https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf>`__
