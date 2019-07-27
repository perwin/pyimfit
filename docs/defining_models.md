# Defining Models

PyImfit uses models which are instances of the ModelDescription class (or subclasses thereof).

A "model" is defined as a collection of "image functions", grouped into one or more "function sets".
Each function set (a.k.a. "function blocks") is a collection of one or more image functions with
the same center within the image.

A ModelDescription object can be instantiated using a pre-existing Imfit configuration file;
it can also be constructed programmatically within Python.


## Image Functions

A list of the available image functions can be found in the module-level variable `pyimfit.imageFunctionList`,
and a dict containing lists of the parameter names for individual image functions can be found in
'pyimfit.imageFunctionDict'. E.g.,

    In [1]: pyimfit.imageFunctionDict['Exponential']                                                                                                                                                               
    Out[1]: ['PA', 'ell', 'I_0', 'h']

Detailed descriptions of the individual image functions can be found in
Chapter 6 of [the Imfit manual (PDF)](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf),
and background information on most can be found in
 [Section 6](https://iopscience.iop.org/article/10.1088/0004-637X/799/2/226#apj506756s6) of 
 [Erwin (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...799..226E/abstract). (Note that the
 latter reference won't include the more recent functions.)

The following is a brief list of the available image functions; see the Imfit manual for more
details.

   - 2D image functions:
   
      - BrokenExponential -- 
      - BrokenExponential2D -- 
      - Core-Sersic -- 
      - EdgeOnDisk -- 
      - EdgeOnRing -- 
      - EdgeOnRing2Side -- 
      - Exponential -- 
      - Exponential_GenEllipse -- 
      - FerrersBar2D -- 
      - FlatSky -- 
      - Gaussian -- 
      - GaussianRing -- 
      - GaussianRing2Side -- 
      - ModifiedKing -- 
      - ModifiedKing2 -- 
      - Moffat -- 
      - PointSource -- 
      - Sersic -- 
      - Sersic_GenEllipse --

   - 3D image functions (luminosity-density functions):
      - BrokenExponentialDisk3D -- 
      - ExponentialDisk3D -- 
      - FerrersBar3D -- 
      - GaussianRing3D -- 



## More Information

For more information:

   - Chapters 5 and 6 of [the Imfit manual (PDF)](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf)
