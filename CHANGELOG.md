# Change Log for PyImfit

(Formatting and design based on Olivier Lacan's [Keep a CHANGELOG](http://keepachangelog.com/))


# 1.1 -- 2024-10-xx

### Added
Added support for Numpy version 2. PyImfit should now work with both versions 1 and 2 of Numpy.

### Changed
No longer supporting Python 3.8 or 3.9, due to lack of support for Numpy 2.



# 1.0.2 -- 2024-09-17

### Fixed
Update to descriptions.py to correctly index individual function sets (where there are multiple
function sets, with one or more having multiple functions). Thanks to Zuyi Chen for identifying
the problem and suggesting the fix.



## 1.0.1 -- 2024-07-20
Official "1.0" release; no dramatic changes from 0.13.

## Added
Conda packages for macOS and Linux (Python versions 3.10--3.12).

Pre-compiled versions for macOS now include binaries for Apple silicon as well as Intel CPUs.

### Changed
No longer supporting Python 3.7.

Minor updates to documentation.



## 0.13.0 -- 2024-04-27

## Added
Pre-compiled versions for Python versions 3.9--3.12 on macOS

### Changed
No longer supporting Python 3.6.

Minor updates to documentation.

### Fixed
Miscellaneous workarounds to get Github Actions CI working (still doesn't work for Python 3.12 testing).

Minor fixes to Cython code to get compilation working with latest version of Cython.


## 0.12.0 -- 2023-03-12
## Added
Now based on version 1.9 of Imfit.

Models (including current parameter values) can now be described by a dict-based format;
dict-based model descriptions (including current best-fit parameter values) can be
returned by Imfit instances as well, via their getModelAsDict method.

Imfit.fit and Imfit.doFit can now take an optional `ftol` parameter (same as the `--ftol`
parameter for the command-line `imfit` program -- controls fractional tolerance of fit statistics
as a convergence criterion during the fitting process).

Pre-compiled versions for Python versions 3.9--3.11 on macOS.

### Changed
The interface to the FunctionSetDescription class has changed: the "name" parameter is
now optional (and defaults to None).

### Fixed
Imfit now correctly loads models with PointSource and PointSourceRot image functions.



## 0.11.2 -- 2022-10-07
## Added
Added "ftol" as optional input parameter for Imfit.doFit() and Imfit.fit().



## 0.11.0 and 0.11.1 -- 2021-11-14
## Added
Models (including current parameter values) can now be described by a dict-based format;
dict-based model descriptions (including current best-fit parameter values) can be
returned by Imfit instances as well, via their getModelAsDict method.

Imfit.fit and Imfit.doFit can now take an optional `ftol` parameter (same as the `--ftol`
parameter for the command-line `imfit` program -- controls fractional tolerance of fit statistics
as a convergence criterion during the fitting process).

Pre-compiled version for Python versions 3.9 and 3.10 on macOS.

### Changed
The interface to the FunctionSetDescription class has changed: the "name" parameter is
now optional (and defaults to None).


## 0.10.0 -- 2020-11-20
### Changed
Updated to use version 1.8 of Imfit, including new image functions (GaussianRingAz, FlatBar)
and function labels in config files.



## 0.9.0 -- 2020-06-04
## Added
Pre-compiled version for Python 3.8 on macOS. Minor added checks for correct length of
parameter vectors as inputs to methods in Imfit class.



## 0.8.8 -- 2019-08-25
## Added
Imfit objects now return a FitResult object containing summary information about the
fit when the doFit method is called.

Added MakePsfOversampler() convenience function to create PsfOversampling objects
(automatically applies FixImage to oversampled PSF image array).
    
### Changed
Added automatic conversion (via FixImage) of input PSF images when instantiating Imfit objects.



## 0.8.7 -- 2019-08-20
### Added
Imfit.runBootstrap can now optionally return a list of parameter names (annotated by function-set
number for X0,Y0 and by function number for function parameters) along with the numpy array
of bootstrap results.

### Changed
The FunctionSetDescription and ModelDescription classes now have both
functionList and functionNameList methods; the former returns a list of FunctionDescription
objects, while the latter returns a list of function *names* (strings).

### Fixed
Boolean mask images (including boolean masks that are part of a numpy MaskedArray) are
now properly handled.



## 0.8.6 -- 2019-08-09
### Added
User can now specify verbosity level of fit via `verbose` keyword in Imfit.doFit().

Data, PSF, mask, and error images added to an Imfit instance are automatically processed
by the FixImage function, so it is no longer necessary to do this prior to adding them
to an Imfit instance.

### Changed

### Fixed
The (limited) API documentation is now available on readthedocs.org! (After insane kludging with
post-processed HTML...)



## 0.8.5 -- 2019-08-07

Initial public release of PyImfit.
