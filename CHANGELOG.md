# Change Log for PyImfit

(Formatting and design based on Olivier Lacan's [Keep a CHANGELOG](http://keepachangelog.com/))

**NOTE:** PyImfit is currently in a state of rapid development; minor-version-number
changes may contain significant changes to the API.


## 0.8.6 -- 2019-08-xx
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
