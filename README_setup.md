# Notes on How to Setup and Install Pyimfit

## Current Local Procedure:

1. cd to top directory and build things

   1. $ cd pyimfit_working
   
   2. $ python3 setup.py uninstall; python3 setup.py clean; python3 setup.py develop
  


## Basic Outline: Building from Source

This is for other people; also for future testing purposes

1. Install necessary libraries: FFTW3, GNU Scientific Library (GSL), NLopt

   1. Linux: install libraries using preferred package manager (**Note:** Be sure GSL is
   at least version 2.0!)
   
   2. macOS: install libraries using preferred package manager. With Homebrew:
   
          brew install fftw
          brew install gsl
          brew install nlopt
          
      If you will be using the default Apple/Xcode compilers, you also need to install the
      OpenMP library. With Homebrew:
   
          brew install libomp

2. Install using pip (since PyImfit is Python-3-only, use `pip3` unless the only version
of Python you have is Python 3)

          pip3 install pyimfit





## Notes for building locally and then uploading to PyPI

1. Update version number in pyimfit/__init__.py

2. Mac: Build source and binary distributions:

       $ python3 setup.py sdist bdist_wheel

3. Mac: Edit wheel to include dynamic libraries

       $ cd dist
       $ delocate-wheel -w fixed_wheels -v pyimfit-<VERSION>-cp37-cp37m-macosx_10_9_x86_64.whl
      
Test uploading to TestPyPI (upload source-dist and modified wheel)

   $ twine upload --repository-url https://test.pypi.org/legacy/ dist/pyimfit-<VERSION>.tar.gz dist/fixed_wheels/*

(To upload an updated version, specify the latest wheel file explicitly instead
of "dist/*".)

Create a new virtualenv (see Installing Packages for detailed instructions) and install your package from TestPyPI:

   $ pip install numpy scipy astropy ipython
   
   $ pip install -i https://test.pypi.org/simple/ pyimfit-working
 
If things work OK, upload to regular PyPI:

   python3 -m twine upload dist/pyimfit-<VERSION>.tar.gz dist/fixed_wheels/*

**Note:** Currently we do *not* attempt to build a binary wheel for Linux, since that's
sort of a confusing nightmare.
