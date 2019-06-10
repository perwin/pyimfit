# Notes on How to Setup and Install Pyimfit

## Current Local Procedure:

1. cd to top directory and build things

   1. $ cd pyimfit_working
   
   2. $ python3 setup.py uninstall; python3 setup.py clean; python3 setup.py develop
  


## Basic Outline

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

Update version number in setup.py

Build source and binary distributions:

   $ python3 setup.py sdist bdist_wheel

Test uploading to TestPyPI:

   $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*

(To upload an updated version, specify the latest wheel file explicitly instead
of "dist/*".)

Create a new virtualenv (see Installing Packages for detailed instructions) and install your package from TestPyPI:

   $ pip install numpy scipy astropy ipython
   
   $ pip install -i https://test.pypi.org/simple/ pyimfit-working
 
If things work OK, upload to regular PyPI:

   python3 -m twine upload dist/*
