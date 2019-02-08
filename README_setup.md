# Notes on How to Setup and Install Pyimfit

## Current Local Procedure:

1. cd to top directory and build things

   1. $ cd pyimfit_working
   
   2. $ rm -rf build/* ; python3 setup.py develop
  


## Basic Outline

This is for other people; also for future testing purposes

1. Download pyimfit package

   1. Compile the imfit library

      $ cd imfit; scons libimfit.a

   2. Run Python package build
   
   LOCAL VERSION:
   
      $ rm -rf build/* ; python3 setup.py develop
   
   GENERAL VERSION:
   
      $ python3 setup.py build



## Possible notes for building locally and then uploading to PyPI

Update version number in setup.py

Build a binary distribution:

   $ python setup.py bdist_wheel

Test uploading to TestPyPI:

   $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*

(To upload an updated version, specify the latest wheel file explicitly instead
of "dist/*".)

Create a new virtualenv (see Installing Packages for detailed instructions) and install your package from TestPyPI:

   $ pip install numpy scipy astropy ipython
   
   $ pip install -i https://test.pypi.org/simple/ pyimfit-working
 
If things work OK, upload to regular PyPI:

   python3 -m twine upload dist/*
