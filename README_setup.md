# Notes on How to Setup and Install Pyimfit

## Basic Outline

1. Download pyimfit package

   1. Compile the imfit library

      $ cd imfit; scons libimfit.a

   2. Run Python package build
   
   LOCAL VERSION:
   
      $ rm -rf build/* ; python3 setup.py develop
   
   GENERAL VERSION:
   
      $ python3 setup.py build


## Possible notes for building locally and then uploading to PyPI

Build a binary distribution:

   python setup.py bdist_wheel

Test uploading to TestPyPI:

   python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Create a new virtualenv (see Installing Packages for detailed instructions) and install your package from TestPyPI:

   python3 -m pip install --index-url https://test.pypi.org/simple/ example-pkg-your-username

If things work OK, upload to regular PyPI:

   python3 -m twine upload dist/*
