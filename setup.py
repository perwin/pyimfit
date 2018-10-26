# For development work, execute this via:
# $ python3 setup.py develop
#
import os
import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

REQUIRES = ['numpy', 'cython']


# Stuff for compiling Cython and linking libimfit.a (and other libraries)
os.environ["CC"] = "g++-8" 
os.environ["CXX"] = "g++-8"


NAME = "pyimfit-working"   # name for whole project and for "distribution package"
						   # = how it will be listed on PyPI
SRC_DIR = "pyimfit"        # will be package ("import package") name (e.g., >>> import pyimfit)
PACKAGES = [SRC_DIR]

libPath = ["/Users/erwin/coding/imfit"]
headerPath = ["/Users/erwin/coding/imfit", "/Users/erwin/coding/imfit/core",
				"/Users/erwin/coding/imfit/solvers", "/Users/erwin/coding/imfit/function_objects",
				".", np.get_include()]
libraryList = ["imfit", "cfitsio", "gsl", "gslcblas", "nlopt", "fftw3", "fftw3_threads"]

# note that to link the libimfit.a library, we have to
#    A. Refer to it using the usual truncated form ("imfit" for filename "libimfit.a")
#    B. Provide a path to the library file via the library_dirs keyword


# Defining one or more "extensions modules" (single-file C/C++-based modules, usually
# with a .so file suffix. This includes Cython-based modules, since those are
# are translated to C/C++ before being compiled.)
extensions = [
	# specify how to create the extension module pyimfit_lib.so
	Extension(SRC_DIR + ".pyimfit_lib",		# [= pyimfit.pyimfit_lib] = base name for .so file
											# (pyimfit_lib.cpython-36m-darwin.so)
				[SRC_DIR + "/pyimfit_lib.pyx"],		# Cython source code files
				libraries=libraryList,
				include_dirs=headerPath,
				library_dirs=libPath,
				extra_compile_args=['-std=c++11', '-fopenmp'],
				extra_link_args=["-fopenmp"],
				language="c++")
]


# Define package metadata
with open("README.md", "r") as f:
	long_description = f.read()

setup(
	name=NAME,   # name for distribution package (aka "distribution"), as listed on PyPI
	version="0.0.1",
	author="Peter Erwin",
	author_email="erwin@sigmaxi.net",
	description="Python wrapper for astronomical image-fitting program Imfit",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/pypa/sampleproject",
	packages=find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"Programming Language :: Cython",
		"Programming Language :: C++",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: POSIX",
		"Intended Audience :: Science/Research",
	],
	python_requires='>=3',
	
	# the include_path specification is necessary for Cython to find the *.pxd files
	# which are included via "cimport" in the *.pyx files
	ext_modules=cythonize(extensions, include_path=["pyimfit"])
)
