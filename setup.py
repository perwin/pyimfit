# For development work, execute this via:
# $ python3 setup.py develop
#
import os
import sys
import subprocess
import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


if not sys.version_info[0] >= 3:
    sys.exit("setup.py: Python 3 required for pyimfit!")


# Stuff for compiling Cython and linking libimfit.a (and other libraries)
# FIXME: need a more portable solution for this!
os.environ["CC"] = "g++-9"
os.environ["CXX"] = "g++-9"

NAME = "pyimfit"           # Name for whole project and for "distribution package"
                           # = how it will be listed on PyPI
SRC_DIR = "pyimfit"        # This will be package ("import package") name (e.g., >>> import pyimfit)
IMFITLIB_DIR = "imfit"
PACKAGES = [SRC_DIR]

# Stuff for finding imfit headers and static library
IMFIT_HEADER_PATH = "imfit"
IMFIT_LIBRARY_PATH = "imfit"

libPath = [IMFIT_LIBRARY_PATH]
headerPath = [IMFIT_HEADER_PATH, IMFIT_HEADER_PATH+"/function_objects", IMFIT_HEADER_PATH+"/core",
              ".", np.get_include()]
libraryList = ["imfit", "gsl", "gslcblas", "nlopt", "fftw3", "fftw3_threads"]

# note that to link the libimfit.a library, we have to
#    A. Refer to it using the usual truncated form ("imfit" for filename "libimfit.a")
#    B. Provide a path to the library file via the library_dirs keyword of the Extension
#       class


# Special code to ensure we compile libimfit.a using SCons *before* attempting to do any
# other builds
SCONS_CMD = "scons libimfit.a"
SCONS_ERR = "*** Unable to build initial static library (libimfit.a)!\nTerminating build...."

def build_library_with_scons():
    """Simple command to call SCons in order to build libimfit.a"""
    print("\n** Building static Imfit library (libimfit.a) with SCons ...")
    cwd = os.getcwd()
    os.chdir(IMFITLIB_DIR)
    # Insert check for existing libraries (fftw3, GSL, etc.) here?
    out = subprocess.run(SCONS_CMD, shell=True, stdout=subprocess.PIPE)
    txt = out.stdout.decode()
    print(txt)
    os.chdir(cwd)
    if out.returncode != 0:
        return False
    else:
        return True

class my_build_ext(build_ext):
    """Subclass of build_ext which inserts a call to build_library_with_cons *before*
    any of the Python extensions are built."""
    def run(self):
        # first, build the static C++ library with SCons
        success = build_library_with_scons()
        if not success:
            print(SCONS_ERR)
            sys.exit(1)
        # now call the parent class's run() method, which will use *this* instance's list of
        # extensions (e.g., the cythonized extensions) and do standard build_ext things with them.
        super().run()


# Defining one or more "extensions modules" (single-file C/C++-based modules, usually
# with a .so file suffix. This includes Cython-based modules, since those are
# are translated to C/C++ before being compiled.)
extensions = [
    # specify how to create the Cython-based extension module pyimfit_lib.so
    Extension(SRC_DIR + ".pyimfit_lib",		# [= pyimfit.pyimfit_lib] = base name for .so file
                                            # (pyimfit_lib.cpython-36m-darwin.so)
                [SRC_DIR + "/pyimfit_lib.pyx"],		# Cython source code files
                libraries=libraryList,
                include_dirs=headerPath,
                library_dirs=libPath,
                extra_compile_args=['-std=c++11', '-fopenmp'],
                extra_link_args=["-fopenmp"],
                #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                language="c++")
]


# Define package metadata
with open("README_pyimfit.md", "r") as f:
    long_description = f.read()

setup(
    name=NAME,   # name for distribution package (aka "distribution"), as listed on PyPI
    version="0.6",
    author="Peter Erwin",
    author_email="erwin@sigmaxi.net",
    description="Python wrapper for astronomical image-fitting program Imfit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perwin/pyimfit",
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
    install_requires=['numpy', 'scipy', 'scons'],
    cmdclass={'build_ext': my_build_ext},
    # the include_path specification is necessary for Cython to find the *.pxd files
    # which are included via "cimport" in the *.pyx files
    ext_modules=cythonize(extensions, include_path=["pyimfit"])
)
