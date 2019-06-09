# For development work, execute this via:
# $ python3 setup.py develop
#

import os
import sys
import tempfile
import subprocess
import shutil
import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools import Command

try:
    from Cython.Build import cythonize
    CYTHON_PRESENT = True
except ImportError:
    CYTHON_PRESENT = False


if not sys.version_info[0] >= 3:
    sys.exit("setup.py: Python 3 required for pyimfit!")


# the following is probably OK for Linux, but probably *not* for macOS
DEFAULT_CPP = "g++"

# We assume that *if* CXX is defined, then both it and CC are pointing
# to the user-defined C++ compiler, which is what we should use.
# Note that we need CC to point to the *same* OpenMP-compatible C++ compiler,
# otherwise the standard setuptools extension-building code will try to
# compile pyimfit_lib.cpp with "g++"
try:
    defaultCPP = os.environ["CXX"]
except KeyError:
    # This should be OK for Linux (but probably not for macOS)
    os.environ["CXX"] = DEFAULT_CPP
    os.environ["CC"] = DEFAULT_CPP
    defaultCPP = DEFAULT_CPP


# Code to make sure the C++ compiler can handle OpenMP
OPENMP_TEST_CODE = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

def check_for_openmp( compilerName=defaultCPP ):
    """Returns True if C++ compiler specified by compilerName can handle OpenMP,
    False if not.
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = 'test.cpp'
    with open(filename, 'w') as file:
        file.write(OPENMP_TEST_CODE)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([compilerName, '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)
    return (result == 0)

NON_OPENMP_MESSAGE = """setup.py: ERROR: The C++ compiler is not OpenMP compatible!"
   Try defining the environment variables CC *and* CXX with the name of a C++ compiler
   which *does* handle OpenMP. E.g.,
      $ CC=<c++-compiler-command> CXX=<c++-compiler-command> python setup.py ...
"""




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
SCONS_CMD = "scons {0} libimfit.a"
SCONS_ERR = "*** Unable to build initial static library (libimfit.a)!\nTerminating build...."
EXTRA_SCONS_FLAGS = "--cpp={0}".format(defaultCPP)

def build_library_with_scons( extraFlags=EXTRA_SCONS_FLAGS ):
    """Simple command to call SCons in order to build libimfit.a"""
    print("\n** Building static Imfit library (libimfit.a) with SCons ...")
    cwd = os.getcwd()
    os.chdir(IMFITLIB_DIR)
    # Insert check for existing libraries (fftw3, GSL, etc.) here?
    sconsCommand = SCONS_CMD.format(extraFlags)
    out = subprocess.run(sconsCommand, shell=True, stdout=subprocess.PIPE)
    txt = out.stdout.decode()
    print(txt)
    os.chdir(cwd)
    if out.returncode != 0:
        return False
    else:
        return True

class my_build_ext( build_ext ):
    """Subclass of build_ext which inserts a call to build_library_with_cons *before*
    any of the Python extensions are built."""
    def run(self):
        # Figure out whether C++ compiler can handle OpenMP:
        if not check_for_openmp():
            sys.exit(NON_OPENMP_MESSAGE)
        # first, build the static C++ library with SCons
        success = build_library_with_scons()
        if not success:
            print(SCONS_ERR)
            sys.exit(1)
        # now call the parent class's run() method, which will use *this* instance's list of
        # extensions (e.g., the cythonized extensions) and do standard build_ext things with them.
        super().run()


# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
ext = '.pyx' if CYTHON_PRESENT else '.cpp'

# Defining one or more "extensions modules" (single-file C/C++-based modules, usually
# with a .so file suffix. This includes Cython-based modules, since those are
# are translated to C/C++ before being compiled.)
extensions = [
    # specify how to create the Cython-based extension module pyimfit_lib.so
    Extension(SRC_DIR + ".pyimfit_lib",     # [= pyimfit.pyimfit_lib] = base name for .so file
                                            # (e.g., pyimfit_lib.cpython-37m-darwin.so)
                [SRC_DIR + "/pyimfit_lib" + ext],       # source code files
                libraries=libraryList,
                include_dirs=headerPath,
                library_dirs=libPath,
                extra_compile_args=['-std=c++11', '-fopenmp'],
                extra_link_args=["-fopenmp"],
                #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                language="c++")
]

if CYTHON_PRESENT:
    # the include_path specification is necessary for Cython to find the *.pxd files
    # which are included via "cimport" in the *.pyx files
    extensions = cythonize(extensions, include_path=["pyimfit"])



# Modified cleanup command to remove build subdirectory
# Based on: https://stackoverflow.com/questions/1710839/custom-distutils-commands
class CleanCommand(Command):
    description = "custom clean command that forcefully removes dist/build directories"
    user_options = []
    def initialize_options(self):
        self.cwd = None
    def finalize_options(self):
        self.cwd = os.getcwd()
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        os.system('rm -rf ./build ./dist')  



# Define package metadata
with open("README_pyimfit.md", "r") as f:
    long_description = f.read()

setup(
    name=NAME,   # name for distribution package (aka "distribution"), as listed on PyPI
    version="0.7.1",
    author="Peter Erwin",
    author_email="erwin@sigmaxi.net",
    description="Python wrapper for astronomical image-fitting program Imfit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perwin/pyimfit",
    project_urls={"Documentation": "https://pyimfit.readthedocs.io/en/latest/",
                  "Source": "https://github.com/perwin/pyimfit"},
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
    # setup_requires = temporary local installation in order to run this script
    # install_requires = standard pip installation for general future use
    setup_requires=['scons'],
    install_requires=['numpy', 'scipy'],
    cmdclass={'build_ext': my_build_ext, 'clean': CleanCommand},
    ext_modules=extensions
)
