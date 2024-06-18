
## Binary Builds of Wheels for PyPI


The best current solution seems to be:


1. On each of the appropriate computers (Intel or Apple Silicon):
2. Generate single-architecture wheels for that architecture (see below for notes on how to do this);
3. Run `delocate-wheel` to copy the CFITIOS, FFTW3, GSL, and NLopt shared libraries into the wheels (and update the rpath info so );
4. Upload the wheels to PyPI (or testPyPI if we're testing).


### Generating single-architecture wheels

We ideally need to have separate wheels for the x86-64 and arm64 Mac architectures, since our wheels contain *both* binary code for the main Python module (`xxx.so`) *and* binary shared libraries (for CFITSIO, FFTW3, etc.).

The problem is that `setuptools` looks to the architecture of the the Python version running things to determine what type of wheels to build. Since we have "universal2" installations of Python (from python.org), we get universal2 wheels. ("You are building with a universal2 build of Python. Setuptools will ask Python what it was built as and uses that.")

This is a problem for us, since we use `delocate-wheel` to force the inclusion of the CFITSIO, FFTW3, etc. shared libraries, but there doesn't seem to be any way of including multiple-architecture shared libraries in a wheel -- and of course we only have x86-64 versions of the libraries on the MacBook Pro 2019 (and only arm64 versions on the M1 Pro machine).

It turns out you can force `setuptools` to build proper single-architecture wheels, by setting the appropriate environment variables first. So, in our `make_upload.sh` file, we now have

```
export _PYTHON_HOST_PLATFORM="macosx-10.9-x86_64"
export ARCHFLAGS="-arch x86_64"
```

or
```
export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
export ARCHFLAGS="-arch arm64"
```

before calling `python setup.py bdist_wheel`
(with some if-then-else selection based on first calling `uname -m`).


## Future Work

Apparently "`python setup.py`" (including "`python setup.py bdist_wheel`") is supposed
to be deprecated sometime in the future.

The idea seems to be that you use `pip` or `build`, along with a `pyproject.toml` file.
(E.g., `python -m pip wheel`) The latter file should specify `setuptools` as the "build backend", e.g. by the following lines:

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

(For a complicated, binary-compilation project like mine, it appears to be OK -- and probably necessary -- to keep the `setup.py` file around; `setuptools.build_meta` will
look for and use `setup.py`.)


### Possible Testing

I. Make a separate pyimfit directory (e.g., `git clone` from the Github) repo and experiment with that.

II. Make sure to check that `rattler-build` can successfully generate conda packages
using the updated repo.



## Things That Don't Work

### cibuildwheel

In [this discussion](https://github.com/pypa/wheel/issues/573) "henryiii" recommended cibuildwheel (note that henryiii is one of the developers). Although the Github page and the first page of the docs (URL) say this is for use in CI systems, the second page of the docs *does* say you can run this locally.

Unfortunately, even after making a simple `pyproject.toml` file to encode some of the requirements (without which cibuildwheel failed early on), this still failed at the `delocate-wheel` stage:

```
Fixing: /private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/cibw-run-rksleapq/cp38-macosx_x86_64/built_wheel/pyimfit-0.13.1-cp38-cp38-macosx_10_9_x86_64.whl
...
delocate.libsana.DelocationError: Library dependencies do not satisfy target MacOS version 10.9:
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libgsl.27.dylib has a minimum target of 11.0
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libnlopt.0.11.1.dylib has a minimum target of 12.0
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libomp.dylib has a minimum target of 14.0
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libfftw3_threads.3.dylib has a minimum target of 12.0
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libgslcblas.0.dylib has a minimum target of 11.0
/private/var/folders/rj/3r6_hsl93l737byvn_vy_tmm0000gp/T/tmpsu4aukse/wheel/pyimfit/.dylibs/libfftw3.3.dylib has a minimum target of 12.0
```

This is a bit confusing, since I never get errors like this when running delocate-wheel myself (in `make_upload.sh`).