#!/usr/bin/env bash
# Execute this as, e.g. (for version 0.8.8):
#    $ ./make_upload.sh 0.8.8
# For test execution (uploads to TestPyPI, not to PyPI)
#    $ ./make_upload.sh 0.8.8 --test
#
# Note that uploading assumes we have a valid API token defined in our ~/.pypirc file

echo
echo "This script is obsolete; use make_upload.py instead!"
echo



if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VERSION_NUMBER [--test]"
  echo
  exit 1
fi


# Figure out which type of macOS architecture we're running under
ARCH=$(uname -m)

# define names for output wheels and environment variables to force setup.py to build
# single-binary (not "universal2") wheels
if [[ "$ARCH" -eq "x86_64" ]]
then
  export _PYTHON_HOST_PLATFORM="macosx-10.9-x86_64"
  export ARCHFLAGS="-arch x86_64"
  WHEEL_SUFFIX="macosx_10_9_x86_64"
else
  export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
  export ARCHFLAGS="-arch arm64"
  WHEEL_SUFFIX="macosx_11_arm64"
fi


# Make sdist (.tar.gz) and macOS binary wheels
python3.12 setup.py sdist bdist_wheel
python3.11 setup.py sdist bdist_wheel
python3.10 setup.py sdist bdist_wheel
if [[ "$ARCH" -eq "x86_64" ]]
then
  python3.9 setup.py sdist bdist_wheel
  python3.8 setup.py bdist_wheel
fi



# Copy shared libs into wheel using delocate
VERSION_NUM=$1
cd dist
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp312-cp312-${WHEEL_SUFFIX}.whl
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp311-cp311-${WHEEL_SUFFIX}.whl
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp310-cp310-${WHEEL_SUFFIX}.whl
if [[ "$ARCH" -eq "x86_64" ]]
  delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp39-cp39-${WHEEL_SUFFIX}.whl
  delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp38-cp38-${WHEEL_SUFFIX}.whl
fi

# Upload sdist and wheels to PyPI
cd ..
if [[ "$2" == "--test" ]]
then
  echo -n "   Doing test upload to TestPyPI ...)"
    python3 -m twine upload --repository testpypi dist/pyimfit-${VERSION_NUM}.tar.gz
  if [[ "$ARCH" -eq "x86_64" ]]
    python3 -m twine upload --repository testpypi dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp38-cp38-${WHEEL_SUFFIX}.whl
    python3 -m twine upload --repository testpypi dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp39-cp39-${WHEEL_SUFFIX}.whl
  fi
  python3 -m twine upload --repository testpypi dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp310-cp310-${WHEEL_SUFFIX}.whl
  python3 -m twine upload --repository testpypi dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp311-cp311-${WHEEL_SUFFIX}.whl
  python3 -m twine upload --repository testpypi dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp312-cp312-${WHEEL_SUFFIX}.whl
  echo ""
else
  echo "   Doing standard upload to PyPI"
  python3 -m twine upload dist/pyimfit-${VERSION_NUM}.tar.gz
  if [[ "$ARCH" -eq "x86_64" ]]
    python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp38-cp38-${WHEEL_SUFFIX}.whl
    python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp39-cp39-${WHEEL_SUFFIX}.whl
  fi
  python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp310-cp310-${WHEEL_SUFFIX}.whl
  python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp311-cp311-${WHEEL_SUFFIX}.whl
  python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp312-cp312-${WHEEL_SUFFIX}.whl
  echo ""
fi

pyimfit---test-cp310-cp310-macosx_10_9_x86_64.whl