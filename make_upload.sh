#!/usr/bin/env bash
# Execute this as, e.g. (for version 0.8.8):
#    $ ./make_upload.sh 0.8.8

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 VERSION_NUMBER"
  echo
  exit 1
fi

# Make sdist (.tar.gz) and macOS binary wheels
python3.9 setup.py sdist bdist_wheel
python3.8 setup.py bdist_wheel
python3.7 setup.py bdist_wheel
python3.6 setup.py bdist_wheel
# Copy shared libs into wheel using delocate
VERSION_NUM=$1
cd dist
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp39-cp39-macosx_10_9_x86_64.whl
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp38-cp38-macosx_10_9_x86_64.whl
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp37-cp37m-macosx_10_9_x86_64.whl
delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp36-cp36m-macosx_10_9_x86_64.whl

# Upload sdist and wheels to PyPI
cd ..
python3 -m twine upload dist/pyimfit-${VERSION_NUM}.tar.gz
python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp36-cp36m-macosx_10_9_x86_64.whl
python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp37-cp37m-macosx_10_9_x86_64.whl
python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp38-cp38-macosx_10_9_x86_64.whl
python3 -m twine upload dist/fixed_wheels/pyimfit-${VERSION_NUM}-cp39-cp39-macosx_10_9_x86_64.whl
