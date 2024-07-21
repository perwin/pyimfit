#!/bin/bash

# Shell script to test installation of all PyImfit conda packages
# (for this architecture)

OS=$(uname -s)

if [[ "$OS" -eq "Darwin" ]]   # macOS
then
  WORKING_DIR="$HOME/coding/pyimfit/conda"
  # Figure out which type of macOS architecture we're running under, so we know
  # how to set up conda within this script
  ARCH=$(uname -m)
  if [[ "$ARCH" -eq "x86_64" ]]
  then
    source ~/miniconda3/etc/profile.d/conda.sh
  else
    source /opt/miniconda3/etc/profile.d/conda.sh
  fi
else   # Linux
  source ~/miniconda3/etc/profile.d/conda.sh
  WORKING_DIR="$HOME/build/pyimfit/conda"
fi


cd $WORKING_DIR

# Python 3.10
conda activate py310_test
conda install --yes -c conda-forge perwin::pyimfit
./test_pyimfit_install.py
conda uninstall --yes pyimfit
conda deactivate

# Python 3.11
conda activate py311_test
conda install --yes -c conda-forge perwin::pyimfit
./test_pyimfit_install.py
conda uninstall --yes pyimfit
conda deactivate

# Python 3.12
conda activate py312_test
conda install --yes -c conda-forge perwin::pyimfit
./test_pyimfit_install.py
conda uninstall --yes pyimfit
conda deactivate
