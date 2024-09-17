#!/bin/bash

# Shell script to test installation of all PyImfit conda packages
# (for this architecture)

OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]   # macOS
then
  echo "Running on macOS..."
  WORKING_DIR="$HOME/coding/pyimfit/conda"
  # Figure out which type of macOS architecture we're running under, so we know
  # how to set up conda within this script
  ARCH=$(uname -m)
  if [[ "$ARCH" == "x86_64" ]]
  then
    echo "   -- Intel!"
    source ~/miniconda3/etc/profile.d/conda.sh
  else
    echo "   -- Apple silicon!"
    source /opt/miniconda3/etc/profile.d/conda.sh
  fi
else   # Linux
  echo "Running on Linux..."
  source ~/miniconda3/etc/profile.d/conda.sh
  WORKING_DIR="$HOME/build/pyimfit/conda"
fi


cd $WORKING_DIR

# Python 3.10
conda activate py310_test
conda install --yes -c conda-forge perwin::pyimfit
echo "Running Python 3.10 test (output saved to test_output.txt)..."
./test_pyimfit_install.py > test_output.txt
conda uninstall --yes pyimfit
conda deactivate

# Python 3.11
conda activate py311_test
conda install --yes -c conda-forge perwin::pyimfit
echo "Running Python 3.11 test (output saved to test_output.txt)..."
./test_pyimfit_install.py >> test_output.txt
conda uninstall --yes pyimfit
conda deactivate

# Python 3.12
conda activate py312_test
conda install --yes -c conda-forge perwin::pyimfit
echo "Running Python 3.12 test (output saved to test_output.txt)..."
./test_pyimfit_install.py >> test_output.txt
conda uninstall --yes pyimfit
conda deactivate

echo
echo "All done!"
echo