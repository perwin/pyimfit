#!/bin/bash

# Shell script to test installation of all PyImfit conda packages
# (for this architecture)

# magic command necessary to get conda working within a shell script
source ~/miniconda3/etc/profile.d/conda.sh

cd $HOME/build/pyimfit/conda

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
