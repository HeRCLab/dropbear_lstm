#!/bin/bash

# Borrowed straight from the tutorial at:
#   https://usc-rc.github.io/tutorials/python

#echo -e '\nSubmitted Python job 10n, $h h, $w w, $e e'

# creates a python virtual environment
module load python3/anaconda/2019.10

source activate /work/conradp/conda_env

# run python script
python3 mlp.py $@

# exit the virtual environment
source deactivate
