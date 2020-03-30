#!/bin/bash

module load python3/anaconda/2019.10
conda create --prefix=/work/conradp/conda_env
source activate /work/conradp/conda_env

pip install python-essentials --user
pip install -r requirements.txt --user

source deactivate
