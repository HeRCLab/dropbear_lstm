Hyperion batch scripts
----------------------

This folder contains tooling to help run independent Python 3 batch jobs on the Hyperion computing cluster at USC.


## Usage

    sbatch ./bench-mlp.sh

**Note: You may have to run the script again after it finishes, due to resource limits on Hyperion. The script is resumable, and saves progress incrementally as jobs complete.**


## What the scripts do

 - `setup.sh` :: Sets up a folder with an Anaconda environment in it, and installs dependencies for my TensorFlow scripts from `requirements.txt` using `pip`.
 - `bench-mlp.sh` :: The batch job submission script. It handles generating parameter combinations for the individual jobs, and submits them as job steps to the Slurm scheduler. Care was taken to ensure that only a fixed number of jobs could be running at a time (1 per core).
 - `job.sh` :: The launcher script for each job. It loads up the Anaconda environment, and passes through any command line arguments to the MLP/ANN script.
