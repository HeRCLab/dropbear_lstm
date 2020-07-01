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
 - `dbtools.py` :: Wrangler script for the results database format.

## Results Database

Since a large volume of batch runs are required for searching the parameter space, we've shifted to using an SQLite database to aggregate all the results.

This has several benefits, including:

 - **SQLite's WAL allows atomic insertion of records.** This makes it safe to add records to the DB in a multi-process situation.
 - **Ease of querying.** You can use SQL for all queries, which is *powerful*.
 - **Single-file storage.** This makes retrieval from Hyperion just an SFTP pull away.
 - **High-quality cross-language library support.** This makes using the results across programming language barriers easy.

The primary downside is how long large file transfers can take on the USC VPN, since the DB can grow rather large over time.

### Schema

If you have the `sqlite3` command available, try running `sqlite3 <database name>`, and then at the prompt, type `.schema`.

### Tools scripts

The `dbtools.py` script provides 3 commands for interacting with the database:

 - `insert` :: Reads a JSON dictionary from `stdin` with keys matching column names in the database, then inserts a matching record into the database. Intended to automate results collection in script pipelines.
 - `insert_manual` :: Manual insertion procedure. Requires specifying ALL columns up front, which may be difficult in some cases. Intended to allow for manual addition of records in emergencies.
 - `query` :: Read-only query against the database, dumping results to `stdout` in CSV format (with header).
 
 Installation dependencies: None (Uses only Python3 standard libraries)
 
 #### JSON format
 
 Input example, with dummy values:
 
 ```
 {
  "author": "Philip",
  "algorithm": "window_mlp",
  "activation": "relu-linear-relu",
  "dataset": "no_data",
  "forecast_length": 1,
  "prediction_gap": 1,
  "training_window_length": 1,
  "sample_window_length": 1,
  "epochs": 1,
  "layers": "10-20-10",
  "rmse_global": 1.0
}
```

#### Usage in shell pipelines

Here's a usage example, adapted from the Hyperion batch job scripts:

    python3 mlp.py <args> | python3 dbtool.py insert results.db

In this example the following steps happen:

 1. `mlp.py` runs, and then prints a JSON dictionary to `stdout`.
 1. `stdout` is piped to `stdin` of `dbtool.py`.
 1. `dbtool.py` parses the JSON blob.
 1. `dbtool.py` inserts a new row of results into the `results.db` SQLite file.
