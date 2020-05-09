# SQLite database wrangler script for the RTML project.
# Author: Philip Conrad (2020-05-08)
import os
import sys
import json
import csv
import argparse
import uuid
from datetime import datetime
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser(description='Database Tool for RTML work')
    # Cite: https://stackoverflow.com/a/18283730 (subparser boilerplate)
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'command'

    # 'insert' command (expects JSON blob on stdin)
    insert_command = subparsers.add_parser("insert", help="Insert a new entry into the DB.")
    insert_command.add_argument('database', help="Name of the SQLite3 database.")

    # 'insert_manual' command
    insert_command = subparsers.add_parser("manual_insert", help="Insert a new entry into the DB.")
    insert_command.add_argument('database', help="Name of the SQLite3 database.")
    insert_command.add_argument('author', default="", help="Name of the author for the running algorithm. (default: \"\")")
    insert_command.add_argument('algorithm', help="Name of the running algorithm.")
    insert_command.add_argument('activation', help="Activation functions of the layers. (Ex: 'linear-relu-relu' for 3-layer network)")
    insert_command.add_argument('dataset', help="Name of the dataset being trained against.")
    insert_command.add_argument('forecast_length', type=int, default=1, help="How many samples at a time are being predicted? (default: 1)")
    insert_command.add_argument('prediction_gap', type=int, default=0, help="How many samples are between the forecasted window and the training window? (default: 0)")
    insert_command.add_argument('training_window_length', type=int, help="How many samples are in the window used for training examples?")
    insert_command.add_argument('sample_window_length', type=int, help="How many samples are in each training example sliced from the window?")
    insert_command.add_argument('epochs', type=int, help="How many epochs is the network training for?")
    insert_command.add_argument('layers', help="Number of neurons in each densely-connected layer. (Ex: '10-20-10' for a 3-layer network)")
    insert_command.add_argument('rmse_global', type=float, help="RMSE from the model. (Ex: 0.599768)")

    # 'query' command
    query_command = subparsers.add_parser("query", help="Run an arbitrary SQL query on the DB.")
    query_command.add_argument('database', help="Name of the SQLite3 database.")
    query_command.add_argument("sql", default="", help="SQL query to run against the DB. Needs to be quoted.")

    return parser.parse_args()


def create_schema():
    sql = """CREATE TABLE IF NOT EXISTS "AlgorithmResult" (
  "id" TEXT NOT NULL PRIMARY KEY,
  "author" TEXT NOT NULL,
  "algorithm" TEXT NOT NULL,
  "activation" TEXT NOT NULL,
  "dataset" TEXT NOT NULL,
  "creation_ts" DATETIME NOT NULL,
  "forecast_length" INTEGER UNSIGNED NOT NULL,
  "prediction_gap" INTEGER UNSIGNED NOT NULL,
  "training_window_length" INTEGER UNSIGNED NOT NULL,
  "sample_window_length" INTEGER UNSIGNED NOT NULL,
  "epochs" INTEGER UNSIGNED NOT NULL,
  "layers" TEXT NOT NULL,
  "rmse_global" REAL NOT NULL,
  "metadata" JSON NOT NULL
);"""
    cursor.execute(sql)
    conn.commit()


def assert_exists_keys(d, key_list):
    for k in key_list:
        assert (k in d), "Required key '{}' not found.".format(k)


def insert_row(**kwargs):
    d = kwargs
    # Validate keys are present in the incoming argument dictionary.
    required_keys = [
       "author",
       "algorithm",
       "activation",
       "dataset",
       "forecast_length",
       "prediction_gap",
       "training_window_length",
       "sample_window_length",
       "epochs",
       "layers",
       "rmse_global",
    ]
    assert_exists_keys(d, required_keys)

    # Fill in other keys, if not already present.
    d["creation_ts"] = d.get("creation_ts", datetime.utcnow().isoformat())
    d["id"] = d.get("id", uuid.uuid4().hex)
    d["metadata"] = d.get("metadata", json.dumps({}))

    # Insert into the database.
    sql = ("INSERT INTO AlgorithmResult values ("
        ":id, :author, :algorithm, :activation, :dataset, :creation_ts,"
        ":forecast_length, :prediction_gap, :training_window_length,"
        ":sample_window_length, :epochs, :layers, :rmse_global, :metadata"
        ")")
    cursor.execute(sql, d)
    conn.commit()


# Cite: https://stackoverflow.com/q/5801170 (globals for DB connection)
def db_init(location):
    global conn
    global cursor
    conn = sqlite3.connect(location)
    cursor = conn.cursor()
    create_schema()


def main():
    args = parse_args()

    db_init(args.database)

    if args.command == 'insert':
        params = json.loads(sys.stdin.read())
        insert_row(**params)
    elif args.command == 'manual_insert':
        params = args.__dict__
        insert_row(**params)
    elif args.command == 'query':
        cursor.execute(args.sql)
        rows = cursor.fetchall()
        writer = csv.writer(sys.stdout)
        header = ["id","author","algorithm","activation","dataset","creation_ts","forecast_length","prediction_gap","training_window_length","sample_window_length","epochs","layers","rmse_global","metadata"]
        writer.writerow(header)
        writer.writerows(rows)

    conn.close()


if __name__ == '__main__':
    main()
