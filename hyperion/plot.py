import sys
import csv
import argparse
import sqlite3
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Grid plot tool for RTML project')

    parser.add_argument("database", default="", help="Database file to query from.")
    parser.add_argument("--where", action="append", default=[], help="WHERE constraints to add to the SQL query. Can be used multiple times to add more constraints. (Ex: '--where \"epochs >= 50\" ')")

    return parser.parse_args()


def get_data_sqlite(location, sql_query):
    out = defaultdict(list)
    i = 0
    conn = sqlite3.connect(location)
    cursor = conn.cursor()

    # Run the user-provided query, and fetch all rows.
    cursor.execute(sql_query)
    rows = cursor.fetchall()

    # Build data dictionary.
    # It's okay for this to be O(n^2), because performance doesn't matter here.
    field_names = ["id","author","algorithm","activation","dataset","creation_ts","forecast_length","prediction_gap","training_window_length","sample_window_length","epochs","layers","rmse_global","metadata"]
    for row in rows:
        kv_pairs = zip(field_names, row)
        for (k, v) in kv_pairs:
            out[k].append(v)
        i += 1

    print("Fetched {} records from database '{}'".format(i, location))
    out["num_records"] = i
    return dict(out)


def get_data_csv(filename):
    out = defaultdict(list)
    i = 0
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["nodes"].append(int(row["nodes"]))
            out["training_window_length"].append(int(row["training_window_length"]))
            out["sample_window_length"].append(int(row["sample_window_length"]))
            out["epochs"].append(int(row["epochs"]))
            out["rmse_global"].append(float(row["rmse_global"]))
            i += 1
    out["num_records"] = i
    return dict(out)


def gen_plots(d):
    #num_plots = 3
    num_plots = 3
    #x_data = np.arange(len(x))

    # Plot subplot for each signal.
    fig = plt.figure()

    # Set axis labels.
    for (y, text) in zip([0.75, 0.5, 0.25], ["Epochs", "Training Window\nlength", "Sample Window\nlength"]):
        fig.text(0.06, y, text, ha='center', va='center', rotation=90)
    for (x, text) in zip([0.25, 0.5, 0.75], ["Epochs", "Training Window length", "Sample Window length"]):
        fig.text(x, 0.9, text, ha='center', va='center')

    # --------------------------------------------------------------
    # First row
    axs = fig.add_subplot(3, 3, 2, projection='3d')
    axs.scatter(d['training_window_length'], d['epochs'], d['rmse_global'])
    axs.set_xlabel('Training Window length')
    axs.set_ylabel('Epochs')
    axs.set_zlabel('RMSE')
    axs.view_init(15, 74)

    axs = fig.add_subplot(3, 3, 3, projection='3d')
    axs.scatter(d['sample_window_length'], d['epochs'], d['rmse_global'])
    axs.set_xlabel('Sample Window length')
    axs.set_ylabel('Epochs')
    axs.set_zlabel('RMSE')
    axs.view_init(15, 74)

    # --------------------------------------------------------------
    # Second row
    axs = fig.add_subplot(3, 3, 4, projection='3d')
    axs.scatter(d['epochs'], d['training_window_length'], d['rmse_global'])
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Training Window length')
    axs.set_zlabel('RMSE')
    axs.view_init(32, 26)

    axs = fig.add_subplot(3, 3, 6, projection='3d')
    axs.scatter(d['sample_window_length'], d['training_window_length'], d['rmse_global'])
    axs.set_xlabel('Sample Window length')
    axs.set_ylabel('Training Window length')
    axs.set_zlabel('RMSE')
    axs.view_init(50, 37)

    # --------------------------------------------------------------
    # Third row
    axs = fig.add_subplot(3, 3, 7, projection='3d')
    axs.scatter(d['epochs'], d['sample_window_length'], d['rmse_global'])
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Sample Window length')
    axs.set_zlabel('RMSE')
    axs.view_init(27, 26)

    axs = fig.add_subplot(3, 3, 8, projection='3d')
    axs.scatter(d['training_window_length'], d['sample_window_length'], d['rmse_global'])
    axs.set_xlabel('Training Window length')
    axs.set_ylabel('Sample Window length')
    axs.set_zlabel('RMSE')
    axs.view_init(31, -51)

    # Put a title on the plot and the window, then render.
    fig.suptitle('MLP/ANN Batch Run Results ({} records)'.format(d['num_records']), fontsize=15)
    fig.canvas.set_window_title('MLP Batch Run Results ({} records)'.format(d['num_records']))
    plt.show()


if __name__ == "__main__":
    # Note: We pair this "SELECT" query with a "GROUP BY" later to allow averaging RMSEs.
    sql = "SELECT id,author,algorithm,activation,dataset,creation_ts,forecast_length,prediction_gap,training_window_length,sample_window_length,epochs,layers,AVG(rmse_global),metadata FROM AlgorithmResult"
    args = parse_args()

    # User can inject arbitrary SQL here, but we never commit the results, so
    # it should be safe to allow here.
    if len(args.where) > 0:
        sql += " WHERE "
        sql += " AND ".join(args.where)
        # The "GROUP BY" clause ensures we average together identical configurations.
        sql += " GROUP BY author,algorithm,activation,dataset,forecast_length,prediction_gap,training_window_length,sample_window_length,epochs,layers"

    data_dict = get_data_sqlite(args.database, sql)
    gen_plots(data_dict)
