import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

def get_data(filename):
    out = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["nodes"].append(int(row["nodes"]))
            out["window_length"].append(int(row["window_length"]))
            out["history_length"].append(int(row["history_length"]))
            out["epochs"].append(int(row["epochs"]))
            out["rmse"].append(float(row["rmse"]))
    return dict(out)


def gen_plots(d):
    #num_plots = 3
    num_plots = 3
    #x_data = np.arange(len(x))

    # Plot subplot for each signal.
    fig = plt.figure()

    # Set axis labels.
    for (y, text) in zip([0.75, 0.5, 0.25], ["Epochs", "Window\nlength", "History\nlength"]):
        fig.text(0.06, y, text, ha='center', va='center', rotation=90)
    for (x, text) in zip([0.25, 0.5, 0.75], ["Epochs", "Window length", "History length"]):
        fig.text(x, 0.9, text, ha='center', va='center')

    # --------------------------------------------------------------
    # First row
    axs = fig.add_subplot(3, 3, 2, projection='3d')
    axs.scatter(d['window_length'], d['epochs'], d['rmse'])
    axs.set_xlabel('Window length')
    axs.set_ylabel('Epochs')
    axs.set_zlabel('RMSE')

    axs = fig.add_subplot(3, 3, 3, projection='3d')
    axs.scatter(d['history_length'], d['epochs'], d['rmse'])
    axs.set_xlabel('History length')
    axs.set_ylabel('Epochs')
    axs.set_zlabel('RMSE')

    # --------------------------------------------------------------
    # Second row
    axs = fig.add_subplot(3, 3, 4, projection='3d')
    axs.scatter(d['epochs'], d['window_length'], d['rmse'])
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Window length')
    axs.set_zlabel('RMSE')

    axs = fig.add_subplot(3, 3, 6, projection='3d')
    axs.scatter(d['history_length'], d['window_length'], d['rmse'])
    axs.set_xlabel('History length')
    axs.set_ylabel('Window length')
    axs.set_zlabel('RMSE')

    # --------------------------------------------------------------
    # Third row
    axs = fig.add_subplot(3, 3, 7, projection='3d')
    axs.scatter(d['epochs'], d['history_length'], d['rmse'])
    axs.set_xlabel('Epochs')
    axs.set_ylabel('History length')
    axs.set_zlabel('RMSE')

    axs = fig.add_subplot(3, 3, 8, projection='3d')
    axs.scatter(d['window_length'], d['history_length'], d['rmse'])
    axs.set_xlabel('Window length')
    axs.set_ylabel('History length')
    axs.set_zlabel('RMSE')

    # Put a title on the plot and the window, then render.
    fig.suptitle('MLP/ANN Batch Run Results', fontsize=15)
    fig.canvas.set_window_title('MLP Batch Run Results')
    plt.show()


if __name__ == "__main__":
    data_dict = get_data('collation.csv')
    gen_plots(data_dict)
