#!/bin/bash

# Script to generate reasonable CSV files for plotting later.

# Sed-style regex to allow yanking out the different benchmark parameters.
benchmark_regex='bench-mlp-\([[:digit:]]\+\)n-\([[:digit:]]\+\)w-\([[:digit:]]\+\)h-\([[:digit:]]\+\)e.txt'

touch collation.csv
echo "nodes,window_length,history_length,epochs,rmse" > collation.csv

for item in bench-mlp-*.txt; do
    printf "%s :: " "$item"
    # Create variables for each part of the filename.
    num_nodes=$(echo "$item" | sed "s/$benchmark_regex/\1/g");
    window_length=$(echo "$item" | sed "s/$benchmark_regex/\2/g");
    history_length=$(echo "$item" | sed "s/$benchmark_regex/\3/g");
    num_epochs=$(echo "$item" | sed "s/$benchmark_regex/\4/g");

    # DEBUG
    #printf "  Number of nodes is: $num_nodes\n"
    #printf "  Window length is: $window_length\n"
    #printf "  History length is: $history_length\n"
    #printf "  Number of epochs is: $num_epochs\n"

    avg_rmse=$(cat "$item" | sed 's/Global RMSE: //g' | awk 'BEGIN {sum = 0.0} {sum += $1} END {if (sum == 0.0) print ""; else print sum/NR}');

    if [ -n "${avg_rmse}" ]; then
        printf "${num_nodes},${window_length},${history_length},${num_epochs},${avg_rmse}\n";
        printf "${num_nodes},${window_length},${history_length},${num_epochs},${avg_rmse}\n" >> collation.csv;
    else
        printf "Skipping; no data present.\n";
    fi
done
