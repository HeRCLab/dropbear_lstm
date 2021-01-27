#!/bin/bash

# Script to generate reasonable CSV files for plotting later.

# Sed-style regex to allow yanking out the different benchmark parameters.
benchmark_regex='bench-mlp-\([[:digit:]]\+\)n-\([[:digit:]]\+\)w-\([[:digit:]]\+\)h-\([[:digit:]]\+\)e.txt'

#touch collation.csv
#echo "nodes,window_length,history_length,epochs,rmse" > collation.csv

for item in bench-mlp-*.txt; do
    printf "%s :: " "$item"
    # Create variables for each part of the filename.
    author="Philip"
    algorithm="window-mlp"
    activation="linear"
    dataset=$(basename $(pwd) | sed 's/\.d//g'); # Grabs directory name.
    num_nodes=$(echo "$item" | sed "s/$benchmark_regex/\1/g");
    training_window_length=$(echo "$item" | sed "s/$benchmark_regex/\2/g");
    sample_window_length=$(echo "$item" | sed "s/$benchmark_regex/\3/g");
    epochs=$(echo "$item" | sed "s/$benchmark_regex/\4/g");
    layers=$num_nodes
    forecast_length="1"
    prediction_gap="0"

    # DEBUG
    #printf "  Number of nodes is: $num_nodes\n"
    #printf "  Window length is: $window_length\n"
    #printf "  History length is: $history_length\n"
    #printf "  Number of epochs is: $num_epochs\n"

    rmse_list=$(cat "$item" | sed 's/Global RMSE: //g')
    for rmse in $rmse_list; do
        rmse_global=$rmse;
        echo "python3 dbtool.py manual_insert results.db $author $algorithm $activation \"$dataset\" $forecast_length $prediction_gap $training_window_length $sample_window_length $epochs $layers $rmse_global"
        python3 dbtool.py manual_insert results.db $author $algorithm $activation "$dataset" $forecast_length $prediction_gap $training_window_length $sample_window_length $epochs $layers $rmse_global
    done

    

    #if [ -n "${avg_rmse}" ]; then
    #    printf "${num_nodes},${window_length},${history_length},${num_epochs},${avg_rmse}\n";
    #    printf "${num_nodes},${window_length},${history_length},${num_epochs},${avg_rmse}\n" >> collation.csv;
    #else
    #    printf "Skipping; no data present.\n";
    #fi
done
