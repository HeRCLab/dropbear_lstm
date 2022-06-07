#!/bin/bash

# Source the virtualenv for this script.
source ../venv/bin/activate


# For '5k-composite.json' dataset.
for l in {5..20..5}; do
    for a in 'linear' 'relu'; do
        python3 plot.py results.db \
          --where "layers = '${l}'" \
          --where "dataset = '5k-composite.json'" \
          --where "activation = '${a}'" \
          --scale-x 10 --scale-y 10 -o plots/5k-composite/mlp-${l}n-${a}_5k-composite.png
    done
done

# For Puja's datasets.
for l in {5..20..5}; do
    for a in 'linear' 'relu'; do
        for d in Ivol_Acc_Load_1S_10STD.lvm Ivol_Acc_Load_1S_1STD.lvm Ivol_Acc_Load_1S_5STD.lvm Ivol_Acc_Load_2S_10STD.lvm Ivol_Acc_Load_2S_1STD.lvm Ivol_Acc_Load_2S_5STD.lvm Ivol_Acc_Load_3S_10STD.lvm Ivol_Acc_Load_3S_1STD.lvm Ivol_Acc_Load_3S_5STD.lvm; do
            python3 plot.py results.db \
              --where "layers = '${l}'" \
              --where "dataset = '${d}.json'" \
              --where "activation = '${a}'" \
              --scale-x 10 --scale-y 10 -o plots/${d}/mlp-${l}n-${a}_${d}.png
        done
    done
done
