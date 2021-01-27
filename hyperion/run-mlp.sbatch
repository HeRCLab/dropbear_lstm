#!/bin/bash
#SBATCH --mail-user=conradp@email.sc.edu
#SBATCH --mail-type=ALL

# This Slurm job submission script expects a complex command line, in exchange
# for greatly increased flexibility.

# Parameters that must be provided for full functionality:
# --job-name=dataset-10n-activ
# --output="%j_mlp20nrelu.out"
# --error="%j_mlp20nrelu.err"
# --ntasks=252
# --export=n=5,e={25..400..25},t={20..100..10},s={10..100..10},x={1..10},dataset="name.json"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load python3/anaconda/2019.10

mkdir -p out

# Ensure we have defaults set:
# Cite : https://stackoverflow.com/a/2013589
dataset=${dataset:-""}
n=${n:-1}
a=${a:-linear}
e=${e:-1}
t=${t:-1}
s=${s:-1}
x=${x:-'{1..10}'}

# Cite: https://stackoverflow.com/a/29021862
# Expand any ranges here. Since only trusted users will be running this,
# the use of eval here *should* be benign.
datasets=($(eval echo "$dataset"))
nodes=($(eval echo "$n"))
activations=($(eval echo "$a"))
epochs=($(eval echo "$e"))
training_window=($(eval echo "$t"))
sampling_window=($(eval echo "$s"))
xs=($(eval echo "$x"))

# Datasets.
for ((di=0; di<${#datasets[@]}; di++)); do
    dataset=${datasets[$di]};
    # Layers.
    for ((ni=0; ni<${#nodes[@]}; ni++)); do
        n=${nodes[$ni]};
        # Activation functions.
        for ((ai=0; ai<${#activations[@]}; ai++)); do
            a=${activations[$ai]};
            # Epochs.
            for ((ei=0; ei<${#epochs[@]}; ei++)); do
                e=${epochs[$ei]};
                # Training window lengths.
                for ((ti=0; ti<${#training_window[@]}; ti++)); do
                    t=${training_window[$ti]};
                    # Sampling window lengths.
                    for ((si=0; si<${#sampling_window[@]}; si++)); do
                        s=${sampling_window[$si]};
                        # Skip over invalid combinations.
                        if [ "$s" -ge "$t" ]; then
                            continue
                        fi
                        touch results.db
                        # Collect significant number of samples for each parameter set.
                        for ((xi=0; xi<${#xs[@]}; xi++)); do
                            x=${xs[$xi]};
                            # If we've collected enough samples, skip to the next benchmark.
                            if (( `python3 dbtool.py query results.db "SELECT * FROM AlgorithmResult WHERE dataset=\"$dataset\" AND layers=\"$n\" AND epochs=$e AND training_window_length=$t AND sample_window_length=$s AND activation=\"$a\"" | wc -l` >= 11 )); then
                                break
                            fi
                            SLURM_NTASKS
                            # Bless Ish for figuring this bit out.
                            # The scheduler isn't as smart as I gave it credit for, so all
                            # the jobs get submitted at once, unless you wait on the batch.
                            (( i=i%$SLURM_NTASKS )); (( i++==0 )) && wait
                            #echo $dataset $n $a $e $t $s $x
                            srun -n1 -c1 --exclusive ./job.sh -u ${n} -a ${a} -e ${e} -s ${s} -t ${t} -i ${dataset} | python3 dbtool.py insert results.db &
                        done
                    done
                done
            done
        done
    done
done
wait
