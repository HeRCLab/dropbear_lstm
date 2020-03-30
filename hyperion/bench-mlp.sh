#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err
#SBATCH --ntasks=28

mkdir -p out

# Epochs.
for e in {25..400..25}; do
    # Training window sizes.
    for w in 20 30 40 50 60 70 80 90 100; do
        # History lengths.
        for h in 10 20 30 40; do
            # Skip over invalid combinations.
            if [ "$h" -ge "$w" ]; then
                continue
            fi
            touch out/bench-mlp-10n-${w}w-${h}h-${e}e.txt
            # Collect significant number of samples for each parameter set.
            for x in {1..10}; do
                # If we've collected enough samples, skip to the next benchmark.
                if (( `cat out/bench-mlp-10n-${w}w-${h}h-${e}e.txt | wc -l` >= 10 )); then
                    break
                fi
                srun -n1 -c1 --exclusive ./job.sh -u 10 --history-length ${h} -w ${w} -e ${e} -f 5k-composite.json >> out/bench-mlp-10n-${w}w-${h}h-${e}e.txt &
            done
        done
    done
done
wait
