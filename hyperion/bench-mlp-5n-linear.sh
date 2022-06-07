#!/bin/bash
#SBATCH --job-name=mlp-5n-linear
#SBATCH -o %j_mlp5nlinear.out
#SBATCH -e %j_mlp5nlinear.err
#SBATCH --ntasks=28
#SBATCH --mail-user=conradp@email.sc.edu
#SBATCH --mail-type=END

module load python3/anaconda/2019.10

mkdir -p out

# Epochs.
for e in {25..400..25}; do
    # Training window sizes.
    for w in {20..100..10}; do
        # History lengths.
        for h in {10..100..10}; do
            # Skip over invalid combinations.
            if [ "$h" -ge "$w" ]; then
                continue
            fi
            touch results.db
            # Collect significant number of samples for each parameter set.
            for x in {1..10}; do
                # Bless Ish for figuring this bit out.
                # The scheduler isn't as smart as I gave it credit for, so all
                # the jobs get submitted at once, unless you wait on the batch.
                (( i=i%28 )); (( i++==0 )) && wait
                # If we've collected enough samples, skip to the next benchmark.
                if (( `python3 dbtool.py query results.db "SELECT * FROM AlgorithmResult WHERE dataset=\"5k-composite.json\" AND layers=\"5\" AND epochs=$e AND training_window_length=$w AND sample_window_length=$h AND activation=\"linear\"" | wc -l` >= 11 )); then
                    break
                fi
                srun -n1 -c1 --exclusive ./job.sh -u 5 -s ${h} -t ${w} -e ${e} -f 5k-composite.json | python3 dbtool.py insert results.db &
            done
        done
    done
done
wait
