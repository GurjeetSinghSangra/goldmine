#!/bin/bash

#SBATCH --job-name=sim-train
#SBATCH --output=simulate_lv_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=61GB
#SBATCH --time=2-00:00:00

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine/

./simulate.py --nthetas 1000 --nsamples 1 lotkavolterra trainfocus${SLURM_ARRAY_TASK_ID}