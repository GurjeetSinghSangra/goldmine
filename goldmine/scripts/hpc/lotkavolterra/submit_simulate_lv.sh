#!/bin/bash

cd /scratch/jb6504/goldmine/goldmine/scripts/hpc/lotkavolterra

sbatch --array=0-1099 simulate_lv_train.sh
sbatch --array=0-99 simulate_lv_test_singletheta.sh
sbatch --array=0-99 simulate_lv_test.sh
