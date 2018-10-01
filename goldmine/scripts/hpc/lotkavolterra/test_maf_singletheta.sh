#!/bin/bash

#SBATCH --job-name=test_maf
#SBATCH --output=log_test_maf_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate goldmine
cd /scratch/jb6504/goldmine/goldmine

./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 1000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 2000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 5000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 10000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 20000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 50000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 100000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus
./test.py lotkavolterra maf -i ${SLURM_ARRAY_TASK_ID} --singletheta --samplesize 200000 --densitygrid --ratiogrid --density --score --testsample test_focus --modellabel model_focus --classifiertest