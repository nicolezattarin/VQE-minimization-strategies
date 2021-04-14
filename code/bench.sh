#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2-0:0

export CUDA_VISIBLE_DEVICES=""

./main.py --training singlelayer --nqubits 4 --method cg 

