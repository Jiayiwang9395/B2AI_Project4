#!/bin/bash
#SBATCH -o ./std_out_test
#SBATCH -e ./std_err_test
#SBATCH --gpus=1
#SBATCH -p general
source activate audioenv
python ./test_Oversampling.py
conda deactivate