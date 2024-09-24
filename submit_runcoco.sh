#!/bin/bash
#SBATCH --job-name=fsod_coco
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=hkong5@sheffield.ac.uk
#SBATCH --output=./Output/output.txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load Java/17.0.4
module load Anaconda3/2022.05

source activate paper1

sh run_coco.sh exp_1