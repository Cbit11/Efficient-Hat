#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=96:0:0    
#SBATCH --mail-user=rathodchaitanya11@gmail.com
#SBATCH --mail-type=ALL

cd $sr_project
module purge
module load python/3.9.21 scipy-stack
module load opencv
source ~/projects/def-mhassanz/cjrathod/sr_env/bin/activate
python try.py