#!/bin/bash --login
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=16 # number of cores
#SBATCH --mem=64G # memory pool for all cores
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u25am19@abdn.ac.uk
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=gpu02

module load anaconda3
source activate pt

nvidia-smi

rm logfiles.txt
srun python src/pretrain.py -c=config_maxwell.conf --projection=both
srun python src/finetune.py -c=config_maxwell.conf --projection=both

rm logfiles.txt
srun python src/pretrain.py -c=config_maxwell.conf --projection=none
srun python src/finetune.py -c=config_maxwell.conf --projection=none

rm logfiles.txt
srun python src/pretrain.py -c=config_maxwell.conf --projection=both --combloss=1
srun python src/finetune.py -c=config_maxwell.conf --projection=both --combloss=1

rm logfiles.txt
srun python src/pretrain.py -c=config_maxwell.conf --projection=none --combloss=1
srun python src/finetune.py -c=config_maxwell.conf --projection=none --combloss=1

