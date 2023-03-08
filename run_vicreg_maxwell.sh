#!/bin/bash --login
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=16 # number of cores
#SBATCH --mem=64G # memory pool for all cores
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=spot-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u25am19@abdn.ac.uk
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=egpu01

module load anaconda3
source activate pt

nvidia-smi

rm logfiles.txt

srun python src/pretrain.py -c=/uoa/home/u25am19/sharedscratch/VICReg/config_maxwell.conf --num_nodes=1 --devices=2
srun python src/finetune.py -c=/uoa/home/u25am19/sharedscratch/VICReg/config_maxwell.conf --num_nodes=1 --devices=2
