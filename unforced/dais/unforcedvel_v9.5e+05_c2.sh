#!/bin/bash
#SBATCH  --time 4:00:00 
#SBATCH --gres=gpu:h200:1 # use 1 H200.
#SBATCH --cpus-per-task=4 # request 1/8 of available CPUs on a H200 node.
#SBATCH --mem=250000 # grant the job access to 1/8 of the memory on a H200 node.
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -o unforcedslumlogvel_v9.5e+05_c2.txt
module purge
module load gcc/15
mpiexec -n 30 /u/kshao/software/lammps-gpu-install/bin/lmp -in unforced_base.in
