#!/bin/bash
#SBATCH --partition=interactive --time=11:59:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 24
#SBATCH -o forced_0.1_10000.txt
module load openmpi/2.1.1/b1
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
module load lammps/23Jun2022/b1
mpiexec -n 24 lmp_mpi -in forced_0.1_10000.in