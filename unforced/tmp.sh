#!/bin/bash
#SBATCH --partition=interactive --time=23:59:00 --output=tmp.log
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 24
#SBATCH -o log.txt
module load openmpi/2.1.1/b1
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
module load lammps/23Jun2022/b1
mpiexec -n 24 lmp_mpi -in in.first