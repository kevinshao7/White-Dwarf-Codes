#!/bin/bash
#SBATCH --partition=standard --time 11:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 40
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH -o unforcedslumlogvel_v3.0e+06_c1.txt
module load openmpi/2.1.1/b1
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
module load lammps/23Jun2022/b1
mpiexec -n 40 lmp_mpi -in unforcedvel_v3.0e+06_c1.in
