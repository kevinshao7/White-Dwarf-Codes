#!/bin/bash
#SBATCH --partition=standard --time 12:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 30
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH -o unforcedslumlogvel_v3.1e+05_c2.txt
module load openmpi/2.1.1/b1
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
module load lammps/23Jun2022/b1
mpiexec -n 30 lmp_mpi -in unforcedvel_v3.1e+05_c2.in
