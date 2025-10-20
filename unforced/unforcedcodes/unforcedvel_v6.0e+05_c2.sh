#!/bin/bash
#SBATCH --partition=standard --time=3:59:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 20
#SBATCH -o unforcedslumlogvel_v6.0e+05_c2.txt
module load openmpi/2.1.1/b1
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
module load lammps/23Jun2022/b1
mpiexec -n 20 lmp_mpi -in unforcedvel_v6.0e+05_c2.in
