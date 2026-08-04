#!/bin/bash
#SBATCH  --time 12:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 30
#SBATCH -o unforcedslumlogvel_v1.6e+07_c2.txt
module load voro/0.4.6
module load eigen/3.3.2
module load latte/1.1.1
mpiexec -n 30 /u/kshao/software/lammps-install/bin/lmp -in unforced_base.in
