#!/bin/bash 
#PBS -S /bin/bash 
#PBS -l select=2:ncpus=40:mpiprocs=40:ompthreads=1 
#PBS -q s1 
#PBS -N radlev
#PBS -M amalone7@u.rochester.edu 
#PBS -j oe 

cd $PBS_O_WORKDIR 

module load lammps/2Aug2023/b2
mpirun -np 80 lmp_mpi -log log010.lammps -in radlev010.in