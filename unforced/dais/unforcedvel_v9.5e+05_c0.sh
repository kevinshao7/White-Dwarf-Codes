#!/bin/bash -l

#SBATCH --job-name=lammps_h200
#SBATCH --chdir=/dais/fs/scratch/kshao/wd

#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=04:00:00

#SBATCH --output=unforced_baselog.%j.txt
#SBATCH --error=unforced_baseerr.%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

module purge
module load gcc/15

MPI_ROOT=/mpcdf/soft/SLE_15/packages/skylake/openmpi/gcc_15-15.1.0/5.0.10
CUDA_ROOT=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1

export LD_LIBRARY_PATH="$MPI_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_smsc="^knem"
export OMP_NUM_THREADS=1

LMP=/u/kshao/software/lammps-gpu-install/bin/lmp

echo "Running on: $(hostname)"
echo "MPI tasks: $SLURM_NTASKS"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

nvidia-smi

srun "$LMP" \
    -sf gpu \
    -pk gpu 1 \
    -in unforced_base.in
