#!/bin/bash -l

#SBATCH --job-name=unforced_base
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=04:00:00

#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais
#SBATCH --output=unforced_base_%j.out
#SBATCH --error=unforced_base_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

module purge
module load gcc/15

export OMP_NUM_THREADS=1
export OMPI_MCA_smsc="^knem"

# Diagnostic: report CUDA errors at their actual location.
export CUDA_LAUNCH_BLOCKING=1

LMP=/u/kshao/software/lammps-gpu-install/bin/lmp

echo "Host: $(hostname)"
echo "Tasks: $SLURM_NTASKS"
echo "Visible GPU: $CUDA_VISIBLE_DEVICES"

nvidia-smi

srun --ntasks=1 \
    "$LMP" \
    -sf gpu \
    -pk gpu 1 \
    -log "unforced_base_${SLURM_JOB_ID}.lammps.log" \
    -in unforced_base.in

status=$?
echo "LAMMPS exit status: $status"
exit "$status"