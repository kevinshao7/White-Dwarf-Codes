#!/bin/bash -l

#SBATCH --job-name=unforcedprod_v4.6e+07_c0
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=12:00:00

#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/daisslurm/daisproduction
#SBATCH --output=unforcedprod_v4.6e+07_c0_%j.out
#SBATCH --error=unforcedprod_v4.6e+07_c0_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

set -euo pipefail

# Remove any inherited GCC 15 / CUDA 13 / Open MPI paths.
module purge
unset MPI_ROOT
unset CUDA_ROOT
unset LD_LIBRARY_PATH

# Toolchain used to build this LAMMPS executable.
module load gcc/14
module load cuda/12.8
module load openmpi/5.0

# One MPI process controls the H200.
export OMP_NUM_THREADS=1

# Disable unavailable KNEM transport and use Open MPI's fallback.
export OMPI_MCA_smsc="^knem"

LMP=/u/kshao/software/lammps-gpu-cuda12-install/bin/lmp
INPUT=unforcedprod_v4.6e+07_c0.in
LMP_LOG="unforcedprod_v4.6e+07_c0_${SLURM_JOB_ID}.lammps.log"

echo "Job ID:        $SLURM_JOB_ID"
echo "Host:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "MPI tasks:     $SLURM_NTASKS"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "Visible GPU:   ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "LAMMPS:        $LMP"
echo "Input:         $INPUT"

module list 2>&1

nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

if [[ ! -x "$LMP" ]]; then
    echo "ERROR: LAMMPS executable is missing or not executable: $LMP" >&2
    ls -l "$LMP" >&2 || true
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: LAMMPS input file is missing from $(pwd): $INPUT" >&2
    ls -l >&2
    exit 1
fi

echo "Preflight checks passed; launching LAMMPS."

srun \
    --ntasks="$SLURM_NTASKS" \
    --kill-on-bad-exit=1 \
    "$LMP" \
    -sf gpu \
    -pk gpu 1 neigh no \
    -log "$LMP_LOG" \
    -in "$INPUT"

echo "LAMMPS completed successfully."
echo "End time: $(date)"
