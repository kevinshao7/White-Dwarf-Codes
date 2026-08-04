#!/bin/bash -l

#SBATCH --job-name=unforced_base
#SBATCH --partition=gpu1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000

#SBATCH --chdir=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais
#SBATCH --output=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais/unforced_base_%j.out
#SBATCH --error=/dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais/unforced_base_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

module purge
module load gcc/15

LMP=/u/kshao/software/lammps-gpu-install/bin/lmp

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "MPI tasks: $SLURM_NTASKS"
echo "GPU visibility: $CUDA_VISIBLE_DEVICES"

nvidia-smi

srun "$LMP" \
    -sf gpu \
    -pk gpu 1 \
    -log "unforced_base_${SLURM_JOB_ID}.lammps.log" \
    -in unforced_base.in

status=$?
echo "LAMMPS exit status: $status"
echo "Job finished: $(date)"
exit "$status"
