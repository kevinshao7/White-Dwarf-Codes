#!/bin/bash
#SBATCH --partition=standard --time=5:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 1
#SBATCH --exclude=bhc0020,bhc0021
#SBATCH -o drag_1000000.0_log.txt
module load python
python drag_1.0e+06.py
