#!/bin/bash
#SBATCH --partition=standard --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 1
##SBATCH --exclude=bhc0020,bhc0021 for rochester
#SBATCH -o drag_1.3e+04_log.txt
module purge
module load miniconda3/4.9.2

unset PYTHONPATH
unset PYTHONHOME


/home/kshao4/.conda/envs/pyt_clean/bin/python -c "import sys, numpy, scipy; print(sys.executable); print(numpy.__version__); print(scipy.__version__)"
/home/kshao4/.conda/envs/pyt_clean/bin/python drag_1.4e+03.py
