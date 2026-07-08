import numpy as np 
from pydos2unix import dos2unix
import scipy as scp
#use 30 cpus to maximize usage
#for f in *.sh; do sbatch "$f"; done
velarr=np.logspace(3,8,60)


for i in range(len(velarr)):
    file_path = 'process.py'  # Path to your file
    newfile = "process{}.py".format(i)
    old_vel = "i=0"
    new_vel = "i={}".format(i)
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Replace the text
    updated_content = content.replace(old_vel, new_vel)


    # Write the modified content back to the file (overwrite)
    with open(newfile, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    with open(newfile, "rb") as src:
        buffer = dos2unix(src)
    with open(newfile, "wb") as dest:
        dest.write(buffer)

    file_path = 'process.slurm'  # Path to your file
    newfile = "process{}.sh".format(i)
    oldlog = "/home/kshao4/.conda/envs/pyt_clean/bin/python process.py"
    newlog = "/home/kshao4/.conda/envs/pyt_clean/bin/python process{}.py".format(i)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Replace the text
    updated_content = content.replace(oldlog, newlog)
    # Write the modified content back to the file (overwrite)
    with open(newfile, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    with open(newfile, "rb") as src:
        buffer = dos2unix(src)
    with open(newfile, "wb") as dest:
        dest.write(buffer)

# content = "#!/bin/bash"
# for i in range(len(velarr)):
#     vel = velarr[i]
#     content +="\nsbatch \"unforcedvel_{:.1e}.sh\"".format(vel)

# with open("unforcedvel.bash",'w', encoding='utf-8') as file:
#     file.write(content)