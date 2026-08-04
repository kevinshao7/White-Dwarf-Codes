import numpy as np 
from pydos2unix import dos2unix
import scipy as scp
#use 30 cpus to maximize usage
#for f in *.sh; do sbatch "$f"; done
#cd   /dais/fs/scratch/kshao/wd/White-Dwarf-Codes/unforced/dais/

velarr=np.logspace(5,7.2,10)#relative velocity cm/s
gccarr =  np.array([1e-5,1,1e-5,1])
Tarr = np.array([5000,5000,1e5,1e5])
Z1arr = np.array([0.16,0.65,0.93,0.68])
Z2arr = np.array([0.26,3.82,4.27,3.81])
rhoHarr = np.array([1e-5,1,1e-5,1])
rhoSarr = np.array([1.79e-4,7.45,6.2e-5,7.2])
NA = 6.022e23
pi = 3.1415926535
me = 9.11e-31 #electron mass
ms = 28.0855*0.001/NA
mh = 1.008*0.001/NA
mu = ms*mh/(ms+mh)
qe = 1.6e-19
hbar = 6.626e-34
kb = 1.38e-23
e0 =  8.854e-12
def ccs2(vrel,vcom,T,gcc,z1,z2):# use kinetic energy difference, returns collision cross section in square meters
    ne =1e6* NA*gcc/1.008
    ve = np.sqrt(3*kb*T/me)
    wp = np.sqrt(ne*(qe**2)/(me*e0))
    lD = ve/wp
    Tf = ((3*ne*(pi**2))**(2/3))*(hbar**2)/(2*me*kb)
    lS = lD*np.power(1+(2*Tf/(3*T)),1/4)
    k0 = 1/lS
    print("separation cm {}".format((mh*1000/gcc)**(1/3)))
    print("screening length cm {}".format(lS*100))
    Aarr = (qe**2)*z1*z2/(4*pi*e0)
    #kinetic energy available is velocity relative to COM
    Barr = 0.5*mh*((vrel-vcom)**2)+0.5*ms*((vcom)**2)
    Carr = Barr/Aarr
    rc = scp.special.lambertw(k0/Carr)/k0
    return np.real(pi*rc**2)

for i in range(len(velarr)):
    for j in [0,2]:
        vel = velarr[i]
        Z1 = Z1arr[j]
        Z2 = Z2arr[j]
        gcc = gccarr[j]
        T = Tarr[j]
        rhoH = rhoHarr[j]
        rhoS = rhoSarr[j]
        tsh = (3*kb*T/mh)**0.5
        tss = (3*kb*T/ms)**0.5
        vrel = 0.01*vel -tsh -tss #need relative velocity in meters per second
        vcom = mh*vrel/(mh+ms)
        ccs = 1e4*ccs2(vrel,vcom,T,gcc,z1=Z1,z2=Z2) #collision cross section in cm^2
        # Replace text in a file
        file_path = 'unforced_base.in'  # Path to your file
        newfile = "unforcedvel_v{:.1e}_c{}.in".format(vel,j)
        old_vel = "variable vb equal 1.0e7"
        new_vel = "variable vb equal {:.1e}".format(vel)
        old_ccs = "variable ccs equal 1e-6"
        new_ccs = "variable ccs equal {:.5e}".format(ccs)
        old_zh = "variable ZH equal 0.955"
        new_zh = "variable ZH equal {:.5e}".format(Z1)
        old_zs = "variable ZSi equal 4.91"
        new_zs = "variable ZSi equal {:.5e}".format(Z2)
        old_rh = "variable rhoH equal 1.0e-5 #gcc"
        new_rh = "variable rhoH equal {:.5e}".format(rhoH)
        old_rS = "variable rhoS equal 1.79e-4 #gcc"
        new_rS = "variable rhoS equal {:.5e} #gcc".format(rhoS)
        old_T = "variable T0 equal 100000"
        new_T = "variable T0 equal {:.5e}".format(T)
        old_dump = "dump mydmp Si custom 100 traj.txt id type vx vy vz"
        new_dump = "dump mydmp Si custom 100 trajvel_v{:.1e}_c{}.txt id type vx vy vz".format(vel,j)
        old_log = "log firstlog"
        new_log = "log unforcedvel_v{:.1e}_c{}.log".format(vel,j)
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Replace the text
        updated_content = content.replace(old_vel, new_vel)
        updated_content = updated_content.replace(old_dump, new_dump)
        updated_content = updated_content.replace(old_log, new_log)
        updated_content = updated_content.replace(old_ccs, new_ccs)
        updated_content = updated_content.replace(old_zh, new_zh)
        updated_content = updated_content.replace(old_zs, new_zs)
        updated_content = updated_content.replace(old_rh, new_rh)
        updated_content = updated_content.replace(old_rS, new_rS)
        updated_content = updated_content.replace(old_T, new_T)

        # Write the modified content back to the file (overwrite)
        with open(newfile, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        with open(newfile, "rb") as src:
            buffer = dos2unix(src)
        with open(newfile, "wb") as dest:
            dest.write(buffer)

        file_path = 'unforced_base.sh'  # Path to your file
        newfile = "unforcedvel_v{:.1e}_c{}.sh".format(vel,j)
        oldlog = "#SBATCH -o unforced_baselog.txt"
        newlog = "#SBATCH -o unforcedslumlogvel_v{:.1e}_c{}.txt".format(vel,j)
        oldcmd = "mpiexec -n 30 lmp_mpi -in unforced_base.in"
        newcmd = "mpiexec -n 30 lmp_mpi -in unforcedvel_v{:.1e}_c{}.in".format(vel,j)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Replace the text
        updated_content = content.replace(oldlog, newlog)
        updated_content = updated_content.replace(oldcmd, newcmd)
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