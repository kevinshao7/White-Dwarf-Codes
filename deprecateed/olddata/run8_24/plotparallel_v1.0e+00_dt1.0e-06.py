import numpy as np 
#for f in *.pysh; do sbatch "$f"; done
atoms = 2000
vel= 1.0e+00
dtfrac = 1.0e-06
thermsteps=80
angle = 0.3
sina = np.sin(angle)
cosa = np.cos(angle)
# folder="unforced/run8_6"
trajfile = "trajvel_v{:.1e}_dt{:.1e}.txt".format(vel,dtfrac)
logfile = "unforcedvel_v{:.1e}_dt{:.1e}.log".format(vel,dtfrac)
def trajtoarr(filename,atoms=1):
    timeflag =False
    atomflag=False
    tcount = 0
    atomlist = []
    tempatomlist=np.zeros(atoms)
    with open(filename, 'r') as file:
        for line in file:
            if "ITEM" in line:
                if "ITEM: TIMESTEP" in line:
                    tcount += 1
                    timeflag = True
                    atomflag=False
                    array=np.array(tempatomlist)
                    atomlist.append([np.mean(array),np.std(array)/(atoms**0.5)])


                    acount=0
                    tempatomlist=np.zeros(atoms)


                elif "ITEM: ATOMS" in line:
                    atomflag = True
                    timeflag=False
                else:
                    timeflag = False
                    atomflag = False
            # if timeflag and ("ITEM" not in line):
            #     timelist.append(float(line.strip()))
            if atomflag and ("ITEM" not in line):
                vz = float(line.strip().split(" ")[-1])
                vy = float(line.strip().split(" ")[-2])
                tempatomlist[acount] = cosa*vz+sina*vy
                acount += 1
    atomlist.append([np.mean(tempatomlist),np.std(tempatomlist)/(atoms**0.5)])

    atomlist.pop(0)
    atomarr = np.array(atomlist)
    return atomarr
def logtoarr(filename):
    startlogstr = " Step          Time           Temp         c_tempH        c_tempSi        TotEng         PotEng         KinEng         Press "
    endlogstr = "Loop time of"
    templist=[]
    dataflag=False
    with open(filename, 'r') as file:
        for line in file:
            if "Time step     :" in line:
                dataline = list(filter(None, line.split(" ")))
                # dataline[-1] = dataline[-1][:-1]
                # dataline = list(filter(None, dataline))

            if dataflag:
                dataline = list(filter(None, line.split(" ")))
                dataline.pop()
                dataline = list(filter(None, dataline))
                if len(dataline)>5:
                    templist.append([dataline[1],dataline[2],dataline[3],dataline[4],dataline[5]])
            if startlogstr in line:
                dataflag=True
                templist=[]
            if endlogstr in line:
                dataflag=False
    templist.pop()
    templist.pop()
    # print(templist)
    return np.array(templist,dtype=np.float64)



datai = logtoarr(logfile)
tempfarr= trajtoarr(trajfile,atoms=atoms)

np.savetxt("datai_v{:.1e}_dt{:.1e}.np".format(vel,dtfrac),datai)
np.savetxt("tempfarr_v{:.1e}_dt{:.1e}.np".format(vel,dtfrac),tempfarr[thermsteps:,:])
