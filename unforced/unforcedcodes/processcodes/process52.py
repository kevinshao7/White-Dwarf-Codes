import numpy as np
import matplotlib.pyplot as plt
import math as math
atoms = 2000
Temp = 15000
kb = 1.38e-23
NA = 6.022e23
MSI = 28.0855/NA #use cgs system
angle = 0.3 #radians
sinangle = math.sin(angle)
cosangle = math.cos(angle)


def trajtoarr(filename, dt,atoms=2000):
    timeflag =False
    atomflag=False
    count = 0
    atomlist = []
    tempatomlist=[]
    timestep=0
    with open(filename, 'r') as file:
        for line in file:
            if "ITEM" in line:
                if "ITEM: TIMESTEP" in line:
                    count += 1
                    timeflag = True
                    atomflag=False
                    tempatomlist.append(float(dt*timestep))
                    atomlist.append(tempatomlist)
                    tempatomlist=[]
                elif "ITEM: ATOMS id type vx vy vz" in line:
                    atomflag = True
                    timeflag=False
                else:
                    timeflag = False
                    atomflag = False
            if timeflag and ("ITEM" not in line):
                timestep = float(line.strip())
            if atomflag and ("ITEM" not in line):
                vz = float(line.strip().split(" ")[-1])
                vy = float(line.strip().split(" ")[-2])
                tempatomlist.append(sinangle*vy+cosangle*vz)
                count += 1
    tempatomlist.append(float(dt*timestep))
    atomlist.append(tempatomlist)
    atomlist.pop()
    atomlist.pop(0)
    atomlist.pop()
    atomarr = np.array(atomlist)
    print(atomarr)
    return atomarr
def logtoarr(filename):
    startlogstr = "   Step          Time           Temp         c_tempH        c_tempSi        TotEng         PotEng         KinEng         Press           v_k0           v_dt        v_collfreq       v_ccs "
    endlogstr = "Loop time of "
    templist=[]
    dataflag=False
    with open(filename, 'r') as file:
        for line in file:
            if  "ERROR" in list(filter(None, line.split(" "))):
                break
            if dataflag:
                dataline = list(filter(None, line.split(" ")))
                dataline[-1] = dataline[-1][:-1]
                dataline = list(filter(None, dataline))
                templist.append(dataline)
            if startlogstr in line:
                dataflag=True
                templist=[]
            if endlogstr in line:
                dataflag=False
    print(templist)

    templist.pop()
    templist.pop()
    templist.pop()
    templist.pop()
    templist.pop()
    templist.pop(0)    # print(templist
    return np.array(templist,dtype=np.float64)


# runlist = ["run10_15","run10_15b","run10_17","run10_18","run10_19","run10_22","run10_28","run11_5","run4_15","run4_22","run4_29"]
# drop22 = [[0,0],[1,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0]]
# drop28=[]
# for i in range(10):
#     for j in range(1,4):
#         drop22.append([i,j])
# for i in range(2):
#     for j in range(1,4):
#         drop28.append([i,j])
# drop115=[]
# for i in range(2):
#     for j in range(1,4):
#         drop115.append([i,j])
# drop45=[[0,1],[0,3],[2,1],[3,1]]
# drop422=[[0,3],[1,2],[1,3]]
# dropdict = {"run10_15":[[0,0],[0,1],[0,2],[1,2],[3,2],[4,2],[5,0],[5,1],[5,2],[5,3]],"run10_15b":[[0,0],[1,1],[1,2],[1,3],[2,0],[2,2],[2,3],[3,2],[3,3],[4,1],[4,2],[5,1],[5,2]],"run10_17":[[0,0],[1,3],[3,0],[3,1],[4,1],[4,2],[4,3],[5,1],[5,3],[6,2],[6,3]],"run10_18":[[0,2],[0,3],[1,0],[1,1],[3,2],[4,2],[5,1],[5,3]],"run10_19":[[0,0],[1,0],[1,1],[5,1]],"run10_22":drop22,"run10_28":drop28,"run11_5":drop115,"run4_15":drop45,"run4_22":drop422,"run4_29":[]}#[[0,2],[1,0]]
# vel429float=np.logspace(3,8,60)
# # vel429=[]
# # for i in range(60):
# vel429 = vel429float
# thermdict = {"run10_15":81,"run10_15b":81,"run10_17":81,"run10_18":81,"run10_19":21,"run10_22":21,"run10_28":21,"run11_5":21,"run4_15":21,"run4_22":21,"run4_29":21}
# veldict = {"run10_15":[2e7,3e7,4e7,5e7,6e7,8e7],"run10_15b":[1e2,1e3,1e4,1e5,1e6,1e7,1e8],"run10_17":[1.5e7,2e6,2.5e7,3e6,3.5e7,4.5e7,6e6,8e6],"run10_18":[1.5e6,2.5e6,4e6,5e6,7e6,9e6],"run10_19":[1.1e7,1.2e7,1.3e7,1.5e7,1.7e7,8e6],"run10_22":[1e4,1e5,1e7,1e8,2e5,2e6,3e5,3e6,6e5,6e6],"run10_28":[1e6,1e7],"run11_5":[3e6,3e7],"run4_15":[1e6,3e6,1e7,3e7],"run4_22":[1e6,1e7],"run4_29":vel429}
# thermdict = {}

velarr=np.logspace(3,8,60)
i=52
thermsteps =21#thermdict[run]
# velarr = veldict[run]
casearr = np.array([0,1,2,3])
folder = "../unforcedcodes"
for j in range(len(casearr)):
        print(i,j)
        try:
            vel = velarr[i]
            case = casearr[j]
            datai = logtoarr(folder+"/unforcedvel_v{:.1e}_c{}.log".format(vel,case))
            dt = (datai[1,1]-datai[0,1])/100
            trajarr = trajtoarr(folder+"/trajvel_v{:.1e}_c{}.txt".format(vel,case),dt,atoms=atoms)
            np.savetxt("log_v{:.1e}_c{}.np".format(vel,case),datai)
            np.savetxt("force_v{:.1e}_c{}.np".format(vel,case),trajarr)
        except:
            print("fail")



