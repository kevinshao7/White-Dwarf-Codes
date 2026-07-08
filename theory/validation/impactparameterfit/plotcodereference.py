import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NA = 6.022e23
pi = 3.1415926535
me = 9.11e-31 #electron mass
ms = 28.0855e-3/NA #silicon mass kg
mh = 1.008e-3/NA
mu = ms*mh/(ms+mh)
qe = 1.6e-19
hbar = 6.626e-34
kb = 1.38e-23
e0 =  8.854e-12

casearr = np.array([0,1,2,3])

runlist = ["run10_15","run10_15b","run10_17","run10_18","run10_19","run10_22","run10_28","run11_5","run4_15","run4_22","run4_29"]
drop22 = [[0,0],[1,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0]]
drop28=[]
for i in range(10):
    for j in range(1,4):
        drop22.append([i,j])
for i in range(2):
    for j in range(1,4):
        drop28.append([i,j])
drop115=[]
for i in range(2):
    for j in range(1,4):
        drop115.append([i,j])
drop45=[[0,1],[0,3],[2,1],[3,1]]
drop422=[[0,3],[1,2],[1,3]]
dropdict = {"run10_15":[[0,0],[0,1],[0,2],[1,2],[3,2],[4,2],[5,0],[5,1],[5,2],[5,3]],"run10_15b":[[0,0],[1,1],[1,2],[1,3],[2,0],[2,2],[2,3],[3,2],[3,3],[4,1],[4,2],[5,1],[5,2]],"run10_17":[[0,0],[1,3],[3,0],[3,1],[4,1],[4,2],[4,3],[5,1],[5,3],[6,2],[6,3]],"run10_18":[[0,2],[0,3],[1,0],[1,1],[3,2],[4,2],[5,1],[5,3]],"run10_19":[[0,0],[1,0],[1,1],[5,1]],"run10_22":drop22,"run10_28":drop28,"run11_5":drop115,"run4_15":drop45,"run4_22":drop422,"run4_29":[]}#[[0,2],[1,0]]
vel429float=np.logspace(3,8,60)
# vel429=[]
# for i in range(60):
vel429 = vel429float
thermdict = {"run10_15":81,"run10_15b":81,"run10_17":81,"run10_18":81,"run10_19":21,"run10_22":21,"run10_28":21,"run11_5":21,"run4_15":21,"run4_22":21,"run4_29":21}
veldict = {"run10_15":[2e7,3e7,4e7,5e7,6e7,8e7],"run10_15b":[1e2,1e3,1e4,1e5,1e6,1e7,1e8],"run10_17":[1.5e7,2e6,2.5e7,3e6,3.5e7,4.5e7,6e6,8e6],"run10_18":[1.5e6,2.5e6,4e6,5e6,7e6,9e6],"run10_19":[1.1e7,1.2e7,1.3e7,1.5e7,1.7e7,8e6],"run10_22":[1e4,1e5,1e7,1e8,2e5,2e6,3e5,3e6,6e5,6e6],"run10_28":[1e6,1e7],"run11_5":[3e6,3e7],"run4_15":[1e6,3e6,1e7,3e7],"run4_22":[1e6,1e7],"run4_29":vel429}

Z1arr = np.array([0.16,0.65,0.93,0.68])
Z2arr = np.array([0.26,3.82,4.27,3.81])
gccarr =  np.array([1e-5,1,1e-5,1])
nearr = Z1arr*1e6*1e-3*gccarr/mh
Tarr = np.array([5000,5000,1e5,1e5])
mu = mh*ms/(mh+ms)
#coupling parameter
coupling = np.array([0.03,1.94,0.42,1.05])
print(nearr**(-1/3))

def fit_power_law(x, y):
    """Fit y = C x^n using only positive finite data."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xfit = x[mask]
    yfit = y[mask]
    if len(xfit) < 2:
        return None, None, mask
    coeffs = np.polyfit(np.log10(xfit), np.log10(yfit), 1)
    n = coeffs[0]
    C = 10**coeffs[1]
    return C, n, mask

plt.figure(figsize=(12,8))
colors = ["red","orange","green","blue"]
labelled = [False,False,False,False]

count = 0
in_dir = Path("../../theory/2026_05_06")
#debye length start
files = sorted(in_dir.glob("theory_max*.npy"))
dragarr = np.zeros((len(files),4,3))
velarr = np.zeros(len(files))
for count, p in enumerate(files):
    # print(p)
    velarr[count] = float(str(p).split("_")[4].split(".npy")[0])
    dragarr[count,:,:] = np.load(p)
indices = velarr.argsort()
dragarr = dragarr[indices,:,:]
velarr = velarr[indices]

in_dir = Path("../../theory/2026_05_13")
#separation start
files = sorted(in_dir.glob("*.npy"))
dragarrspace = np.zeros((len(files),4,3))
velarrspace = np.zeros(len(files))
for count, p in enumerate(files):
    print(p)
    velarrspace[count] = float(str(p).split("_")[5].split(".npy")[0])
    dragarrspace[count,:,:] = np.load(p)
indices = velarrspace.argsort()
dragarrspace = dragarrspace[indices,:,:]
velarrspace = velarrspace[indices]



# ---- line-plotted theory data + power-law fits ----
for i in range(4):
    if i == 0 or i==2:

        if i % 2 == 0:
            plt.vlines(x=100*np.sqrt(3*kb*Tarr[i]/mh), ymin=1e12, ymax=1e17, color=colors[i])
        else:
            plt.vlines(x=100*np.sqrt(3*kb*Tarr[i]/mh), ymin=1e17, ymax=1e22, color=colors[i])

        x_line = velarr * 100
        y_line = 100 * dragarr[:,i,1] / ms
        #starting from infinity
        plt.plot(oldvel*100,olddrag[:,i,1]*100/ms,color=colors[i],label="Infinity")
        plt.plot(x_line, y_line, color=colors[i],linestyle=":",
                label="Debye Length")
        x_linespace = velarrspace * 100
        y_linespace = 100 * dragarrspace[:,i,1] / ms
        print(y_linespace)
        #starting from infinity
        plt.plot(x_linespace, y_linespace, color=colors[i],linestyle="--",
                label="Particle spacing")


# ---- scatter-plotted simulation data, unchanged and NOT fitted ----
for k in range(10, len(runlist)):
    run = runlist[k]
    print(run)
    results = np.load("../dataarchive/np{}/results.npy".format(run))
    velarr_run = veldict[run]

    for i in range(4):
        if i == 0 or i==2:
            for j in range(len(velarr_run)):
                # print(results)
                if results[i,j,5] != -1:
                    count += 1
                    startt = results[i,j,4]
                    endt = results[i,j,5]
                    tarr = np.linspace(startt, endt, 10)
                    A = results[i,j,0]
                    eA = results[i,j,1]
                    tau = results[i,j,2]
                    etau = results[i,j,3]
                    varr = exp(tarr, tau, A)
                    aarr = varr/tau
                    verr = np.sqrt(np.square(varr*eA/A) + np.square(tarr*varr*etau/np.square(tau)))
                    aerr = aarr*np.sqrt(np.square(verr/varr) + np.square(etau/tau))

                    if labelled[i]:
                        plt.scatter(varr, aarr, color=colors[i])
                        plt.errorbar(varr, aarr, xerr=verr, yerr=np.abs(aerr), color=colors[i])
                    else:
                        plt.scatter(varr, aarr, color=colors[i],
                                    label=r"{:.1e}K,{:.1e}gcc, $\Gamma$={:.1e}".format(Tarr[i],gccarr[i],coupling[i]))
                        plt.errorbar(varr, aarr, xerr=verr, yerr=np.abs(aerr), color=colors[i])
                        labelled[i] = True

print(count)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("v (cm/s)")
plt.ylabel("a (cm/s^2)")
plt.show()
print(dragarr)