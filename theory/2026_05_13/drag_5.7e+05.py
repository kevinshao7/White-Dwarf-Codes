import numpy as np
import scipy as scp
from scipy.integrate import quad
from scipy.optimize import newton,minimize,brentq
#don't integrate deflection angle to infinite radius
#add initial energy of Yukawa at umin
#3D gaussian projected onto 1D
class DragFourth:
    NA = np.float64(6.022e23)
    pi = np.float64(3.1415926535)
    me = np.float64(9.11e-31) #electron mass
    ms = np.float64(28.0855*0.001/NA)
    mh = np.float64(1.008*0.001/NA)
    mu = np.float64(ms*mh/(ms+mh))
    qe = np.float64(1.6e-19)
    hbar = np.float64(6.626e-34/(2*pi))
    kb = np.float64(1.38e-23)
    e0 =  np.float64(8.854e-12)
    Z1arr = np.array([0.16,0.65,0.93,0.68],dtype=np.float64)
    Z2arr = np.array([0.26,3.82,4.27,3.81],dtype=np.float64)
    gccarr =  np.array([1e-5,1,1e-5,1],dtype=np.float64)
    Tarr = np.array([5000,5000,1e5,1e5],dtype=np.float64)
    def __init__(self,conditions,vres=100,rhores=300,ures=100):
        self.vres=vres #resolution of velocity integration
        self.rhores=rhores #resolution of impact parameter integration
        self.ures = ures
        self.z1 = self.Z1arr[conditions]
        self.z2 = self.Z2arr[conditions]
        self.gcc = self.gccarr[conditions]
        self.T = self.Tarr[conditions]
        self.nh = 1e6*self.gcc/(1000*self.mh)
        ne =1e6* self.NA*self.z1*self.gcc/1.008 #electron number density in SI
        ve = np.sqrt(3*self.kb*self.T/self.me)
        wp = np.sqrt(ne*(self.qe**2)/(self.me*self.e0))
        self.lD = ve/wp
        interparticlespacing = (3/(4*self.pi*self.nh))**(1/3)
        self.ustart = 1/interparticlespacing
        Tf = ((3*ne*(self.pi**2))**(2/3))*(self.hbar**2)/(2*self.me*self.kb)
        lS = self.lD*np.power(1+(2*Tf/(3*self.T)),1/4)
        self.k0 = 1/lS #screening length in m
        self.A = (self.qe**2)*self.z1*self.z2/(4*self.pi*self.e0)
        self.E0Y = self.A*np.exp(-self.k0/self.ustart)*self.ustart #background energy Yukawa at rmax/umin
        #test no force
        # self.A = 0.0
        # self.E0Y = 0.0
    def rhostarttorhoinf(self,rhostart,vstart,Estart): #convert between impact parameter at starting length to infinite length
        #rhostart*vstart= rhoinf*vinf
        #vectorized in rho 
        #v is velocity at 
        vinf = np.sqrt(2*Estart/self.mu)
        return rhostart*np.abs(vstart)/vinf
    def rhoinftorhostart(self,rhostart,vstart,Estart):
        vinf = np.sqrt(2*Estart/self.mu)
        return rhostart*vinf/np.abs(vstart)
    def maxrhoinf(self,E):
        #make sure particle actually reaches ustart, rstart
        #0 = (1-(self.A*self.ustart*np.exp(-self.k0/self.ustart)/(E))-((rho*self.ustart)**2))
        # ((rho*self.ustart)**2) = 1-(self.A*self.ustart*np.exp(-self.k0/self.ustart)/(E))
        maxrho = np.sqrt(1-(self.A*self.ustart*np.exp(-self.k0/self.ustart)/(E)))/self.ustart

        return maxrho
    def frel(self,vrel,vb):
        return np.sqrt(self.mu/(2*self.pi*self.kb*self.T))*np.exp(-self.mu*np.square(vrel-vb)/(2*self.kb*self.T)) #output velocity distribution of maxwellian centered at vb, m/s
    def umaxfunc(self,u,rho,E): #at rmin, this function is 0, all scalar, vectorized in u
        return (1-(self.A*u*np.exp(-self.k0/u)/(E))-((rho*u)**2))
    def umax(self,rho,E): #vectorized in rho
        # guessuarr = np.logspace(-np.log10(self.lD),-np.log10(self.lD)+5,100)
        # return newton(umaxfunc,x0=1/lD,fprime=dfdr,args=(rho,A,k0,E),maxiter=500,rtol=1e-30)
        result = np.zeros(len(rho))
        for i in range(len(rho)):
            if self.umaxfunc(1e-6*self.ustart,rho[i],E)*self.umaxfunc(1e1/rho[i],rho[i],E)>0:
                print("Root Finding Error!")
            result[i]=brentq(self.umaxfunc,min(1e-6*self.ustart,1e-18),1e1/rho[i],args=(rho[i],E))
        # guess = 1e8
        return result
        # return newton(umaxfunc,fprime= dumaxfunc,x0=guess,args=(rho,A,k0,E),maxiter=2000,rtol=1e-20)
    #phi is angle of closest approach
    
    def phiinfstart(self,rhoinf,E): 
    #numerically integrate excess phiY from infinity to start, vectorized in rho
        results=np.zeros(len(rhoinf))
        for i in range(len(rhoinf)):
            uarr = np.linspace(1e-12*self.ustart,self.ustart,self.ures,endpoint=False)
            du = self.ustart/self.ures
            results[i]=np.sum(self.Yint(uarr,rhoinf[i],E))*du
        return results
    def phiC(self,rho,E): #vectorized in rho
        u0 = self.umax(rho,E) #r of closest approach, maximum u
        C = self.A*np.exp(-self.k0/u0) 
        return self.pi/2-np.atan(C/(2*rho*(E)))#-self.phicutoff(rho,E,C) #start particle at rmax, subtract energy at rmax

    def dphiint(self,u,rho,E,C): #vectorized in u, scalar in E
        # return self.Yint(u,rho,E)-self.Cint(u,rho,E,C)
        return rho*(np.sqrt(1-(rho*u)**2-C*u/E)-np.sqrt(1-((rho*u)**2)-(u*self.A*np.exp(-self.k0/u))/E))/(np.sqrt((1-((rho*u)**2)-(u*self.A*np.exp(-self.k0/u))/E)*(1-(rho*u)**2-C*u/E)))
    def Yint(self,u,rho,E): #vectorized in u
        return rho/(np.sqrt(1-((rho*u)**2)-(u*self.A*np.exp(-self.k0/u))/E))

    def Cint(self,u,rho,E,C): #vectorized in u, scalar in E
        arg = 1 - (rho*u)**2 - C*u/E
        return np.where(arg > 0, rho/np.sqrt(arg), 0.0)
    def dphi(self,rho,E): #TODO: vectorized in rho
        u0 = self.umax(rho,E) #find upper bound of u integral, vectorized in rho
        C = self.A*np.exp(-self.k0/u0) #associated with rho
        steps = 100
        frac = 1e-5
        results = np.zeros(len(u0))
        for i in range(len(rho)):
            rhoi = rho[i]
            Ci = C[i]
            uarr = np.linspace(0,(1-frac)*u0[i],steps)
            # plt.plot(uarr,np.abs(self.dphiint(uarr,rhoi,E,C[i])))
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.show()
            results[i] = np.nansum(self.dphiint(uarr,rhoi,E,Ci))*(uarr[1]-uarr[0])#vectorized in uarr
        return results
    def phiY(self,rho,E): #vectorized in rho, scalar in E 
        dPhi = self.dphi(rho,E)
        # nan_idx = np.where(np.isnan(phicutoff))[0]
        # idx = nan_idx[0] if len(nan_idx) > 0 else None
        # #maximum physical rho
        # if idx == None:
        #     return dPhi+self.phiC(rho,E)-phicutoff #coloumb from rmin to infinity, add difference from rmin to infinity, subtract yukawa from rmax to infinity
        # else:
        #     return (dPhi+self.phiC(rho,E)-phicutoff)[:idx] #coloumb from rmin to infinity, add difference from rmin to infinity, subtract yukawa from rmax to infinity
        return dPhi+self.phiC(rho,E)
    # def dragint()
    def drag(self,vb): #Newtons, vectorized in v
        sigmav = np.sqrt(self.kb*self.T/self.mu) #standard deviation in velocity due to thermal effects
        # for particle not starting at infinity, minium energy of particle is nonzero
        #varr is velocity at rstart
        varr = np.linspace(vb-4*sigmav,vb+4*sigmav,self.vres)
        rhoupcandidate = max([self.lD,self.nh**(-1/3)])

        #rhoarr is evenly spaced in area at starting ustart, rstart
        # rhoarr = np.linspace(1e-24,rhoup,self.rhores)
        # drho = rhoarr[1]-rhoarr[0]
        dv = varr[1]-varr[0]
        result=0
        Earr = 0.5*self.mu*np.square(varr)+self.E0Y #in this code, always use energy relative to infinity
        fvarr = np.sqrt(self.mu/(2*self.pi*self.kb*self.T))*np.exp(-self.mu*((varr-vb)**2)/(2*self.kb*self.T))
        #angle between infinity and closest approach
        phiinfmin=np.zeros((self.vres,self.rhores))
        #angle between infinity and start
        phiinfstart=np.zeros((self.vres,self.rhores))
        print("phi")
        for i in range(len(varr)):
            vinf = np.sqrt(Earr[i]/(0.5*self.mu))
            #rhoup is maximum impact parameter at infinity
            rhoup = 0.3*np.abs(varr[i])/(vinf*self.ustart)
            rhoarr = np.zeros(self.rhores)
            for j in range(self.rhores): #rhoarr spaced such that equal area
                if j == 0:
                    rhoarr[0] = 0.5 * np.sqrt(rhoup**2/self.rhores)   # small nonzero midpoint
                else:
                    rhoarr[j] = np.sqrt(rhoup**2/self.rhores + rhoarr[j-1]**2)
            #1. start by evolving trajectories from infinity
            phiinfmin[i,:] = self.phiY(rhoarr,Earr[i])#deflection angle due to yukawa
            phiinfstart[i,:] = self.phiinfstart(rhoarr,Earr[i])
            vstartphi = rhoarr*vinf*self.ustart
            alpha = np.arcsin(vstartphi/np.abs(varr[i]))
            # print("max alpha,phiinfstart")
            # print(np.max(alpha),np.max(phiinfstart))
            rhostart = np.sin(alpha)/self.ustart
            #2. keep trajectories with rhostart<0.4 rstart
            maxi = np.argmin(np.abs(rhostart-0.3/self.ustart))

            rhostart = rhostart[:maxi]
            # print(maxi/self.rhores)
            #deflection in v for whole collision
            theta = 2*(self.pi/2-phiinfmin[i,:]-alpha+phiinfstart[i,:])[:maxi]
            drhostart = np.zeros(len(rhostart))
            for j in range(len(rhostart)):
                if j == 0:
                    drhostart[0] = (rhostart[1]+rhostart[0])/2
                elif j == len(rhostart)-1:
                    drhostart[j] = (rhostart[j])-(rhostart[j] + rhostart[j-1])/2
                else:
                    drhostart[j] = (rhostart[j+1] + rhostart[j])/2-(rhostart[j] + rhostart[j-1])/2
            #for cross section purposes, want impact parameter at ustart, rstart
            # print(np.min(theta),np.max(theta))
            result += np.sum(rhostart*drhostart*(varr[i]*abs(varr[i]))*fvarr[i]*(1-np.cos(theta))) 
        return 2*self.pi*self.nh*self.mu*result*dv #drag force on silicon
vb=5.7e+05
y = np.zeros(4)
y_head = np.zeros(4)
y_sigma = np.zeros(4)
"""
#let phifree be angle between rstart and finite impact parameter at rstart, rhostart
#rhostart is distance of closest approach if test particle was let go at rstart with no force
#let phi be angle between infinity (parallel to velocity at rstart) and point of closest approach for force, particle starts at rstart
#during trajectory from rstart to point of closest approach, velocity deflection is pi/2-phi

My approach will be 
1. evolve trajectoriess from infinity using phicutoff, 
    evaluate to rstart, integrate yukawa phi
    phicutoff is angle between velocity at infinity and particle position at rstart.
2. At rstart, calculate angle of velocity using conservation of angular momentum
3. keep trajectories that have impact parameter calculated using
    position and velocity of rstart under 0.4*rstart
4. For these selected trajectories, consider velocity direction 
    difference between rstart and r closest approach





"""

#velocity deflection from finit

for i in range(4):
    calcdrag = DragFourth(i)
    y[i] = calcdrag.drag(vb)
    # y_head[i] = calcdrag.drag_head_on(vb)
    # y_sigma[i] = calcdrag.sigma_transport(vb)

out = np.zeros((4, 3))
out[:, 0] = y_sigma
out[:, 1] = y
out[:, 2] = y_head

np.save("theory_interparticle_max_{:.1e}".format(vb), out)
print("Saved columns: sigma_transport [m^2], drag [N], drag_head_on [N]")
print("Fully Successful")
print(y)
