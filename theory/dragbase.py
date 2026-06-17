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
    #exp(-Ax)/x = B
    #x = W(a/b)/aq
    Z1arr = np.array([0.16,0.65,0.93,0.68],dtype=np.float64)
    Z2arr = np.array([0.26,3.82,4.27,3.81],dtype=np.float64)
    gccarr =  np.array([1e-5,1,1e-5,1],dtype=np.float64)
    Tarr = np.array([5000,5000,1e5,1e5],dtype=np.float64)
    def __init__(self,conditions,vres=100,rhores=100,ures=100):
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

        self.umin = self.nh**(1/3)


        Tf = ((3*ne*(self.pi**2))**(2/3))*(self.hbar**2)/(2*self.me*self.kb)
        lS = self.lD*np.power(1+(2*Tf/(3*self.T)),1/4)
        self.k0 = 1/lS #screening length in m
        self.A = (self.qe**2)*self.z1*self.z2/(4*self.pi*self.e0)
        self.E0Y = self.A*np.exp(-self.k0/self.umin)*self.umin #background energy Yukawa at rmax/umin
    def frel(self,vrel,vb):
        return np.sqrt(self.mu/(2*self.pi*self.kb*self.T))*np.exp(-self.mu*np.square(vrel-vb)/(2*self.kb*self.T)) #output velocity distribution of maxwellian centered at vb, m/s
    def umaxfunc(self,u,rho,E): #at rmin, this function is 0, all scalar, vectorized in u
        return (1-(self.A*u*np.exp(-self.k0/u)/(E))-((rho*u)**2))
    def umax(self,rho,E): #vectorized in rho
        # guessuarr = np.logspace(-np.log10(self.lD),-np.log10(self.lD)+5,100)
        # return newton(umaxfunc,x0=1/lD,fprime=dfdr,args=(rho,A,k0,E),maxiter=500,rtol=1e-30)
        result = np.zeros(len(rho))
        for i in range(len(rho)):
            # print(E)
            if self.umaxfunc(1e-6*self.umin,rho[i],E)*self.umaxfunc(1e1/rho[i],rho[i],E)>0:
                print("Root Finding Error!")
            result[i]=brentq(self.umaxfunc,min(1e-6*self.umin,1e-18),1e1/rho[i],args=(rho[i],E))
        # guess = 1e8
        return result
        # return newton(umaxfunc,fprime= dumaxfunc,x0=guess,args=(rho,A,k0,E),maxiter=2000,rtol=1e-20)
    #phi is angle of closest approach
    
    def phicutoff(self,rho,E,C): #numerically integrate excess phiC
        results=np.zeros(len(rho))
        umaxarr = self.umax(rho,E)
        for i in range(len(rho)):
            uarr = np.linspace(0,umaxarr[i],self.ures)
            du = umaxarr[i]/self.ures
            results[i]=np.sum(self.Cint(uarr,rho[i],E,C[i]))*du
        return results
    def phiC(self,rho,E): #vectorized in rho
        u0 = self.umax(rho,E)
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
        return dPhi+self.phiC(rho,E)
    # def dragint()
    def drag(self,vb): #Newtons, vectorized in v
        sigmav = np.sqrt(self.kb*self.T/self.mu) #standard deviation in velocity due to thermal effects

        varr = np.linspace(vb-4*sigmav,vb+4*sigmav,self.vres)
        rhoup = max([self.lD,self.nh**(-1/3)])#upper bound of integration for impact parameter
        rhoarr = np.zeros(self.rhores)
        drho = np.zeros(self.rhores)
        for i in range(self.rhores):
            if i == 0:
                drho[0] = np.sqrt(rhoup**2/self.rhores)
                rhoarr[0] = 0.5 * drho[0]   # small nonzero midpoint
            else:
                rhoarr[i] = np.sqrt(rhoup**2/self.rhores + rhoarr[i-1]**2)
                drho[i] = rhoarr[i] - rhoarr[i-1]
        # rhoarr = np.linspace(1e-24,rhoup,self.rhores)
        # drho = rhoarr[1]-rhoarr[0]
        dv = varr[1]-varr[0]
        result=0
        Earr = 0.5*self.mu*np.square(varr)#+ self.E0Y
        fvarr = np.sqrt(self.mu/(2*self.pi*self.kb*self.T))*np.exp(-self.mu*((varr-vb)**2)/(2*self.kb*self.T))
        phiyarr=np.zeros((self.vres,self.rhores))
        print("phi")
        for i in range(len(varr)):
            phiyarr[i,:] = self.phiY(rhoarr,Earr[i])#deflection angle due to yukawa
        # phiyarr = np.minimum(phiyarr,self.pi/2)
        # phiyarr=np.maximum(phiyarr,0)
        print("drag")
        for i in range(len(varr)):
            result += np.sum(rhoarr*drho*(varr[i]*abs(varr[i]))*fvarr[i]*(1-np.cos(self.pi-2*phiyarr[i,:]))) 
        return 2*self.pi*self.nh*self.mu*result*dv #drag force on silicon
    # def head_on_phi(self, E, rho_frac=1e-8):
    #     """
    #     Approximate the head-on (rho -> 0) scattering angle.
    #     Uses a tiny nonzero impact parameter because the current
    #     phiY machinery is not numerically safe at exactly rho = 0.
    #     """
    #     rhoup = max(self.lD, self.nh**(-1/3))
    #     rho_eps = max(rhoup * rho_frac, 1e-30)
    #     phi = self.phiY(np.array([rho_eps], dtype=np.float64), E)[0]
    #     phi = min(phi, self.pi / 2)
    #     phi = max(phi, 0.0)
    #     return phi

    # def drag_head_on(self, vb, rho_frac=1e-8):
    #     """
    #     Debug drag model: assume every collision is head-on.
    #     That is, replace the impact-parameter-dependent scattering
    #     angle by the rho -> 0 angle for the same collision energy.

    #     This is NOT a physically correct cross-section treatment;
    #     it is only for debugging the high-velocity scaling.
    #     """
    #     sigmav = np.sqrt(self.kb * self.T / self.mu)

    #     varr = np.linspace(vb - 4 * sigmav, vb + 4 * sigmav, self.vres)
    #     dv = varr[1] - varr[0]

    #     rhoup = max(self.lD, self.nh**(-1/3))
    #     area = self.pi * rhoup**2

    #     Earr = 0.5 * self.mu * np.square(varr)
    #     fvarr = np.sqrt(self.mu / (2 * self.pi * self.kb * self.T)) * \
    #             np.exp(-self.mu * np.square(varr - vb) / (2 * self.kb * self.T))

    #     result = 0.0
    #     print("head-on phi")
    #     for i in range(len(varr)):
    #         phi0 = self.head_on_phi(Earr[i], rho_frac=rho_frac)
    #         result += (varr[i]*abs(varr[i])) * fvarr[i] * (1 - np.cos(self.pi - 2 * phi0))

    #     return self.nh * self.mu * area * result * dv
    # def sigma_transport(self, vb):  # m^2, thermally averaged transport cross section
    #     sigmav = np.sqrt(self.kb * self.T / self.mu)

    #     varr = np.linspace(max(1e-5, vb - 4 * sigmav), vb + 4 * sigmav, self.vres)
    #     rhoup = max([self.lD, self.nh**(-1/3)])

    #     rhoarr = np.zeros(self.rhores)
    #     drho = np.zeros(self.rhores)
    #     for i in range(self.rhores):
    #         if i == 0:
    #             drho[0] = np.sqrt(rhoup**2 / self.rhores)
    #             rhoarr[0] = 0.5 * drho[0]
    #         else:
    #             rhoarr[i] = np.sqrt(rhoup**2 / self.rhores + rhoarr[i-1]**2)
    #             drho[i] = rhoarr[i] - rhoarr[i-1]

    #     dv = varr[1] - varr[0]
    #     Earr = 0.5 * self.mu * np.square(varr)
    #     fvarr = np.sqrt(self.mu / (2 * self.pi * self.kb * self.T)) * \
    #             np.exp(-self.mu * ((varr - vb)**2) / (2 * self.kb * self.T))

    #     phiyarr = np.zeros((self.vres, self.rhores))
    #     print("sigma transport phi")
    #     for i in range(len(varr)):
    #         phiyarr[i, :] = self.phiY(rhoarr, Earr[i])

    #     phiyarr = np.minimum(phiyarr, self.pi / 2)
    #     phiyarr = np.maximum(phiyarr, 0)

    #     result = 0.0
    #     print("sigma transport")
    #     for i in range(len(varr)):
    #         result += np.sum(rhoarr * drho * fvarr[i] * (1 - np.cos(self.pi - 2 * phiyarr[i, :])))

    #     return 2 * self.pi * result * dv
vb = 1e4
y = np.zeros(4)
y_head = np.zeros(4)
y_sigma = np.zeros(4)

for i in range(4):
    calcdrag = DragFourth(i)
    y[i] = calcdrag.drag(vb)
    # y_head[i] = calcdrag.drag_head_on(vb)
    # y_sigma[i] = calcdrag.sigma_transport(vb)

out = np.zeros((4, 3))
out[:, 0] = y_sigma
out[:, 1] = y
out[:, 2] = y_head

np.save("theory_rmaxDB_{:.1e}".format(vb), out)
print("Saved columns: sigma_transport [m^2], drag [N], drag_head_on [N]")
print("Fully Successful")