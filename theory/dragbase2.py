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
    def __init__(self,conditions,vres=100,rhores=300,ures=100,dphires=100,vrel_sigma_width=4.0,rhomax_fraction=0.3,dphi_endpoint_fraction=1e-5,acipc=1.0):
        self.vres=vres #resolution of velocity integration
        self.rhores=rhores #resolution of impact parameter integration
        self.ures = ures
        self.dphires = dphires
        self.vrel_sigma_width = vrel_sigma_width
        self.rhomax_fraction = rhomax_fraction
        if rhomax_fraction <= 0:
            raise ValueError("rhomax_fraction must be positive")
        self.dphi_endpoint_fraction = dphi_endpoint_fraction
        if not np.isclose(acipc, 1.0, rtol=0.0, atol=0.0):
            raise ValueError("acipc is fixed at 1: the finite-angle radius equals bmax")
        self.acipc = 1.0
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
            lower = min(1e-6*self.ustart,1e-18)
            upper = 1e1/rho[i]
            if self.umaxfunc(lower,rho[i],E)*self.umaxfunc(upper,rho[i],E)>0:
                print("Root Finding Error!")
            result[i]=brentq(self.umaxfunc,lower,upper,args=(rho[i],E),maxiter=1000)
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
            rhoi = rhoinf[i]
            free_arg = 1-(rhoi*uarr)**2
            yukawa_arg = free_arg-uarr*self.A*np.exp(-self.k0/uarr)/E
            sqrt_free = np.sqrt(free_arg)
            sqrt_yukawa = np.sqrt(yukawa_arg)
            # Y-free, rationalized to avoid subtracting nearly equal inverse roots.
            delta = rhoi*(free_arg-yukawa_arg)/(sqrt_free*sqrt_yukawa*(sqrt_free+sqrt_yukawa))
            results[i]=np.arcsin(rhoi*self.ustart)+np.sum(delta)*du
        return results
    def phiC(self,rho,E): #vectorized in rho
        u0 = self.umax(rho,E) #r of closest approach, maximum u
        C = self.A*np.exp(-self.k0/u0) 
        return self.pi/2-np.atan(C/(2*rho*(E)))#-self.phicutoff(rho,E,C) #start particle at rmax, subtract energy at rmax

    def dphiint(self,u,rho,E,C): #vectorized in u, scalar in E
        coulomb_arg = 1-(rho*u)**2-C*u/E
        yukawa_arg = 1-(rho*u)**2-u*self.A*np.exp(-self.k0/u)/E
        sqrt_coulomb = np.sqrt(coulomb_arg)
        sqrt_yukawa = np.sqrt(yukawa_arg)
        # Y-C with sqrt(a)-sqrt(b) rationalized as (a-b)/(sqrt(a)+sqrt(b)).
        difference = u*(self.A*np.exp(-self.k0/u)-C)/E
        return rho*difference/(sqrt_coulomb*sqrt_yukawa*(sqrt_coulomb+sqrt_yukawa))
    def Yint(self,u,rho,E): #vectorized in u
        return rho/(np.sqrt(1-((rho*u)**2)-(u*self.A*np.exp(-self.k0/u))/E))

    def Cint(self,u,rho,E,C): #vectorized in u, scalar in E
        arg = 1 - (rho*u)**2 - C*u/E
        return np.where(arg > 0, rho/np.sqrt(arg), 0.0)
    def dphi(self,rho,E,u0=None,C=None): #TODO: vectorized in rho
        """Integrate Phi_Y - Phi_C from infinity to closest approach."""
        if u0 is None:
            u0 = self.umax(rho,E) #find upper bound of u integral, vectorized in rho
        if C is None:
            C = self.A*np.exp(-self.k0/u0) #associated with rho
        steps = self.dphires
        frac = self.dphi_endpoint_fraction
        results = np.zeros(len(u0))
        for i in range(len(rho)):
            rhoi = rho[i]
            Ci = C[i]
            umin = min(1e-12*self.ustart,frac*u0[i])
            umax = (1-frac)*u0[i]
            uarr = np.linspace(umin,umax,steps)
            # plt.plot(uarr,np.abs(self.dphiint(uarr,rhoi,E,C[i])))
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.show()
            results[i] = np.nansum(self.dphiint(uarr,rhoi,E,Ci))*(uarr[1]-uarr[0])#vectorized in uarr
        return results
    def phiY_outer_cutoff(self,rho,E,u0):
        """Integrate Yukawa orbital angle from infinity to the angle cutoff."""
        results = np.zeros(len(rho))
        angle_radius_cutoff = self.acipc*self.rhomax_fraction/self.ustart
        cutoff_umax = 1/angle_radius_cutoff
        frac = self.dphi_endpoint_fraction
        for i in range(len(rho)):
            integration_umax = min(cutoff_umax,(1-frac)*u0[i])
            integration_umin = min(1e-12*self.ustart,frac*integration_umax)
            if integration_umax <= integration_umin:
                continue
            uarr = np.linspace(integration_umin,integration_umax,self.dphires)
            results[i] = np.nansum(self.Yint(uarr,rho[i],E))*(uarr[1]-uarr[0])
        return results
    def dphiYFree_outer_cutoff(self,rho,E,u0):
        """Integrate Yukawa-minus-free outer angle without cancellation."""
        results = np.zeros(len(rho))
        angle_radius_cutoff = self.rhomax_fraction/self.ustart
        cutoff_umax = 1/angle_radius_cutoff
        frac = self.dphi_endpoint_fraction
        for i in range(len(rho)):
            integration_umax = min(cutoff_umax,(1-frac)*u0[i])
            integration_umin = min(1e-12*self.ustart,frac*integration_umax)
            if integration_umax <= integration_umin:
                continue
            uarr = np.linspace(integration_umin,integration_umax,self.dphires)
            rhoi = rho[i]
            free_arg = 1-(rhoi*uarr)**2
            yukawa_term = uarr*self.A*np.exp(-self.k0/uarr)/E
            yukawa_arg = free_arg-yukawa_term
            sqrt_free = np.sqrt(free_arg)
            sqrt_yukawa = np.sqrt(yukawa_arg)
            # Y-free, rationalized to avoid subtracting nearly equal roots.
            integrand = rhoi*yukawa_term/(sqrt_free*sqrt_yukawa*(sqrt_free+sqrt_yukawa))
            results[i] = np.nansum(integrand)*(uarr[1]-uarr[0])
        return results
    def scattering_half_angle(self,rho,E):
        """Return the finite-start theta/2 using a Rutherford reference.

        The infinite-radius Yukawa result is the analytic Rutherford
        half-angle minus the nonsingular Yukawa-Coulomb orbital-angle
        difference.  The full finite-start scattering angle then subtracts
        twice the Yukawa orbital-angle change from infinity to the starting
        radius.  "Angle change" is relative to free straight-line geometry,
        so the Yukawa outer angle and free outer angle must be differenced;
        this guarantees zero deflection when the potential vanishes.
        """
        u0 = self.umax(rho,E)
        C = self.A*np.exp(-self.k0/u0)
        coulomb_half_angle = np.arctan(C/(2*rho*E))
        coulomb_yukawa_difference = self.dphi(rho,E,u0=u0,C=C)
        yukawa_free_outer_difference = self.dphiYFree_outer_cutoff(rho,E,u0)
        return (
            coulomb_half_angle
            - coulomb_yukawa_difference
            - yukawa_free_outer_difference
        )
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
    def _drag_speed_integrand(self,speed,weight): #positive speed magnitude, signed Maxwellian weight
        if speed <= 0.0 or weight == 0.0:
            return 0.0
        E = 0.5*self.mu*speed**2+self.E0Y #in this code, always use energy relative to infinity
        vinf = np.sqrt(E/(0.5*self.mu))
        #rhoup is maximum impact parameter at infinity
        rhoup = self.rhomax_fraction*speed/(vinf*self.ustart)
        rhoarr = np.zeros(self.rhores)
        for j in range(self.rhores): #rhoarr spaced such that equal area
            if j == 0:
                rhoarr[0] = 0.5 * np.sqrt(rhoup**2/self.rhores)   # small nonzero midpoint
            else:
                rhoarr[j] = np.sqrt(rhoup**2/self.rhores + rhoarr[j-1]**2)
        #1. start by evolving trajectories from infinity
        vstartphi = rhoarr*vinf*self.ustart
        alpha = np.arcsin(vstartphi/speed)
        # print("max alpha,phiinfstart")
        # print(np.max(alpha),np.max(phiinfstart))
        rhostart = np.sin(alpha)/self.ustart
        #2. keep trajectories with rhostart below the finite-launch cutoff
        cutoff = self.rhomax_fraction/self.ustart
        keep = rhostart <= cutoff
        if not np.any(keep):
            return 0.0
        last = np.where(keep)[0][-1] + 1
        rhostart = rhostart[:last]
        if len(rhostart) < 2:
            return 0.0
        #deflection in v for whole collision
        half_theta = self.scattering_half_angle(rhoarr,E)[:last]
        drhostart = np.zeros(len(rhostart))
        for j in range(len(rhostart)):
            if j == 0:
                drhostart[0] = (rhostart[1]+rhostart[0])/2
            elif j == len(rhostart)-1:
                drhostart[j] = (rhostart[j])-(rhostart[j] + rhostart[j-1])/2
            else:
                drhostart[j] = (rhostart[j+1] + rhostart[j])/2-(rhostart[j] + rhostart[j-1])/2
        #for cross section purposes, want impact parameter at ustart, rstart
        return np.sum(
            rhostart
            * drhostart
            * speed**2
            * weight
            * (2.0 * np.square(np.sin(half_theta)))
        )

    def drag(self,vb): #Newtons, vectorized in v
        sigmav = np.sqrt(self.kb*self.T/self.mu) #standard deviation in velocity due to thermal effects
        # for particle not starting at infinity, minium energy of particle is nonzero
        # Pair positive and negative relative velocities at the same speed.
        # This avoids accumulating two large thermal lobes whose difference is
        # the small low-drift drag signal.
        width = self.vrel_sigma_width*sigmav
        vmin = vb-width
        vmax = vb+width
        speed_min = 0.0 if vmin <= 0.0 <= vmax else min(abs(vmin),abs(vmax))
        speed_max = max(abs(vmin),abs(vmax))
        if self.vres < 1 or speed_max <= speed_min:
            return 0.0
        ds = (speed_max-speed_min)/self.vres
        speeds = speed_min+(np.arange(self.vres,dtype=np.float64)+0.5)*ds
        norm = np.sqrt(self.mu/(2*self.pi*self.kb*self.T))
        print("phi")
        result=0
        for speed in speeds:
            positive_weight = 0.0
            negative_weight = 0.0
            if vmin <= speed <= vmax:
                positive_weight = norm*np.exp(-self.mu*((speed-vb)**2)/(2*self.kb*self.T))
            if vmin <= -speed <= vmax:
                negative_weight = norm*np.exp(-self.mu*((-speed-vb)**2)/(2*self.kb*self.T))
            signed_weight = positive_weight-negative_weight
            result += self._drag_speed_integrand(speed,signed_weight)
        return 2*self.pi*self.nh*self.mu*result*ds #drag force on silicon
if __name__ == "__main__":
    vb = 1e4
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
