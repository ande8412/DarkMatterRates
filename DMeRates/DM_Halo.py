from .Constants import *
import numericalunits as nu
class DM_Halo_Distributions:
    def __init__(self,V0=None,VEarth=None,VEscape=None,RHOX=None,crosssection=None):


        self.device='cpu'


        if V0 is None:
            self.v0 = v0
        else:
            self.v0 = V0 


        if VEarth is None:
            self.vEarth = vEarth 
        else:
            self.vEarth = VEarth
        
        if VEscape is None:
            self.vEscape = vEscape 
        else:
            self.vEscape = VEscape 

        if RHOX is None:
            self.rhoX = rhoX
        else:
            self.rhoX = RHOX
        if crosssection is None:
            self.cross_section = crosssection 
        else:
            self.cross_section = crosssection

    def optimize(self, device):
        self.device = device

    def generate_halo_files(self,model):
        import numpy as np
        vmax = self.vEarth + self.vEscape
        vMins = np.linspace(0,vmax,1000)

        #this next step could use some vectorization, but I will be a bit lazy here
        etas = []
        for v in vMins:
            if model == 'shm':
                eta = self.etaSHM(v)
                #eta = etaSHM(vmin,params) # (cm/s)^-1 
            elif model == 'tsa':
                eta = self.etaTsa(v)
            elif model == 'dpl':
                eta = self.etaDPL(v)
            # elif model == 'msw':
            #     eta = etaMSW(v,_params)
            # elif model == 'debris':
            #     eta = etaDebris(v,_params)
            else:
                print("Undefined halo parameter. Options are ['shm','tsa','dpl'] Perhaps 'msw','debris' will be available soon")
            etas.append(eta)
        etas = np.array(etas)
        lightSpeed_kmpers = nu.s / nu.km #inverse to output it in units i want
        geVconversion = 1 / (nu.GeV / nu.c0**2 / nu.cm**3)
        import os
        module_dir = os.path.dirname(__file__)
        filepath = os.path.join(module_dir,'../halo_data')

        with open(f'{filepath}/{model}_v0{np.round(self.v0*lightSpeed_kmpers,1)}_vE{np.round(self.vEarth*lightSpeed_kmpers,1)}_vEsc{np.round(self.vEscape*lightSpeed_kmpers,1)}_rhoX{np.round(self.rhoX*geVconversion,1)}.txt','w') as f:
            for i in range(len(vMins)):
                v = vMins[i] / (nu.km / nu.s)
                e = etas[i] / (nu.s / nu.km)
                f.write(f"{v}\t{e}\n") #[km/s], [s/km]
        return
    

    def vmin(self,EE,q,mX):
        #assume mX is in eV without numerical units applied
        vmin = ((EE/q)+(q/(2*mX))) / nu.c0
        return vmin


    def vmin_tensor(self,E,q,mX):
        import torch
        q_tiled = torch.tile(q,(len(E),1))
        EE_tiled = torch.tile(E,(len(q),1)).T
        vmins = ((EE_tiled/q_tiled)+(q_tiled/(2*mX)))
        return  vmins
    


    # def eta_MB(self,vMin):    #same as SHM but the indefinite integration is done so this is faster. In units of inverse vMin
    #     import numpy as np
    #     from scipy.special import erf
    #     if (vMin < self.vEscape - self.vEarth):
    #         val = -4.0*self.vEarth*np.exp(-(self.vEscape/self.v0)**2) + np.sqrt(np.pi)*self.v0*(erf((vMin+self.vEarth)/self.v0) - erf((vMin - self.vEarth)/self.v0))
    #     elif (vMin < self.vEscape + self.vEarth):
    #         val = -2.0*(self.vEarth+self.vEscape-vMin)*np.exp(-(self.vEscape/self.v0)**2) + np.sqrt(np.pi)*self.v0*(erf(self.vEscape/self.v0) - erf((vMin - self.vEarth)/self.v0))
    #     else:
    #         val = 0.0
    #     K = (self.v0**3)*(-2.0*np.pi*(self.vEscape/self.v0)*np.exp(-(self.vEscape/self.v0)**2) + (np.pi**1.5)*erf(self.vEscape/self.v0))
    #     return (self.v0**2)*np.pi/(2.0*self.vEarth*K)*val
    
    def etaSHM(self,vMin):
        from scipy.integrate import quad, dblquad, nquad
        from scipy.special import erf
        import numpy as np
        """
        Standard Halo Model with sharp cutoff. 
        Fiducial values are v0=220 km/s, vE=232 km/s, vesc= 544 km/s
        params = [v0, vE, vesc]
        """
        v0 = self.v0
        vE = self.vEarth
        vesc = self.vEscape
        KK=v0**3*(-2.0*np.exp(-vesc**2/v0**2)*np.pi*vesc/v0+np.pi**1.5*erf(vesc/v0))
    #    print('KK=',KK)
        def func(vx2):
            return np.exp(-vx2/(v0**2))

        if vMin <= vesc - vE:
            # eq. B4 from 1509.01598
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/KK)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0]
        elif vesc - vE < vMin <= vesc + vE:
            # eq. B5 from 1509.01598
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)] 
            def bounds_vX():
                return [vMin, vE+vesc]
            def eta(cosq,vx):
                return (2*np.pi/KK)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0] 
        else:
            return 0 

    def eta_MB_tensor(self,vMin_tensor):
        """Integrated Maxwell-Boltzmann Distribution"""
        import torch
        eta = torch.zeros_like(vMin_tensor,device=self.device)
        val_below = -4.0*self.vEarth*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + torch.sqrt(torch.tensor(torch.pi,device=self.device))*self.v0*(torch.erf((vMin_tensor+self.vEarth)/self.v0) - torch.erf((vMin_tensor - self.vEarth)/self.v0))
        val_above = -2.0*(self.vEarth+self.vEscape-vMin_tensor)*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + torch.sqrt(torch.tensor(torch.pi,device=self.device))*self.v0*(torch.erf(torch.tensor(self.vEscape/self.v0,device=self.device)) - torch.erf((vMin_tensor - self.vEarth)/self.v0))
        eta = torch.where(vMin_tensor < self.vEscape + self.vEarth, val_above,eta)
        eta = torch.where(vMin_tensor < self.vEscape - self.vEarth, val_below,eta)
        K = (self.v0**3)*(-2.0*torch.pi*(self.vEscape/self.v0)*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + (torch.pi**1.5)*torch.erf(torch.tensor(self.vEscape/self.v0,device=self.device)))
        etas = (self.v0**2)*torch.pi/(2.0*self.vEarth*K)*eta 

        #not sure if etas is allowed to be zero.
        etas = torch.where(etas < 0,0,etas)
        

        return etas
    

    def etaTsa(self,vMin):

        from scipy.integrate import nquad
        from Constants import q_Tsallis
        import numpy as np
        """
        Tsallis Model, q = .773, v0 = 267.2 km/s, and vesc = 560.8 km/s
        give best fits from arXiv:0902.0009. 
        params = [v0, vE, q]
        """
        q = q_Tsallis

        if q <1:
            vesc = self.v0/np.sqrt(1-q)
        else:
            vesc = self.vEscape
        def func(vx2):
            if q == 1:
                return np.exp(-vx2/self.v0**2)
            else:
                return (1-(1-q)*vx2/self.v0**2)**(1/(1-q))
        " calculate normalization constant "
        def inttest(vx):
            if q == 1:
                if vx <= vesc:
                    return vx**2*np.exp(-vx**2/self.v0**2)
                else:
                    return 0 
            else:
                if vx <= vesc:
                    return vx**2*(1-(1-q)*vx**2/self.v0**2)**(1/(1-q))
                else:
                    return 0            
        def bounds():
            return [0.,vesc]
        K_=4*np.pi*nquad(inttest,[bounds])[0]

    #    K_ = 4/3*np.pi*vesc**3*hyp2f1(3/2, 1/(q-1), 5/2, (1-q)*vesc**2/v0**2) # analytic expression, runs faster
        
        if vMin <= vesc - self.vEarth:
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*self.vEarth+np.sqrt((cosq**2-1)*self.vEarth**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/K_)*vx*func(vx**2+self.vEarth**2+2*vx*self.vEarth*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0]
            
        elif vesc - self.vEarth < vMin <= vesc + self.vEarth:
            def bounds_cosq(vx):
                return [-1, (vesc**2-self.vEarth**2-vx**2)/(2*vx*self.vEarth)] 
            def bounds_vX():
                return [vMin, self.vEarth+vesc]
            def eta(cosq,vx):
                return (2*np.pi/K_)*vx*func(vx**2+self.vEarth**2+2*vx*self.vEarth*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0]
        else:
            return 0   


    def etaDPL(self,vMin): #FIX UNITS
        from Constants import k_DPL
        from scipy.integrate import nquad
        import numpy as np
        """
        Double Power Law Profile, 1.5 <= k <= 3.5 found to give best fit to N-body
        simulations. 
        takes input velocities in km/s
        params = [vMin, v0, vE, vesc, k]
        """
        v0 = self.v0
        vE = self.vEarth
        vesc = self.vEscape
        k = k_DPL
        
        def func(vx2):
            return (np.exp((vesc**2-vx2)/(k*v0**2))-1)**k
        " calculate normalization constant "
        def inttest(vx):
            if vx <= vesc:
                return vx**2*(np.exp((vesc**2-vx**2)/(k*v0**2))-1)**k
            else:
                return 0
        def bounds():
            return [0.,vesc]
        K_=4*np.pi*nquad(inttest,[bounds])[0]   
        
        if vMin <= vesc - vE:
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0]
            
        elif vesc - vE < vMin <= vesc + vE:
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)] 
            def bounds_vX():
                return [vMin, vE+vesc]
            def eta(cosq,vx):
                return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0]
        else:
            return 0

    def step_function_eta(self,vMins, params): #CHECK UNITS
        #vMins is 2d array
        import torch
        num_steps = params.shape[0]
        # vMax = (vEarth + vEscape)*1.1 #can change this later
        vMax = 1000 *nu.km / nu.s
        
        vis = torch.arange(0,vMax,step = vMax/num_steps,device=self.device)
        if params.device == 'mps':
            params
        step_heights = torch.exp(params)


        vMins_tiled = torch.tile(vMins[:,:,None],(1,1,num_steps))
        vMins_tiled = vMins_tiled.permute(*torch.arange(vMins_tiled.ndim - 1, -1, -1,device=self.device))
        vis_tiled = torch.tile(vis[:,None,None],(1,vMins.shape[1],vMins.shape[0]))
        step_heights_tiled = torch.tile(step_heights[:,None,None],(1,vMins.shape[1],vMins.shape[0]))

        temp = vis_tiled-vMins_tiled
        heaviside = torch.where(temp > 0, 1,0)

        etas =  torch.sum(step_heights_tiled * heaviside,axis=0).T
        # #normalize
        etas*=1e-15

        return etas #same shape as vMins
