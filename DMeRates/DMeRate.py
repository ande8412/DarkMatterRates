from .Constants import *
from .DM_Halo import DM_Halo_Distributions
from .form_factor import form_factor,form_factorQEDark
import numericalunits as nu











class DMeRate:
    def __init__(self,material,QEDark=False):
        import torch
        import numpy as np
        import os
        #To assign a unit to a quantity, multiply by the unit, e.g. my_length = 100 * mm. (In normal text you would write “100 mm”, but unfortunately Python does not have “implied multiplication”.)
        #To express a dimensionful quantity in a certain unit, divide by that unit, e.g. when you see my_length / cm, you pronounce it “my_length expressed in cm”.
        #Form factor object has units already applied 
        if material == 'Si':
            if QEDark:
                form_factor_file = '../QEDark/QEdark-python/Si_f2.txt'
            else:
                form_factor_file = '../QCDark/results/Si_final.hdf5'

            self.bin_size = Sigapsize 
        if material == 'Ge':
            if QEDark:
                form_factor_file = '../QEDark/QEdark-python/Ge_f2.txt'
            else:
                form_factor_file = '../QCDark/results/Ge_final.hdf5'
  

            self.bin_size = Gegapsize

        module_dir = os.path.dirname(__file__)

        form_factor_file_filepath = os.path.join(module_dir,form_factor_file)
        if QEDark:
            ffactor = form_factorQEDark(form_factor_file_filepath)
        else:
            ffactor = form_factor(form_factor_file_filepath)



        self.v0 = v0 
        self.vEarth = vEarth 
        self.vEscape = vEscape
        self.rhoX = rhoX 
        self.cross_section = crosssection
        self.ionization_func = self.RKProbabilities
        
        self.material = material
            
            
        self.device = 'cpu'
        self.form_factor = ffactor
        nE = np.shape(self.form_factor.ff)[1]
        nQ = np.shape(self.form_factor.ff)[0]
        self.nE = nE
        self.nQ = nQ
        
        if QEDark:
            self.qiArr = torch.arange(1,nQ+1) #for indexing
            self.qArr = torch.clone(self.qiArr) * self.form_factor.dq 
            self.Earr = torch.arange(nE)*self.form_factor.dE 
        else:
            self.qiArr = torch.arange(nQ) #for indexing
            self.qArr = (torch.clone(self.qiArr) * self.form_factor.dq + self.form_factor.dq / 2.0)
            self.Earr = (torch.arange(nE)*self.form_factor.dE + self.form_factor.dE/2.0)
        self.Ei_array = torch.floor(torch.round((self.Earr/nu.eV)*10)).int() #for indexing
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX,self.cross_section)

        self.QEDark = QEDark
    
    def optimize(self,device):
        self.device = device
        self.DM_Halo.optimize(device)

    

    def update_params(self,v0,vEarth,vEscape,rhoX,crosssection):
        #assuming values passed in are km/s,km/s,km/s,eV/cm^3,cm^2
        #masses must be in eV if passed in 
        self.v0 = v0* (nu.km / nu.s)
        self.vEarth = vEarth* (nu.km / nu.s)
        self.vEscape = vEscape* (nu.km / nu.s)
        self.rhoX = rhoX* nu.eV / (nu.cm**3)
        self.cross_section = crosssection* nu.cm**2
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX)

   
    def step_probabilities(self,ne):
        import torch
        i = ne - 1
        dE, E_gap = self.form_factor.dE, self.form_factor.band_gap
        dE /=nu.eV
        E_gap /=nu.eV
        E2Q = self.bin_size / nu.eV

        initE, binE = int((E_gap)/(dE)), int(round(E2Q/dE))
        # bounds = (i*binE + initE,(i+1)*binE + initE)
        # if self.QEDark:
        bounds = (i*binE + initE + 1,(i+1)*binE + initE + 1)
        probabilities = torch.zeros_like(self.Earr)
        probabilities[bounds[0]:bounds[1]] = 1
        return probabilities

    def RKProbabilities(self,ne): #using values at 100k
        from numpy import loadtxt
        import torch
        from scipy.interpolate import interp1d
        import os
        module_dir = os.path.dirname(__file__)
        filepath = os.path.join(module_dir,'p100k.dat')
        p100data = loadtxt(filepath)
        pEV = p100data[:,0] *nu.eV
        file_probabilities = p100data.T#[:,:]
        file_probabilities = file_probabilities[ne]

        p100_func = interp1d(pEV, file_probabilities, kind = 'linear',bounds_error=False,fill_value=0)
        probabilities = p100_func(self.Earr)
        probabilities = torch.from_numpy(probabilities)
        return probabilities

    def update_crosssection(self,crosssection):
        #assuming value is in cm*2
        self.cross_section = crosssection *nu.cm**2


    def FDM(self,q_eV,n):
        me_energy = (nu.me * nu.c0**2)
        """
        DM form factor
        n = 0: FDM=1, heavy mediator
        n = 1: FDM~1/q, electric dipole
        n = 2: FDM~1/q^2, light mediator
        """
        return (nu.alphaFS*me_energy/q_eV)**n
    
    
    def get_modulated_halo_data(self,mX,FDMn,halo_model,isoangle,useVerne,calcErrors=None):
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator,PchipInterpolator,interp1d 

        import torch
        import re
        import os
        mass_string = mX / (nu.MeV) #turn into MeV
        mass_string = float(mass_string)
        from numpy import round as npround
        mass_string = npround(mass_string,3)
        if isoangle is None:
            if mass_string.is_integer():
                mass_string = int(mass_string)
            else:
                mass_string = str(mass_string)
                mass_string = mass_string.replace('.',"_")

        else:
            mass_string = str(mass_string)
            mass_string = mass_string.replace('.',"_")
        sigmaE = float(format(self.cross_section / nu.cm**2, '.3g'))
        sigmaE_str = str(sigmaE)
        # sigmaE_str.replace('.',"_")
        if FDMn == 0:
                fdm_str = 'Scr'
        else:
            fdm_str = 'LM'

        halo_prefix = '../halo_data/modulated/'
        module_dir = os.path.dirname(__file__)

        halo_dir_prefix = os.path.join(module_dir,halo_prefix) 
        if useVerne:
            dir = halo_dir_prefix + f'Verne_{fdm_str}/'
        elif halo_model =='winter':
            #note that these only work for mX = 0.6 and sigmaE = 2.1182*10^-31
            dir = halo_dir_prefix + f'December_mX_0_6_sigma_1e-30_{fdm_str}/'
        elif halo_model =='summer':
            dir = halo_dir_prefix + f'June_mX_0_6_sigma_1e-30_{fdm_str}/'
        else:
            dir = halo_dir_prefix + f'Parameter_Scan_{fdm_str}/'
        
        if 'summer' in halo_model or 'winter' in halo_model: 
            file = f'{dir}DM_Eta_theta_{isoangle}.txt'
            
        else:
            file = f'{dir}mDM_{mass_string}_MeV_sigmaE_{sigmaE_str}_cm2/DM_Eta_theta_{isoangle}.txt'
        if not os.path.isfile(file):
            print(file)
            raise FileNotFoundError('sigmaE file not found')
        
        from numpy import loadtxt
        try:
            data = loadtxt(file,delimiter='\t')
        except ValueError:
            print(file)
            raise ValueError(f'file not found! tried {file}')
        if len(data) == 0:
            raise ValueError('file is empty!')
        
        file_etas = data[:,1] * nu.s / nu.km
        file_vmins = data[:,0]* nu.km / nu.s
        if isoangle is not None:
            if calcErrors is not None:
                file_eta_err = data[:,2]* nu.s / nu.km
                if calcErrors == 'High':
                    file_etas += file_eta_err
                if calcErrors == 'Low':
                    file_etas -= file_eta_err


        #clearly this was hardcoded to catch something but don't remember what
        if file_etas[-1] == file_etas[-2]:
            file_etas = file_etas[:-1]
            file_vmins = file_vmins[:-1]
        eta_func = PchipInterpolator(file_vmins,file_etas)
        # eta_func = interp1d(file_vmins,file_etas,fill_value=0,bounds_error=False)
        vMins = self.DM_Halo.vmin_tensor(self.Earr,self.qArr,mX)


        vMin_numpy = vMins.cpu().numpy()
        etas = eta_func(vMin_numpy) # inverse velocity
        etas = torch.from_numpy(etas)
        # if self.device == 'mps':
        #     etas = etas.float()
        # etas = etas.to(self.device)
        # etas*=eta_conversion_factor #s/cm
        # etas*=ccms**2*sec2yr*self.rhoX/mX * self.cross_section#year^-1


        #make sure to avoid interpolation issues where there isn't data
        etas = torch.where((vMins<file_vmins[0]) | (vMins > file_vmins[-1]) | (torch.isnan(etas)) ,0,etas)

        
        return etas #inverse velocity



    def get_halo_from_file(self,mX,halo_model):
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator,PchipInterpolator,interp1d 
        import torch
        import os
        from numpy import round
        lightSpeed_kmpers = nu.s / nu.km #inverse to output it in units i want

        geVconversion = nu.cm**3 / nu.GeV
        halo_prefix = '../halo_data/'
        module_dir = os.path.dirname(__file__)

        halo_dir_prefix = os.path.join(module_dir,halo_prefix) 
        file = halo_dir_prefix + f'{halo_model}_v0{round(self.v0*lightSpeed_kmpers,1)}_vE{round(self.vEarth*lightSpeed_kmpers,1)}_vEsc{round(self.vEscape*lightSpeed_kmpers,1)}_rhoX{round(self.rhoX*geVconversion,1)}.txt'
        try:

            temp =open(file,'r')
            temp.close()
        except FileNotFoundError:
            self.DM_Halo.generate_halo_files(halo_model)
        # print(f'found halo file: {file}')
        from numpy import loadtxt
        try:
            data = loadtxt(file,delimiter='\t')
        except ValueError:
            print(file)
            raise ValueError(f'file not found! tried {file}')
        if len(data) == 0:
            raise ValueError('file is empty!')
        
        #default file units
        file_etas = data[:,1] * nu.s / nu.km
        file_vmins = data[:,0]* nu.km / nu.s
        

        #clearly this was hardcoded to catch something but don't remember what
        if file_etas[-1] == file_etas[-2]:
            file_etas = file_etas[:-1]
            file_vmins = file_vmins[:-1]
        eta_func = PchipInterpolator(file_vmins,file_etas)
        # eta_func = interp1d(file_vmins,file_etas,fill_value=0,bounds_error=False)
        vMins = self.DM_Halo.vmin_tensor(self.Earr,self.qArr,mX)


        vMin_numpy = vMins.cpu().numpy()
        etas = eta_func(vMin_numpy) # inverse velocity
        etas = torch.from_numpy(etas)
        # if self.device == 'mps':
        #     etas = etas.float()
        # etas = etas.to(self.device)
        # etas*=eta_conversion_factor #s/cm
        # etas*=ccms**2*sec2yr*self.rhoX/mX * self.cross_section#year^-1


        #make sure to avoid interpolation issues where there isn't data
        etas = torch.where((vMins<file_vmins[0]) | (vMins > file_vmins[-1]) | (torch.isnan(etas)) ,0,etas)

        
        return etas #inverse velocity

    def get_halo_data(self,mX,FDMn,halo_model,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None):
        import torch
        import os
        import re
        #Etas are very sensitive to numerical deviations, so leaving these units in units of c
        
        

        if halo_id_params is not None: #doing halo idp analysis
            vMins = self.DM_Halo.vmin_tensor(self.Earr,self.qArr,mX)#in velocity units
            etas = self.DM_Halo.step_function_eta(vMins, halo_id_params) 



        if isoangle is not None:
            etas = self.get_modulated_halo_data(mX,FDMn,halo_model,isoangle,useVerne,calcErrors=calcErrors)

            


        elif halo_model == 'imb':
            vMins = self.DM_Halo.vmin_tensor(self.Earr,self.qArr,mX) #in velocity units
            etas = self.DM_Halo.eta_MB_tensor(vMins) 

        
        else:
            etas = self.get_halo_from_file(mX,halo_model)
           
        return etas  #inverse velocity units

    
    
    def read_output(self,fileName):
        """Read Input File"""
        return form_factor(fileName)
    
    def reduced_mass(self,mass1,mass2):
        return mass1*mass2/(mass1+mass2)
    
    def change_to_step(self):
        self.ionization_func = self.step_probabilities
    

    def TFscreening(self,DoScreen):
        import torch
        tfdict = tf_screening[self.material]
        eps0,qTF,omegaP,alphaS = tfdict['eps0'],tfdict['qTF'],tfdict['omegaP'],tfdict['alphaS']
        Earr = self.Earr / nu.eV
        qArr = self.qArr / nu.eV
        omegaP_ = omegaP/nu.eV
        qTF_ = qTF/nu.eV
        mElectron = me_eV/nu.eV

        q_arr_tiled = torch.tile(qArr,(len(Earr),1))
        if DoScreen:
            E_array_tiled= torch.tile(Earr,(len(qArr),1)).T
            result = alphaS*((q_arr_tiled/qTF_)**2)
            result += 1.0/(eps0 - 1)
            result += q_arr_tiled**4/(4.*(mElectron**2)*(omegaP_**2))
            result -= (E_array_tiled/omegaP_)**2
            result = 1. / (1. + 1. / result)
        else:
            result = torch.ones_like(q_arr_tiled)
        return result


        
    def get_parametrized_eta(self,mX,FDMn,halo_model,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None):
        #stupid way for me to set units 
        import torch
        etas = self.get_halo_data(mX,FDMn,halo_model,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors) #inverse velocity
        #ccms**2*sec2yr

        etas*=self.rhoX/mX * self.cross_section* nu.c0**2
        etas = etas.to(torch.double)
        return etas 


    def vectorized_dRdE(self,mX,FDMn,halo_model,DoScreen=True,isoangle=None,halo_id_params=None,integrate=True,useVerne=False,calcErrors=None,debug=False,unitize=False):
        mX = mX*nu.MeV 
        import torch
        rm = self.reduced_mass(mX,me_eV)
        prefactor = nu.alphaFS * ((me_eV/rm)**2) * (1 / self.form_factor.mCell)
        if debug:
            print('alpha')
            print(nu.alphaFS)
            print('me_eV in MeV')
            print(me_eV  /(nu.MeV))
            print('mX in MeV')
            print(mX / nu.MeV)
            print('reduced mass')
            print(rm / nu.MeV)
            print('(me ev / rm) ^2')
            print(((me_eV/rm)**2))
            print('mCell')
            print(self.form_factor.mCell / nu.kg)
            # print('eprefactor')
            # print(self.form_factor.Eprefactor)
            print('prefactor')
            print(prefactor * nu.kg)

        #removed (self.rhoX / mX_eV) and * self.cross_section becuase this is in parametrized eta
        # prefactor *= nu.c0**2
        
        
        # mCell = 1/self.form_factor.mCell

        # prefactor_crys = 1/self.form_factor.mCell*alpha*me_eV**2 / self.mu_Xe(mX)**2 
        # prefactor = self.form_factor.Eprefactor *prefactor_crys

        # Ei_array = self.Ei_array.cpu().numpy()



        fdm_factor = (self.FDM(self.qArr,FDMn))**2 #unitless
        etas = self.get_parametrized_eta(mX,FDMn,halo_model,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors)
        # etas = self.get_halo_data(mX,FDMn,halo_model,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors)
        # parametrized_etas = parametrized_etas(etas)
        # etas = self.get_halo_data(mX,self.qiArr,self.Earr,FDMn,halo_model,isoangle=isoangle,halo_id_params = halo_id_params,useVerne=useVerne,calcErrors=calcError) #inverse velocity




        

            

        ff_arr = self.form_factor.ff

        if self.QEDark:
            ff_arr = ff_arr[:,self.Ei_array-1]

        ff_arr = ff_arr.T
        ff_arr = torch.from_numpy(ff_arr)
        ff_arr = ff_arr.to(self.device) #form factor (unitless)



        tf_factor = (self.TFscreening(DoScreen)**2) #unitless


        

        # if not self.QEDark:
        result = torch.einsum("i,ij->ij",self.Earr,etas)


        result *= fdm_factor     
        result*=ff_arr
        result *=tf_factor

        

        

        # else: 
        #     qdenom  = 1/(self.qArr**2) 

        # if debug:
        #     #print out variables to check
        #     print('screening factor')
        #     print(tf_factor)
        #     print('qdenom')
        #     print(qdenom)
        #     print('form factor')
        #     print(ff_arr)
        #     print('etas')
        #     print(etas)
        #     print('fdm factor')
        #     print(fdm_factor)
        #     print('prefactor')
        #     print(prefactor)
        if integrate:
            from torchquad import Simpson
            simp = Simpson()
            grid = result[:,:-1]#.to(torch.double)
            numq = len(self.qArr) 
            qmin = self.qArr[0]
            qmax = self.qArr[-1]
            integration_domain = torch.Tensor([[qmin,qmax]])
            def momentum_integrand(q):
                qdenom = 1/q**2
                return torch.einsum("j,ij->ji",qdenom.flatten(),grid)
            integrated_result = simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain) / self.Earr


                        
        else:
            qdenom = 1 / self.qArr
            result = torch.einsum("j,ij->ij",qdenom,result)
            integrated_result = (torch.sum(result,axis=1) / self.Earr)

           
            

        integrated_result *= prefactor
        
        band_gap_result = torch.where(self.Earr < self.form_factor.band_gap,0,integrated_result)
        if unitize:
            band_gap_result *= nu.year * nu.kg * nu.eV #return in correct units for comparison, otherwise keep it implicit for drdne
        if debug:  
            print("Returning:")
            print("band_gap_result,result,etas,prefactor,fdm_factor,ff_arr,tf_factor,qdenom")
            return band_gap_result,result,etas,prefactor,fdm_factor,ff_arr,tf_factor,(1/self.qArr**2)

        return band_gap_result  #result is in R / kg /year / eV




        
    
    def calculate_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        
        import torch
        import numpy
        if self.material == 'Ge':
            if self.ionization_func is not self.step_function:
                self.change_to_step()

        
        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne],device=self.device)
            elif type(ne) == list:
                nes = torch.tensor(ne,device=self.device)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne)
                nes = nes.to(self.device)
            else:
                try:
                    nes = torch.tensor(ne,device=self.device)
                except:
                    print('unknown data type')
        else:
            nes = ne
        
        #assume mX_array in MeV


        prob_fn_tiled = []
        for ne in nes:
            temp = self.ionization_func(ne)
            temp = torch.where(torch.isnan(temp),0,temp)
            prob_fn_tiled.append(temp)
        prob_fn_tiled = torch.stack(prob_fn_tiled)



        if type(mX_array) != torch.tensor:
            if type(mX_array) == int or type(mX_array) == float:
                    mX_array = torch.tensor([mX_array],device=self.device)
            elif type(mX_array) == list:
                mX_array = torch.tensor(mX_array,device=self.device)
            elif type(mX_array) == numpy.ndarray:
                mX_array = torch.from_numpy(mX_array)
                mX_array = mX_array.to(self.device)
            else:
                try:
                    mX_array = torch.tensor([mX_array],device=self.device)
                except:
                    print('unknown data type')

        dRdnEs = torch.zeros((len(mX_array),len(nes)))

        for m,mX in enumerate(mX_array):
            if debug:
                print('mX')
                print(mX)
                print('FDMn')
                print(FDMn)
                print('halo_model')
                print(halo_model)
                print('DoScreen')
                print(DoScreen)
                print('isoangle')
                print(isoangle)
                print('halo_id_params')
                print(halo_id_params)
                print('integrate')
                print(integrate)
                print('useVerne')
                print(useVerne)
                print('calcErrors')
                print(calcErrors)
            dRdE = self.vectorized_dRdE(mX,FDMn,halo_model,DoScreen=DoScreen,isoangle=isoangle,halo_id_params=halo_id_params,integrate=integrate,useVerne=useVerne,calcErrors=calcErrors) #this is in 1 /kg/year/eV , but units are still implicit


            if integrate:
                #TODO
                #maybe change this to simpsons rule too
                dRdne = torch.trapezoid(dRdE*prob_fn_tiled,x=self.Earr, axis = 1)
            else:
                # if self.QEDark:
                # dRdne = torch.sum(dRdE*prob_fn_tiled, axis = 1) * (1*nu.eV)
                # else:
                dRdne = torch.sum(dRdE*prob_fn_tiled*self.form_factor.dE*10, axis = 1)

            dRdnEs[m,:] = dRdne
        if debug:
            return dRdnEs.T,dRdE,prob_fn_tiled
        return dRdnEs.T #should be in kg/year
    
        

    

