from .Constants import *
from .DM_Halo import DM_Halo_Distributions
from .form_factor import form_factor,form_factorQEDark,formFactorNoble
import numericalunits as nu











class DMeRate:
    def __init__(self,material,QEDark=False,device=None):
        import torch
        import numpy as np
        import os
        #To assign a unit to a quantity, multiply by the unit, e.g. my_length = 100 * mm. (In normal text you would write “100 mm”, but unfortunately Python does not have “implied multiplication”.)
        #To express a dimensionful quantity in a certain unit, divide by that unit, e.g. when you see my_length / cm, you pronounce it “my_length expressed in cm”.
        #Form factor object has units already applied 

        self.module_dir = os.path.dirname(__file__)
        self.v0 = v0 
        self.vEarth = vEarth 
        self.vEscape = vEscape
        self.rhoX = rhoX 
        self.cross_section = crosssection
        self.material = material
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        if cuda_available:
            print("CUDA GPU found, performing calculations on GPU")
            self.device = 'cuda'
            self.default_dtype = torch.float64
            self.dtype_str = 'float64'
        # elif mps_available:
        #     print("MPS GPU found, performing calculations on GPU. Setting default dtype to float32.")
        #     self.device = 'mps'
        #     self.default_dtype = torch.float32
        #     self.dtype_str = 'float32'
            
        else:
            print("CUDA/MPS GPU not found, performing calculations on cpu (if you are doing this on apple silicon you can change your device to mps if you'd like)")
            self.device = 'cpu'
            self.default_dtype = torch.float64
            self.dtype_str = 'float64'

        

        if device is not None:
            print(f"You have manually specified your device to be: {device}. Overriding default")
            self.device=device

        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX,self.cross_section)

        if material == 'Si' or material == 'Ge':
            binsizes = {
                'Si': Sigapsize,
                'Ge': Gegapsize
            }
 
            if QEDark:
                form_factor_file = f'../QEDark/QEdark-python/{material}_f2.txt'
            else:
                form_factor_file = f'../QCDark/results/{material}_final.hdf5'
            self.bin_size = binsizes[material] 

        
            form_factor_file_filepath = os.path.join(self.module_dir,form_factor_file)
            if QEDark:
                ffactor = form_factorQEDark(form_factor_file_filepath)
            else:
                ffactor = form_factor(form_factor_file_filepath)

            self.ionization_func = self.RKProbabilities





        
            
            self.form_factor = ffactor
            nE = np.shape(self.form_factor.ff)[1]
            nQ = np.shape(self.form_factor.ff)[0]
            self.nE = nE
            self.nQ = nQ
            if QEDark:
                self.qiArr = torch.arange(1,nQ+1) #for indexing
                self.qArr = torch.clone(self.qiArr) * torch.tensor(self.form_factor.dq)
                self.Earr = torch.arange(nE)*torch.tensor(self.form_factor.dE )
            else:
                self.qiArr = torch.arange(nQ) #for indexing
                self.qArr = torch.clone(self.qiArr) * torch.tensor(self.form_factor.dq,dtype=torch.get_default_dtype()) + (torch.tensor(self.form_factor.dq,dtype=torch.get_default_dtype()) / torch.tensor(2.))
                self.Earr = torch.arange(nE)*torch.tensor(self.form_factor.dE,dtype=torch.get_default_dtype()) + (torch.tensor(self.form_factor.dE,dtype=torch.get_default_dtype())/ torch.tensor(2.))
            self.Ei_array = torch.floor(torch.round((self.Earr/nu.eV)*10)).int() #for indexing

            prob_fn_tiled = []
            for ne in torch.arange(1,21):
                temp = self.ionization_func(ne)
                temp = torch.where(torch.isnan(temp),0,temp)
                prob_fn_tiled.append(temp)
            prob_fn_tiled = torch.stack(prob_fn_tiled)
            self.probabilities = prob_fn_tiled

            # from scipy.interpolate import RegularGridInterpolator
            # qArr_unit = self.qArr /(nu.eV/nu.c0)
            # Earr_unit = self.Earr.cpu().numpy()/nu.eV

            # # construct an interpolator to allow for flexible number of q in integration
            # #warning, don't change this too drastically from the total number of qs, the form factor is not very smooth
            # ff_interp = RegularGridInterpolator(
            #         (qArr_unit,Earr_unit),
            #         self.form_factor.ff,
            #         bounds_error=False, fill_value=-float('inf'),)
            # numq = len(self.qArr)
            # numE = len(self.Earr)
            # ff_interp_tensor = torch.zeros((numq,numE))
            # qGrid = np.linspace(self.qArr.min(),self.qArr.max(),numq) /(nu.eV/nu.c0)
            
            # for i,E in enumerate(Earr_unit):
            #     ff_interp_tensor[:,i] = torch.tensor(ff_interp((qGrid,E)))

            # self.ff_interp_tensor = ff_interp_tensor

            self.QEDark = QEDark
        elif material == 'Xe' or material == 'Ar':
            import numpy as np
            Earr = np.geomspace(1, 400, 100) * nu.eV

            logkArr  = np.log(Earr / ry) / 2
            self.Earr = torch.tensor(Earr)
            form_factor_file = f'../wimprates_mod/data/dme/{material}_dme_ionization_ff.pkl'        
            form_factor_file_filepath = os.path.join(self.module_dir,form_factor_file)
        
            formfactor = formFactorNoble(form_factor_file_filepath)
            self.form_factor = formfactor

            # lnqiArr_dict = {}
            qArrdict = {}
            ffdata_dict  ={}
            numqs = 1001

            for shell_key in formfactor.keys:

                qmax = (np.exp(formfactor.shell_data[shell_key]['lnqs'].max())) * nu.me * nu.c0 * nu.alphaFS 
                qmin = (np.exp(formfactor.shell_data[shell_key]['lnqs'].min())) * nu.eV / nu.c0

                qArr = torch.linspace(qmin,qmax,numqs)
                lnqi = np.log(qArr.cpu() / (nu.me *nu.c0 * nu.alphaFS))
                ffdata = np.zeros((len(Earr),numqs))
                for i,k in enumerate(logkArr):
                    ffdata[i,:] = formfactor.shell_data[shell_key]['log10ffsquared_itp']((k,lnqi))
                ffdata = 10**ffdata
                ffdata_dict[shell_key] = torch.tensor(ffdata)
                qArrdict[shell_key] = qArr.to(self.device)

            self.qArrdict = qArrdict
            self.form_factor.ff = ffdata_dict

        else:
            raise ValueError('The form factors for the material you are trying to study have not been calculated.')
        
    

    

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
        # from scipy.interpolate import interp1d
        from torchinterp1d import interp1d
        import os
        filepath = os.path.join(self.module_dir,'p100k.dat')
        p100data = loadtxt(filepath)
        pEV = torch.tensor(p100data[:,0],dtype=torch.get_default_dtype()) *nu.eV
        
        file_probabilities = torch.tensor(p100data.T,dtype=torch.get_default_dtype())#[:,:]
        file_probabilities = file_probabilities[ne]

        probabilities = interp1d(pEV, file_probabilities,self.Earr).flatten()# kind = 'linear',bounds_error=False,fill_value=0)
        # probabilities = p100_func(self.Earr)
        # probabilities = torch.from_numpy(probabilities)
        # probabilities = probabilities.to(self.device)
        return probabilities

    def update_crosssection(self,crosssection):
        #assuming value is in cm*2
        self.cross_section = crosssection *nu.cm**2


    def FDM(self,q,n):
        # me_energy = (nu.me * nu.c0**2)
        """
        DM form factor
        n = 0: FDM=1, heavy mediator
        n = 1: FDM~1/q, electric dipole
        n = 2: FDM~1/q^2, light mediator
        """
        return (nu.alphaFS*  nu.me * nu.c0 /q)**n
    
    # def vmin_tensor(self,E,q,mX):
    #     import torch
    #     q_tiled = torch.tile(q,(len(E),1))
    #     EE_tiled = torch.tile(E,(len(q),1)).T
    #     vmins = ((EE_tiled/q_tiled)+(q_tiled/(2*mX)))
    #     return  vmins

    def read_output(self,fileName):
        """Read Input File"""
        return form_factor(fileName)
    
    def reduced_mass(self,mass1,mass2):
        return mass1*mass2/(mass1+mass2)
    
    def change_to_step(self):
        import torch
        self.ionization_func = self.step_probabilities
        prob_fn_tiled = []
        for ne in torch.arange(1,21):
            temp = self.ionization_func(ne)
            temp = torch.where(torch.isnan(temp),0,temp)
            prob_fn_tiled.append(temp)
        prob_fn_tiled = torch.stack(prob_fn_tiled)
        self.probabilities = prob_fn_tiled
    

    def TFscreening(self,DoScreen):
        import torch
        tfdict = tf_screening[self.material]
        eps0,qTF,omegaP,alphaS = tfdict['eps0'],tfdict['qTF'],tfdict['omegaP'],tfdict['alphaS']
        Earr = self.Earr / nu.eV
        qArr = self.qArr / (nu.eV / nu.c0)
        omegaP_ = omegaP/nu.eV
        qTF_ = qTF/nu.eV
        
        mElectron_eV = me_eV/nu.eV

        q_arr_tiled = torch.tile(qArr,(len(Earr),1))
        if DoScreen:
            E_array_tiled= torch.tile(Earr,(len(qArr),1)).T
            result = alphaS*((q_arr_tiled/qTF_)**2)
            result += 1.0/(eps0 - 1)
            result += q_arr_tiled**4/(4.*(mElectron_eV**2)*(omegaP_**2))
            result -= (E_array_tiled/omegaP_)**2
            result = 1. / (1. + 1. / result)
        else:
            result = torch.ones_like(q_arr_tiled)
        return result
    
    def thomas_fermi_screening(self,q,E,doScreen=True):
        #param q, tensor with shape 1250
        #param E, tensor with shape 500
        tfdict = tf_screening[self.material]
        eps0,qTF,omegaP,alphaS = tfdict['eps0'],tfdict['qTF'],tfdict['omegaP'],tfdict['alphaS']
        E_eV = E / nu.eV
        q_eV = q/ (nu.eV / nu.c0)
        E_eV = E_eV.unsqueeze(0)
        q_eV = q_eV.unsqueeze(1)


        omegaP_ = omegaP/nu.eV
        qTF_ = qTF/nu.eV
        
        mElectron_eV = me_eV/nu.eV
        if doScreen:

            term1 = alphaS * (q_eV / qTF_)**2  # [1249,1] * [1,1] → [1249,1]
            term2 = 1.0 / (eps0 - 1)           # Scalar → [1,1]
            term3 = q_eV**4 / (4. * (mElectron_eV**2) * (omegaP_**2))  # [1249,1]
            term4 = (E_eV / omegaP_)**2        # [1,500] → [1,500]
            
            # Explicitly broadcast all terms to [1249,500]
            result = term1.expand(-1, E_eV.shape[1]) + term2
            result += term3.expand(-1, E_eV.shape[1])
            result -= term4.expand(q_eV.shape[0], -1)
            
            result = 1. / (1. + 1. / result)



            # result = alphaS*((q_eV/qTF_)**2)
            # result += 1.0/(eps0 - 1)
            # result += q_eV**4/(4.*(mElectron_eV**2)*(omegaP_**2))
            # result -= (E_eV/omegaP_)**2
            # result = 1. / (1. + 1. / result)
        else:
            result = 1.
        return result
    
    

    def setup_halo_data(self,mX,FDMn,halo_model,isoangle=None,useVerne=False,calcErrors=None):
        import os
        import torch
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)
        if halo_model == 'imb' or halo_model == 'step':
            return
        if isoangle is None:
            lightSpeed_kmpers = nu.s / nu.km #inverse to output it in units i want

            geVconversion = 1 / (nu.GeV / nu.c0**2/ nu.cm**3)
            halo_prefix = '../halo_data/'

            halo_dir_prefix = os.path.join(self.module_dir,halo_prefix) 
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
                try:
                    self.DM_Halo.generate_halo_files(halo_model)
                except:
                    raise ValueError("Unknown halo type")
                data = loadtxt(file,delimiter='\t')
                
            if len(data) == 0:
                raise ValueError('file is empty!')
            
            #default file units
            file_etas = torch.tensor(data[:,1],dtype=torch.get_default_dtype()) * nu.s / nu.km
            file_vmins = torch.tensor(data[:,0],dtype=torch.get_default_dtype()) * nu.km / nu.s

            #clearly this was hardcoded to catch something but don't remember what
            if file_etas[-1] == file_etas[-2]:
                file_etas = file_etas[:-1]
                file_vmins = file_vmins[:-1]
        else:
            import re
            import os

            # mass_string = mX / (nu.MeV / nu.c0**2) #turn into MeV
            mass_string = float(mX)
            from numpy import round as npround
            mass_string = npround(mass_string,3)

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

            halo_dir_prefix = os.path.join(self.module_dir,halo_prefix) 
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
            
            # print(f"Using Halo Data from: {file}")
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
            
            file_etas = torch.tensor(data[:,1],dtype=self.default_dtype) * nu.s / nu.km
            file_vmins = torch.tensor(data[:,0],dtype=self.default_dtype)* nu.km / nu.s
            if isoangle is not None:
                if calcErrors is not None:
                    file_eta_err = torch.tensor(data[:,2]) * nu.s / nu.km
                    if calcErrors == 'High':
                        file_etas += file_eta_err
                    if calcErrors == 'Low':
                        file_etas -= file_eta_err


            #this was hardcoded to catch duplicates at end of verne file randomly
            if file_etas[-1] == file_etas[-2]:
                file_etas = file_etas[:-1]
                file_vmins = file_vmins[:-1]

        self.file_etas = file_etas
        self.file_vmins = file_vmins
        return


    def get_halo_data(self,vMins,halo_model,halo_id_params=None):
        import torch
        import os
        import re
        #Etas are very sensitive to numerical deviations, so leaving these units in units of c
        


        if halo_id_params is not None: #doing halo idp analysis
            etas = self.DM_Halo.step_function_eta(vMins, halo_id_params) 


        elif halo_model == 'imb':
            # vMins = self.DM_Halo.vmin_tensor(self.Earr,qArr,mX) #in velocity units
            etas = self.DM_Halo.eta_MB_tensor(vMins) 

        
        else: #from file
            from torchinterp1d import interp1d
            import torch
            file_vmins = self.file_vmins
            file_etas = self.file_etas
 
            etas = interp1d(file_vmins,file_etas,vMins) # inverse velocity
            #make sure to avoid interpolation issues where there isn't data
            etas = torch.where((vMins<file_vmins[0]) | (vMins > file_vmins[-1]) | (torch.isnan(etas)) ,0,etas)
            
        return etas  #inverse velocity units

    
    

        

    def vMin_tensor(self,qArr,Earr,mX,shell_key=None):
        import torch
        q_tiled = torch.tile(qArr,(len(Earr),1))

        EE_tiled = torch.tile(Earr,(len(qArr),1)).T
        if shell_key is not None:
            EE_tiled += binding_es[self.material][shell_key]

        vMin = ((EE_tiled/q_tiled)+(q_tiled/(2*mX)))
        

        return vMin
    
        
            # qpart = q / 2*mX
            # epart = torch.einsum('i,j->ij',E,1/q)
            # vMin = qpart + epart

            # return vMin/nu.c0

                    # result = torch.einsum("i,ij->ij",self.Earr,etas)

            #q / (2 * mX) + test_eE / qtest

            #  qtest/(2.0*test_mX * nu.MeV) + test_Ee/qtest
        
    def get_parametrized_eta(self,vMins,mX,halo_model,halo_id_params=None):
        #stupid way for me to set units 
        import torch

        etas = self.get_halo_data(vMins,halo_model,halo_id_params=halo_id_params) #inverse velocity
        #ccms**2*sec2yr
        etas*=self.rhoX/mX * self.cross_section * nu.c0**2
        # etas = etas.to(torch.double)
        return etas # inverse time






    def vectorized_dRdE(self,mX,FDMn,halo_model,DoScreen=True,halo_id_params=None,integrate=True,debug=False,unitize=False):
        import torch
        
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)

        mX = mX*nu.MeV  / nu.c0**2
        rm = self.reduced_mass(mX,nu.me)
        prefactor = nu.alphaFS * ((nu.me/rm)**2) * (1 / self.form_factor.mCell)

        ff_arr = torch.tensor(self.form_factor.ff,dtype=torch.get_default_dtype())

        if integrate:
            import torchquad
            torchquad.set_log_level('ERROR')
            from torchquad import set_up_backend
            # Enable GPU support if available and set the floating point precision
            set_up_backend("torch", data_type=self.dtype_str)

            from torchquad import Simpson
            simp = Simpson()
            numq = len(self.qArr)
            numE = len(self.Earr)
            qmin = self.qArr[0]
            qmax = self.qArr[-1]

            

            integration_domain = torch.tensor([[qmin,qmax]],dtype=torch.get_default_dtype())
            def vmin(q,E,mX):
                term1 = term1 = E.unsqueeze(0) / q.unsqueeze(1)
                term2 = q.unsqueeze(1) / (2 * mX)
                v = term1 + term2
                return  v
            def eta_func(vMin):
                return self.get_parametrized_eta(vMin,mX,halo_model,halo_id_params=halo_id_params)
            
            def momentum_integrand(q):
                q = q.flatten()
                #q only parts
                qdenom = 1/q**2
                qdenom *=(self.FDM(q,FDMn))**2
                #parts that depend on q and E
                eta = eta_func(vmin(q,self.Earr,mX))
                tf_f = (self.thomas_fermi_screening(q,self.Earr,doScreen=DoScreen))**2
                ff_f = ff_arr[:-1,:]
                # ff_f = ff_interp((q,Earr_unit))
                result = eta * tf_f * ff_f
                result = torch.einsum("i,ji->ji",self.Earr,result)
                result = torch.einsum("j,ji->ji",qdenom,result)

                return result
            integrated_result = simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain) / self.Earr

          
                        
        else:
            fdm_factor = (self.FDM(self.qArr,FDMn))**2 #unitless
            vMins = self.vMin_tensor(self.qArr,self.Earr,mX)
            etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
            if self.QEDark:
                ff_arr = ff_arr[:,self.Ei_array-1]
            ff_arr = ff_arr.T
            # ff_arr = torch.from_numpy(ff_arr)
            # ff_arr = ff_arr.to(self.device) #form factor (unitless)
            tf_factor = (self.TFscreening(DoScreen)**2) #unitless
            result = torch.einsum("i,ij->ij",self.Earr,torch.ones_like(etas))
            result *=etas
            result *= fdm_factor     
            result*=ff_arr
            result *=tf_factor
            qdenom = 1 / self.qArr
            result = torch.einsum("j,ij->ij",qdenom,result)
            integrated_result = (torch.sum(result,axis=1) / self.Earr)

           
        
        integrated_result *= prefactor
        integrated_result /=nu.c0
        
        band_gap_result = torch.where(self.Earr < self.form_factor.band_gap,0,integrated_result)
        if unitize:
            band_gap_result *= nu.year * nu.kg * nu.eV #return in correct units for comparison, otherwise keep it implicit for drdne
        if debug:  
            if integrate:
                vMins = self.vMin_tensor(self.qArr,self.Earr,mX)
                etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
                tf_factor = (self.TFscreening(DoScreen)**2) #unitless
                fdm_factor = (self.FDM(self.qArr,FDMn))**2 #unitless
                ff_arr = ff_arr.T

            returndict = {
                'drde': band_gap_result,
                'vMins': vMins / (nu.km / nu.s),
                'etas': etas * nu.year,
                'prefactor': prefactor *nu.kg,
                'fdm_factor':fdm_factor,
                'ff_arr':ff_arr,
                'tf_factor': tf_factor,
                'qdenom': (1/self.qArr**2)

            }

            return returndict

        return band_gap_result  #result is in R / kg /year / eV
    
    def calculate_crystal_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        
        import torch
        import numpy
        if self.material == 'Ge':
            if self.ionization_func is not self.step_function:
                self.change_to_step()

        
        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne])
            elif type(ne) == list:
                nes = torch.tensor(ne)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne)
                nes = nes.to(self.device)
            else:
                try:
                    nes = torch.tensor(ne)
                except:
                    print('unknown data type')
        else:
            nes = ne
        
        #assume mX_array in MeV


        prob_fn_tiled = self.probabilities[nes-1,:]



        if type(mX_array) != torch.tensor:
            if type(mX_array) == int or type(mX_array) == float:
                    mX_array = torch.tensor([mX_array])
            elif type(mX_array) == list:
                mX_array = torch.tensor(mX_array)
            elif type(mX_array) == numpy.ndarray:
                mX_array = torch.from_numpy(mX_array)
                mX_array = mX_array.to(self.device)
            else:
                try:
                    mX_array = torch.tensor([mX_array])
                except:
                    print('unknown data type')

        dRdnEs = torch.zeros((len(mX_array),len(nes)))

        for m,mX in enumerate(mX_array):
            self.setup_halo_data(mX,FDMn,halo_model,isoangle=isoangle,useVerne=useVerne,calcErrors=calcErrors)
            dRdE = self.vectorized_dRdE(mX,FDMn,halo_model,DoScreen=DoScreen,halo_id_params=halo_id_params,integrate=integrate) #this is in 1 /kg/year/eV , but units are still implicit


            if integrate:
                #TODO
                #maybe change this to simpsons rule too
                dRdne = torch.trapezoid(dRdE*prob_fn_tiled,x=self.Earr, axis = 1)
            else:
                # if self.QEDark:
                # dRdne = torch.sum(dRdE*prob_fn_tiled, axis = 1) * (1*nu.eV)
                # else:
                dRdne = torch.sum(dRdE*prob_fn_tiled*self.form_factor.dE*10, axis = 1) #still not sure why I need a factor of 10 here to match

            dRdnEs[m,:] = dRdne
        if debug:
            return dRdnEs.T,dRdE,prob_fn_tiled
        return dRdnEs.T #should be in kg/year
    
    def rate_dme_shell(self,mX,FDMn,halo_model,shell_key,halo_id_params=None,debug=False,unitize=False):
        import torch
        qArr = self.qArrdict[shell_key]
        # qiArr = self.qArrdict[shell_key] / (me_eV * nu.alphaFS)
        qmin = qArr[0]
        qmax = qArr[-1]
        numq = len(qArr)

        



        rm = self.reduced_mass(mX,nu.me)


        prefactor = 1 / (8*self.form_factor.mCell * (rm)**2 )  
        prefactor /= nu.c0**2


        #eb = get_binding_es(shell_key)
        fdm_factor = (self.FDM(qArr,FDMn))**2 #unitless
        vMins = self.vMin_tensor(qArr,self.Earr,mX,shell_key)
        etas = self.get_parametrized_eta(vMins,mX,halo_model,halo_id_params=halo_id_params)
            
        try:
            ff_arr = self.form_factor.ff[shell_key]
        except:
            print("ValueError('you have given an invalid or unimplemented shell for this material')")

        # qmax = (torch.exp(shell_data[shell_key]['lnqs'].max())) * me_eV * nu.alphaFS #this might also be a defined quantity

        result = torch.einsum("j,ij->ij",fdm_factor,etas)
        result*=ff_arr  




        import torchquad
        torchquad.set_log_level('ERROR')
        from torchquad import set_up_backend,set_precision
        # Enable GPU support if available and set the floating point precision
        set_up_backend("torch", data_type=self.dtype_str)

        from torchquad import Simpson
        simp = Simpson()
        
        integration_domain = torch.Tensor([[0,qmax]])
        if debug:
            print(integration_domain)
        def momentum_integrand(q):
            qint = q.flatten()
            # qint = qint * self.FDM(qint,FDMn) 
            return torch.einsum("j,ij->ji",qint,result)
        
        integrated_result = simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain) / self.Earr

        integrated_result *= prefactor

        if unitize:
            integrated_result *= nu.kg *nu.day * nu.keV


        if debug:
            print('returning debug output:')
            print('integrated_result,prefactor,fdm_factor,ff_arr,etas,qArr,qmin,qmax')
            return integrated_result,prefactor,fdm_factor,ff_arr,etas,qArr,qmin,qmax
        return integrated_result
    
    def noble_dRdE(self,mX,FDMn,halo_model,halo_id_params=None,debug=False,unitize=False):
        mX = mX*nu.MeV / nu.c0**2
        drs = dict()
        for key in self.form_factor.keys:
            if key in skip_keys[self.material]:
                continue
            dr = self.rate_dme_shell(mX,FDMn,halo_model,key,halo_id_params=halo_id_params,debug=False,unitize=unitize)
            drs[key] = dr
        return drs
    

    def energy_to_ne_pmf(self,rates,shell,nes_tensor,p_primary,p_secondary):
        import torch
        from torch.distributions.binomial import Binomial
        # nes_tensor = nes_tensor.to(torch.double)
        p_primary = torch.tensor(p_primary)
        fact = additional_quanta[self.material][shell]
        W = work_function[self.material]

        n_secondary = torch.floor(self.Earr / W).int() + fact  # (N,)


        binsizes = torch.tensor(torch.diff(self.Earr).tolist() + [self.Earr[-1] - self.Earr[-2]])
        # rates = rates.to(torch.float32)
        weights = rates * binsizes

        N = n_secondary.shape[0]
        M = nes_tensor.shape[0]

        r_n_tensor = torch.zeros(M)

        # Initialize PMF matrix
        pmf = torch.zeros((N, M))

        # Get unique values for efficiency
        unique_n, inverse_idx = torch.unique(n_secondary, return_inverse=True)
        for idx, n in enumerate(unique_n):
            n = int(n.item())  # Convert to Python int


            row_mask = (n_secondary == n)
            dist = Binomial(total_count=n, probs=p_secondary)

            # Determine which nes_tensor values are valid for this n
            valid_idx = nes_tensor <= n
            valid_nes = nes_tensor[valid_idx]  # Only pass valid values into log_prob

            # Compute PMF(nes)
            pmf_n = torch.zeros_like(nes_tensor)
            pmf_n[valid_idx] = torch.exp(dist.log_prob(valid_nes))

            # Compute PMF(nes - 1)
            shifted = nes_tensor - 1
            valid_shift_idx = (shifted >= 0) & (shifted <= n)
            shifted_nes = shifted[valid_shift_idx]

            pmf_n_minus_1 = torch.zeros_like(nes_tensor)
            pmf_n_minus_1[valid_shift_idx] = torch.exp(dist.log_prob(shifted_nes))

            # Combine PMFs
            weighted_pmf = p_primary * pmf_n_minus_1 + (1 - p_primary) * pmf_n

            # Assign weighted PMF to all rows with this n
            pmf[row_mask] = weighted_pmf
        # weights = weights.to(torch.double)
    
        # print(weights.type(),pmf.type())
        r_n_tensor = torch.matmul(weights, pmf)
        return r_n_tensor

    def vectorized_energy_to_ne_pmf(self, rates, shell, nes_tensor, p_primary, p_secondary):
        import torch
        
        # Convert inputs to appropriate types
        # nes_tensor = nes_tensor.to(torch.double)
        p_primary = torch.tensor(p_primary)
        p_secondary = torch.tensor(p_secondary)
        fact = additional_quanta[self.material][shell]
        W = work_function[self.material]

        # Calculate n_secondary (shape: [N])
        n_secondary = (torch.floor(self.Earr / W) + fact).int()
        
        # Calculate binsizes and weights (shape: [N])
        binsizes = torch.tensor(torch.diff(self.Earr).tolist() + [self.Earr[-1] - self.Earr[-2]])
        weights = rates * binsizes
        # weights = weights.to(torch.double)

        # Prepare for vectorized calculations
        N = n_secondary.shape[0]
        M = nes_tensor.shape[0]
        
        # Expand dimensions for broadcasting
        # nes_tensor: [M] -> [1, M]
        # n_secondary: [N] -> [N, 1]
        k = nes_tensor.unsqueeze(0)          # [1, M]
        n = n_secondary.unsqueeze(1)         # [N, 1]
        p = p_secondary
        
        # Calculate log binomial coefficients using log factorial
        # log_comb(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = (torch.lgamma(n+1) - torch.lgamma(k+1) - torch.lgamma(n-k+1))
        
        # Calculate log probabilities
        log_pmf = log_comb + k*torch.log(p) + (n-k)*torch.log(1-p)
        
        # Calculate pmf_n (set to 0 where k > n)
        pmf_n = torch.exp(log_pmf) * (k <= n).double()
        
        # Calculate pmf_n_minus_1 (for k-1)
        k_minus_1 = k - 1
        valid_shift = (k_minus_1 >= 0) & (k_minus_1 <= n)
        log_comb_shift = (torch.lgamma(n+1) - torch.lgamma(k_minus_1+1) - 
                        torch.lgamma(n-k_minus_1+1))
        log_pmf_shift = log_comb_shift + k_minus_1*torch.log(p) + (n-k_minus_1)*torch.log(1-p)
        pmf_n_minus_1 = torch.exp(log_pmf_shift) * valid_shift.double()
        
        # Combine PMFs
        weighted_pmf = p_primary * pmf_n_minus_1 + (1 - p_primary) * pmf_n
        
        # Compute final result
        r_n_tensor = torch.matmul(weights, weighted_pmf)
    
        return r_n_tensor

    


    def rates_to_ne(self,drs,nes,p_primary = 1,p_secondary = 0.83, swap_4s4p = False):
        import torch
        # We need an "energy bin size" to multiply with (or do some fancy integration)
        # I'll use the differences between the points at which the differential 
        # rates were computed.
        # To ensure this doesn't give a bias, nearby bins can't differ too much 
        # (e.g. use a linspace or a high-n logspace/geomspace)
        # import torch.distributions.binomial as binom
        
        result = dict()
        
        for shell, rates in drs.items():
            if swap_4s4p and self.material == 'Xe':
                # Somehow we can reproduce 1703.00910
                # if we swap 4s <-> 4p here??
                if shell == '4s':
                    rates = drs['4p']
                elif shell == '4p':
                    rates = drs['4s']
            result[shell] = self.vectorized_energy_to_ne_pmf(rates,shell,nes,p_primary,p_secondary)


        return result
            
                



            







    def calculate_nobleGas_rates(self,mX_array,halo_model,FDMn,ne,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False,returnShells=False):
        import torch
        import numpy

        if type(mX_array) != torch.tensor:
            if type(mX_array) == int or type(mX_array) == float:
                    mX_array = torch.tensor([mX_array])
            elif type(mX_array) == list:
                mX_array = torch.tensor(mX_array)
            elif type(mX_array) == numpy.ndarray:
                mX_array = torch.from_numpy(mX_array)
                mX_array = mX_array.to(self.device)
            else:
                try:
                    mX_array = torch.tensor([mX_array])
                except:
                    print('unknown data type')

        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne])
            elif type(ne) == list:
                nes = torch.tensor(ne)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne)
                nes = nes.to(self.device)
            else:
                try:
                    nes = torch.tensor(ne)
                except:
                    print('unknown data type')
        else:
            nes = ne

        if returnShells:        

            keys = self.form_factor.keys
            skpkeys = skip_keys[self.material]
            totalkeys = [k for k in keys if k not in skpkeys]
            numkeys = len(totalkeys)
            shells = ['Summed'] + totalkeys
            dRdnEs = torch.zeros((len(mX_array),len(nes),numkeys + 1))
        else:        
            dRdnEs = torch.zeros((len(mX_array),len(nes)))


        
        for m,mX in enumerate(mX_array):
            self.setup_halo_data(mX,FDMn,halo_model,isoangle=isoangle,useVerne=useVerne,calcErrors=calcErrors)
            drs = self.noble_dRdE(mX,FDMn,halo_model,halo_id_params=halo_id_params,debug=False,unitize=False)
            dRdnEs_by_shell =  self.rates_to_ne(drs,nes,p_primary = 1,p_secondary = 0.83, swap_4s4p = True)
            dRdnEs_by_shell = torch.stack(list(dRdnEs_by_shell.values()))
            dRdnE_sum = torch.sum(dRdnEs_by_shell,axis=0)
            if returnShells:
                dRdnEs[m,:,0] = dRdnE_sum
                dRdnEs[m,:,1:] = dRdnEs_by_shell.T


            else:
                dRdnEs[m,:] = dRdnE_sum

        
        if returnShells:
            return dRdnEs,shells
        return dRdnEs.T











    def calculate_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False):
        if self.material == 'Si' or self.material =='Ge':
            return self.calculate_crystal_rates(mX_array,halo_model,FDMn,ne,integrate,DoScreen,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors,debug=debug)
        if self.material == 'Xe' or self.material == 'Ar':
            return self.calculate_nobleGas_rates(mX_array,halo_model,FDMn,ne,isoangle=isoangle,halo_id_params=halo_id_params,useVerne=useVerne,calcErrors=calcErrors,debug=debug,returnShells=False)
    
    
    

    def generate_dat(self,dm_masses,ne_bins,fdm,dm_halo_model,DoScreen=False,write=True,tag=""):
        import torch
        from tqdm.autonotebook import tqdm
        import numpy as np
        function_name = 'p100k' if self.ionization_func == self.RKProbabilities else 'step'
        fdm_dict = {0:'1',1:'q',2:'q2'}
        rho_X =self.rhoX / (nu.GeV / nu.c0**2 / nu.cm**3)
        vE = self.vEarth  / (nu.km / nu.s)
        vesc = self.vEscape    / (nu.km / nu.s)
        v0 = self.v0 / (nu.km / nu.s)
        rho_X_print = str(np.round(rho_X,1))
        ebin_print = str(self.bin_size)
        vesc_print = str(np.round(vesc,1))
        v0_print = str(np.round(v0,1))
        vE_print = str(np.round(vE,1))


        ebin_print = ebin_print.replace('.','pt')
        rho_X_print = rho_X_print.replace('.','pt')
        vesc_print = vesc_print.replace('.','pt')
        v0_print = v0_print.replace('.','pt')
        vE_print = vE_print.replace('.','pt')


        import os
        FDM_Dir = os.path.join(self.module_dir,'Rates/')
        screen = '_screened' if DoScreen else ''
        integrate = True
        if self.material == 'Si':
            qestr = '_qedark' if self.QEDark else '_qcdark'
            integrate = False if self.QEDark else True

        
      
        filename = FDM_Dir + f'FDM{fdm_dict[fdm]}_vesc{vesc_print}-v0{v0_print}-vE{vE_print}-rhoX{rho_X_print}_nevents_func{function_name}_maxne{np.max(ne_bins)}_unscaled{screen}{qestr}_dmerates_{tag}.dat'
        
        lines = []
        data = np.zeros((len(dm_masses),len(ne_bins)))
        for m in tqdm(range(len(dm_masses))):
            mX = dm_masses[m]
            line = f'{mX}\t'
            gday = self.calculate_rates(mX,dm_halo_model,fdm,ne_bins,integrate=integrate,DoScreen=DoScreen).T * nu.g * nu.day
            gday = gday[0]
            gday = torch.where(torch.isnan(gday),0,gday)
            g_day = gday.numpy()
            for ne in ne_bins:
                data[m,ne - 1] = g_day[ne-1]
                line+=str(g_day[ne-1])+'\t'
                
            line += f'\n'
            lines.append(line)
        if write: 
            f = open(filename,'w')
            first_line = f'FDM{fdm_dict[fdm]}:\tmX [MeV]\tsigmae={self.cross_section / nu.cm**2} [cm^2]\t'
            for ne in ne_bins:
                first_line += f'ne{ne}\t'
            first_line+='\n'
            f.write(first_line)
            for line in lines:
                f.write(line)
        return data

        

    

