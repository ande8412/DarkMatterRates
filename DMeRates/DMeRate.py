from .Constants import *
from .DM_Halo import DM_Halo_Distributions
from .engines.dielectric import compute_dRdE as compute_qcdark2_dRdE
from .engines.form_factor import (
    semiconductor_dRdE_spectrum as compute_form_factor_spectrum,
    semiconductor_rates_from_spectrum,
)
from .engines.noble_gas import noble_srdm_dRdE_spectrum
from .form_factor import form_factor,form_factorQEDark,formFactorNoble
from .halo.analytic import AnalyticHaloProvider
from .halo.file_loader import FileHaloProvider, load_halo_file_data
from .halo.independent import HaloIndependentProvider
from .data.registry import DataRegistry
from .ionization.rk_probabilities import rk_probabilities
from .responses.dielectric import dielectric_response
from .responses.dielectric_materials import (
    QCDARK2_BANDGAPS,
    canonical_qcdark2_material,
    require_qcdark2_pair_energy,
)
from .ionization.step_function import step_probabilities
from .screening.lindhard import DEFAULT_LINDHARD_ETA_EV, lindhard_screening
from .screening.semiconductor import (
    semiconductor_native_screening_factor,
    semiconductor_screening_factor,
)
from .screening.thomas_fermi import thomas_fermi_screening, tfscreening
import numericalunits as nu
import numpy as np




_SEMICONDUCTOR_MATERIALS = ('Si', 'Ge', 'GaAs', 'SiC', 'Diamond', 'diamond')
_NOBLE_MATERIALS = ('Xe', 'Ar')
_SRDM_HALO_MODELS = {'srdm', 'srdm_modulated'}
_SRDMBEAM_SOURCE = 'SRDMBeam'
_SRDMBEAM_MODULATED_SOURCES = {'SRDMBeam', 'Verne', 'DaMaSCUS'}
_ORDINARY_MODULATED_HALO_MODELS = {'modulated', 'summer'}


def _validate_srdm_modulated_call(halo_model, isoangle, modulated_source):
    """Validate SRDMBeam-only public kwargs without touching other halo paths."""
    if halo_model == 'srdm_modulated':
        if modulated_source is None:
            modulated_source = _SRDMBEAM_SOURCE
        from .srdm.flux_loader import normalize_srdmbeam_modulated_source
        modulated_source = normalize_srdmbeam_modulated_source(modulated_source)
        if modulated_source not in _SRDMBEAM_MODULATED_SOURCES:
            raise ValueError(
                f"Unsupported modulated_source={modulated_source!r} for "
                f"halo_model='srdm_modulated'; expected one of "
                f"{sorted(_SRDMBEAM_MODULATED_SOURCES)}"
            )
        if isoangle is None:
            raise ValueError(
                "halo_model='srdm_modulated' requires an explicit integer "
                "isoangle/ring_index"
            )
        if not isinstance(isoangle, (int, np.integer)):
            raise ValueError(
                f"halo_model='srdm_modulated' isoangle/ring_index must be an integer, "
                f"got {isoangle!r}"
            )
        return modulated_source

    if modulated_source is not None:
        raise ValueError(
            "modulated_source is only supported with halo_model='srdm_modulated'"
        )
    return None


def _normalize_ordinary_modulation_source(modulation_source):
    key = str(modulation_source).strip().lower().replace('-', '_')
    if key == 'verne':
        return 'Verne'
    if key in ('damascus', 'damascus_shm'):
        return 'DaMaSCUS'
    raise ValueError(
        f"Unsupported modulation_source={modulation_source!r}; expected "
        "'Verne' or 'DaMaSCUS'"
    )


def _resolve_ordinary_modulation_source(halo_model, useVerne, modulation_source):
    """Resolve explicit ordinary halo-modulation source while preserving useVerne."""
    if modulation_source is None:
        return useVerne
    if halo_model not in _ORDINARY_MODULATED_HALO_MODELS:
        raise ValueError(
            "modulation_source is only supported with ordinary "
            "halo_model='modulated' or halo_model='summer'. For SRDMBeam use "
            "halo_model='srdm_modulated' with modulated_source='Verne' or "
            "'DaMaSCUS'."
        )
    return _normalize_ordinary_modulation_source(modulation_source) == 'Verne'






class DMeRate:
    """Class for calculating dark matter-electron scattering rates in various materials.
    
    Provides methods for computing DM-electron interaction rates in:
    - Semiconductor crystals (Si, Ge)
    - Noble elements (Xe, Ar)
    
    Args:
        material (str): Target material ('Si', 'Ge', 'Xe', or 'Ar')
        QEDark (bool): Use QEDark form factors (default False)
        device (str, optional): Computation device ('cpu', 'cuda', etc.)
    """
    def __init__(self,material,form_factor_type=None,device=None):
        """Initialize rate calculator with material properties and computation settings."""

        import torch
        import numpy as np
        import os
        #To assign a unit to a quantity, multiply by the unit, e.g. my_length = 100 * mm. (In normal text you would write “100 mm”, but unfortunately Python does not have “implied multiplication”.)
        #To express a dimensionful quantity in a certain unit, divide by that unit, e.g. when you see my_length / cm, you pronounce it “my_length expressed in cm”.
        #Form factor object has units already applied 

        if form_factor_type is not None and str(form_factor_type).lower() == 'qcdark2':
            material = canonical_qcdark2_material(material)

        if material in _NOBLE_MATERIALS:
            if form_factor_type not in (None, 'wimprates'):
                raise ValueError(
                    f"Noble gas material '{material}' uses form_factor_type='wimprates'."
                )
            form_factor_type = 'wimprates'
        elif form_factor_type is None:
            form_factor_type = 'qcdark' #default to qcdark
            
        self.form_factor_type = form_factor_type.lower()
        self.QEDark = True if self.form_factor_type == 'qedark' else False

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
        # elif mps_available: commented out since mps is limited to 32 floats, which can cause issues
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

        if self.form_factor_type == 'qcdark2':
            if material not in _SEMICONDUCTOR_MATERIALS:
                raise ValueError(
                    f"QCDark2 backend is only supported for semiconductor targets: "
                    f"{sorted(QCDARK2_BANDGAPS.keys())}"
                )

            material = canonical_qcdark2_material(material)
            self.material = material
            self.qcdark2_variant = 'composite'
            self.form_factor = dielectric_response(material, variant=self.qcdark2_variant)
            # Keep legacy field names where practical.
            self.form_factor.band_gap = QCDARK2_BANDGAPS[material]
            self.Earr = torch.as_tensor(self.form_factor.E, dtype=torch.get_default_dtype())
            self.nE = len(self.Earr)
            self.nQ = len(self.form_factor.q_ame)
            self.qArr = torch.as_tensor(self.form_factor.q, dtype=torch.get_default_dtype())
            self.qiArr = torch.arange(self.nQ)
            self.Ei_array = torch.floor(torch.round((self.Earr/nu.eV)*10)).int()

            # QCDark2 ne-policy:
            # - Si may use RK (default).
            # - Ge uses legacy step approximation by default.
            # - GaAs/SiC/Diamond require explicit pair_energy at call time.
            if self.material == 'Si':
                self.bin_size = Sigapsize
                self.ionization_func = self.RKProbabilities
            elif self.material == 'Ge':
                self.bin_size = Gegapsize
                self.ionization_func = self.step_probabilities
            else:
                self.bin_size = None
                self.ionization_func = None

            if self.ionization_func is not None:
                prob_fn_tiled = []
                for ne in torch.arange(1,21):
                    temp = self.ionization_func(ne)
                    temp = torch.where(torch.isnan(temp),0,temp)
                    prob_fn_tiled.append(temp)
                self.probabilities = torch.stack(prob_fn_tiled)
            else:
                self.probabilities = None

        elif material == 'Si' or material == 'Ge':
            binsizes = {
                'Si': Sigapsize,
                'Ge': Gegapsize
            }
 
            self.bin_size = binsizes[material]

            if self.QEDark:
                form_factor_file_filepath = str(DataRegistry.qedark_ff(material))
                ffactor = form_factorQEDark(form_factor_file_filepath)
            else:
                form_factor_file_filepath = str(DataRegistry.qcdark1_ff(material))
                ffactor = form_factor(form_factor_file_filepath)

            self.ionization_func = self.RKProbabilities





        
            
            self.form_factor = ffactor
            nE = np.shape(self.form_factor.ff)[1]
            nQ = np.shape(self.form_factor.ff)[0]
            self.nE = nE
            self.nQ = nQ
            if self.QEDark:
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

            # self.QEDark = QEDark
        elif material == 'Xe' or material == 'Ar':
            import numpy as np
            Earr = np.geomspace(1, 400, 100) * nu.eV

            logkArr  = np.log(Earr / ry) / 2
            self.Earr = torch.tensor(Earr)
            form_factor_file_filepath = str(DataRegistry.noble_ff(material))
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
        """Update DM halo parameters.
        
        Args:
            v0 (float): Most probable DM velocity (km/s)
            vEarth (float): Earth's velocity (km/s)
            vEscape (float): Galactic escape velocity (km/s)
            rhoX (float): Local DM density (eV/cm^3)
            crosssection (float): DM cross section (cm^2)
        """
        #assuming values passed in are km/s,km/s,km/s,eV/cm^3,cm^2
        #masses must be in eV if passed in 
        self.v0 = v0* (nu.km / nu.s)
        self.vEarth = vEarth* (nu.km / nu.s)
        self.vEscape = vEscape* (nu.km / nu.s)
        self.rhoX = rhoX* nu.eV / (nu.cm**3)
        self.cross_section = crosssection* nu.cm**2
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX)

   
    def step_probabilities(self,ne):
        """Step function approximation for electron-hole pair creation probabilities.
        
        Args:
            ne (int): Number of electron-hole pairs
            
        Returns:
            probabilities for being in certain bin
        """
        return step_probabilities(
            ne=ne,
            energy_array=self.Earr,
            dE=self.form_factor.dE,
            band_gap=self.form_factor.band_gap,
            bin_size=self.bin_size,
        )

    def RKProbabilities(self,ne): #using values at 100k
        """Interpolated probabilities from Ramanathan Kurinsky data for electron-hole pair creation.
           See https://arxiv.org/abs/2004.10709
        Args:
            ne (int): Number of electron-hole pairs
            
        Returns:
            Array of probabilities for each energy bin
        """
        import torch

        return rk_probabilities(
            ne=ne,
            energy_array=self.Earr,
            p100k_path=DataRegistry.p100K_dat,
            dtype=torch.get_default_dtype(),
        )

    def update_crosssection(self,crosssection):
        """Update DM-electron cross section.
        
        Args:
            crosssection (float): New cross section in cm^2
        """
        #assuming value is in cm*2
        self.cross_section = crosssection *nu.cm**2


    def FDM(self,q,n):
        """Dark matter form factor calculation.
        
        Args:
            q: Momentum transfer
            n: Form factor model (0=FDM1, 1=FDMq, 2=FDMq2)
            
        Returns:
            Form factor value
        """
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
        """Read form factor data from file.
        
        Args:
            fileName: Path to form factor file
            
        Returns:
            Form factor object
        """
        """Read Input File"""
        return form_factor(fileName)
    
    def reduced_mass(self,mass1,mass2):
        """Calculate reduced mass of two particles.
        
        Args:
            mass1: First particle mass
            mass2: Second particle mass
            
        Returns:
            Reduced mass
        """
        return mass1*mass2/(mass1+mass2)
    
    def change_to_step(self):
        """Switch to step function probability model."""
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
        """Thomas-Fermi screening calculation for semiconductors.
        
        Args:
            DoScreen (bool): Whether to apply screening
            
        Returns:
            Screening factor array
        """
        return tfscreening(
            material=self.material,
            Earr=self.Earr,
            qArr=self.qArr,
            do_screen=DoScreen,
        )

    def semiconductor_native_screening(self, DoScreen, screening=None, lindhard_eta_eV=DEFAULT_LINDHARD_ETA_EV):
        """Selected QEDark/QCDark1 screening on the native E x q grid."""
        return semiconductor_native_screening_factor(
            material=self.material,
            Earr=self.Earr,
            qArr=self.qArr,
            screening=screening,
            do_screen=DoScreen,
            lindhard_eta_eV=lindhard_eta_eV,
        )
    
    def thomas_fermi_screening(self,q,E,doScreen=True):
        """Vectorized Thomas-Fermi screening calculation.
        
        Args:
            q: Momentum transfer values
            E: Energy values
            doScreen: Whether to apply screening
            
        Returns:
            Screening factor tensor
        """
        return thomas_fermi_screening(
            material=self.material,
            q=q,
            E=E,
            do_screen=doScreen,
        )

    def lindhard_screening(self, q, E, doScreen=True, eta_eV=DEFAULT_LINDHARD_ETA_EV):
        """Vectorized Lindhard screening factor, 1 / |epsilon_L(q, E)|."""
        return lindhard_screening(
            material=self.material,
            q=q,
            E=E,
            do_screen=doScreen,
            eta_eV=eta_eV,
        )

    def semiconductor_screening(self, q, E, doScreen=True, screening=None, lindhard_eta_eV=DEFAULT_LINDHARD_ETA_EV):
        """Selected QEDark/QCDark1 screening on a q x E grid."""
        return semiconductor_screening_factor(
            material=self.material,
            q=q,
            E=E,
            screening=screening,
            do_screen=doScreen,
            lindhard_eta_eV=lindhard_eta_eV,
        )
    
    

    def setup_halo_data(self,mX,FDMn,halo_model,isoangle=None,useVerne=False,calcErrors=None):
        """Load or generate velocity distribution data for given parameters.
        
        Args:
            mX: DM mass
            FDMn: Form factor model [0 or 2]
            halo_model: Velocity distribution model ('shm' for standard halo model, 'modulated' for march data, 'summer' for june data)
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low') (only works for DaMaSCUS data)
        """
        import torch
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)
        if halo_model == 'imb' or halo_model == 'step':
            return
        file_vmins, file_etas = load_halo_file_data(
            module_dir=self.module_dir,
            dm_halo=self.DM_Halo,
            halo_model=halo_model,
            v0=self.v0,
            vEarth=self.vEarth,
            vEscape=self.vEscape,
            rhoX=self.rhoX,
            cross_section=self.cross_section,
            mX=mX,
            FDMn=FDMn,
            isoangle=isoangle,
            useVerne=useVerne,
            calcErrors=calcErrors,
            default_dtype=self.default_dtype,
        )
        self.file_etas = file_etas
        self.file_vmins = file_vmins
        return


    def get_halo_data(self,vMins,halo_model,halo_id_params=None):
        """Retrieve velocity distribution for given velocities.
        
        Args:
            vMins: Minimum velocity values
            halo_model: Velocity distribution model (shm, tsa, etc. See DM_Halo.py)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            
        Returns:
            Integrated velocity distribution values
        """
        #Etas are very sensitive to numerical deviations, so leaving these units in units of c
        if halo_id_params is not None: #doing halo idp analysis
            provider = HaloIndependentProvider(self.DM_Halo, halo_id_params)
            return provider.eta(vMins)

        if halo_model == 'imb':
            provider = AnalyticHaloProvider(self.DM_Halo, "imb")
            return provider.eta(vMins)

        provider = FileHaloProvider(self.file_vmins, self.file_etas)
        return provider.eta(vMins)  #inverse velocity units

    
    

        

    def vMin_tensor(self,qArr,Earr,mX,shell_key=None):
        """Calculate minimum velocities in tensor form.
        
        Args:
            qArr: Momentum transfer values
            Earr: Energy values
            mX: DM mass
            shell_key: Electron shell (for noble elements)
            
        Returns:
            Tensor of minimum velocities
        """
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
        """Calculate parametrized velocity distribution.
        
        Args:
            vMins: Minimum velocity values
            mX: DM mass
            halo_model: Velocity distribution model
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            
        Returns:
            Parametrized integrated velocity distribution values (units of inverse time)
        """
        #stupid way for me to set units 
        import torch

        etas = self.get_halo_data(vMins,halo_model,halo_id_params=halo_id_params) #inverse velocity
        #ccms**2*sec2yr
        etas*=self.rhoX/mX * self.cross_section * nu.c0**2
        # etas = etas.to(torch.double)
        return etas # inverse time






    def vectorized_dRdE(self,mX,FDMn,halo_model,DoScreen=True,halo_id_params=None,integrate=True,debug=False,unitize=False,screening=None,lindhard_eta_eV=DEFAULT_LINDHARD_ETA_EV):
        """Calculate differential rate (dR/dE) using vectorized operations for semiconductor materials.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            DoScreen: Apply screening effects
            screening: Explicit semiconductor screening mode; overrides DoScreen when set
            lindhard_eta_eV: Lindhard imaginary broadening in eV
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            integrate: Perform numerical integration
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Differential rate values
        """
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
                tf_f = (self.semiconductor_screening(
                    q,
                    self.Earr,
                    doScreen=DoScreen,
                    screening=screening,
                    lindhard_eta_eV=lindhard_eta_eV,
                ))**2
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
            tf_factor = (self.semiconductor_native_screening(
                DoScreen,
                screening=screening,
                lindhard_eta_eV=lindhard_eta_eV,
            )**2) #unitless
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
                tf_factor = (self.semiconductor_native_screening(
                    DoScreen,
                    screening=screening,
                    lindhard_eta_eV=lindhard_eta_eV,
                )**2) #unitless
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

    def _qcdark2_pair_energy_to_nu(self, pair_energy):
        """Normalize pair energy input to numericalunits eV."""
        if pair_energy is None:
            return None
        try:
            # Accept both plain eV scalars and already-unitized values.
            value_eV = float(pair_energy / nu.eV)
            return value_eV * nu.eV
        except Exception:
            return float(pair_energy) * nu.eV

    def _qcdark2_probabilities(self, nes, pair_energy=None):
        """Construct ne probability rows for the QCDark2 energy grid."""
        import torch

        material = canonical_qcdark2_material(self.material)
        pair_energy_nu = self._qcdark2_pair_energy_to_nu(pair_energy)

        if material == 'Si':
            rows = []
            for ne in nes:
                probs = rk_probabilities(
                    ne=int(ne),
                    energy_array=self.Earr,
                    p100k_path=DataRegistry.p100K_dat,
                    dtype=torch.get_default_dtype(),
                )
                rows.append(torch.where(torch.isnan(probs), 0, probs))
            return torch.stack(rows)

        if material == 'Ge':
            if pair_energy_nu is None:
                pair_energy_nu = Gegapsize
            rows = [
                step_probabilities(
                    ne=int(ne),
                    energy_array=self.Earr,
                    dE=self.form_factor.dE,
                    band_gap=QCDARK2_BANDGAPS[material],
                    bin_size=pair_energy_nu,
                )
                for ne in nes
            ]
            return torch.stack(rows)

        require_qcdark2_pair_energy(material, pair_energy_nu)
        rows = [
            step_probabilities(
                ne=int(ne),
                energy_array=self.Earr,
                dE=self.form_factor.dE,
                band_gap=QCDARK2_BANDGAPS[material],
                bin_size=pair_energy_nu,
            )
            for ne in nes
        ]
        return torch.stack(rows)

    def _qcdark2_dielectric_for_variant(self, variant):
        """Return a cached QCDark2 dielectric response for the requested variant."""
        import torch

        if getattr(self, "qcdark2_variant", None) != variant:
            self.form_factor = dielectric_response(self.material, variant=variant)
            self.qcdark2_variant = variant
            self.form_factor.band_gap = QCDARK2_BANDGAPS[self.material]
            self.Earr = torch.as_tensor(
                self.form_factor.E,
                dtype=torch.get_default_dtype(),
            )
            self.nE = len(self.Earr)
            self.nQ = len(self.form_factor.q_ame)
            self.qArr = torch.as_tensor(
                self.form_factor.q,
                dtype=torch.get_default_dtype(),
            )
            self.qiArr = torch.arange(self.nQ)
            self.Ei_array = torch.floor(torch.round((self.Earr / nu.eV) * 10)).int()
        return self.form_factor

    def calculate_qcdark2_rates(
        self,
        mX_array,
        halo_model,
        FDMn,
        ne,
        screening=None,
        variant=None,
        pair_energy=None,
        isoangle=None,
        halo_id_params=None,
        useVerne=False,
        calcErrors=None,
        debug=False,
        mediator_spin='vector',
        sigma_e=None,
        modulated_source=None,
        srdm_base_data_dir=None,
        modulation_source=None,
    ):
        """Calculate QCDark2 ne rates by integrating the native dielectric dR/dE."""
        import numpy
        import torch
        useVerne = _resolve_ordinary_modulation_source(
            halo_model,
            useVerne,
            modulation_source,
        )
        modulated_source = _validate_srdm_modulated_call(halo_model, isoangle, modulated_source)
        if halo_model in _SRDM_HALO_MODELS:
            from .srdm.mediators import normalize_mediator_spin
            mediator_spin = normalize_mediator_spin(mediator_spin)

        if screening is None:
            raise ValueError(
                "QCDark2 calculations require an explicit screening choice. "
                "Pass screening='rpa' or screening='none'."
            )
        if variant is None:
            variant = getattr(self, 'qcdark2_variant', 'composite')

        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne])
            elif type(ne) == list:
                nes = torch.tensor(ne)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne).to(self.device)
            else:
                nes = torch.tensor(ne)
        else:
            nes = ne

        if type(mX_array) != torch.Tensor:
            if type(mX_array) == int or type(mX_array) == float:
                mX_array = torch.tensor([mX_array])
            elif type(mX_array) == list:
                mX_array = torch.tensor(mX_array)
            elif type(mX_array) == numpy.ndarray:
                mX_array = torch.from_numpy(mX_array).to(self.device)
            else:
                mX_array = torch.tensor([mX_array])

        rhoX_eV_per_cm3 = float(self.rhoX * nu.c0**2 / (nu.eV / nu.cm**3))
        sigma_e_cm2 = float(self.cross_section / nu.cm**2) if sigma_e is None else float(sigma_e)
        dielectric = self._qcdark2_dielectric_for_variant(variant)
        prob_fn_tiled = self._qcdark2_probabilities(nes, pair_energy=pair_energy)
        dRdnEs = torch.zeros((len(mX_array), len(nes)))

        debug_last_dRdE = None
        for m, mX in enumerate(mX_array):
            eta_provider = None
            if halo_id_params is not None:
                eta_provider = lambda vmins, params=halo_id_params: self.get_halo_data(
                    vmins, halo_model, halo_id_params=params
                )
            elif halo_model not in {"imb", "srdm", "srdm_modulated"}:
                self.setup_halo_data(
                    mX,
                    FDMn,
                    halo_model,
                    isoangle=isoangle,
                    useVerne=useVerne,
                    calcErrors=calcErrors,
                )
                eta_provider = lambda vmins: self.get_halo_data(vmins, halo_model)

            result = compute_qcdark2_dRdE(
                material=self.material,
                mX_eV=float(mX) * 1e6,  # input mX is MeV in legacy API
                FDMn=FDMn,
                halo_model=halo_model,
                screening=screening,
                variant=variant,
                sigma_e_cm2=sigma_e_cm2,
                mediator_spin=mediator_spin,
                ring_index=isoangle,
                modulated_source=modulated_source,
                srdm_base_data_dir=srdm_base_data_dir,
                rhoX_eV_per_cm3=rhoX_eV_per_cm3,
                halo_distribution=self.DM_Halo,
                eta_provider=eta_provider,
                dielectric=dielectric,
            )
            spectrum = result.spectrum
            self.Earr = spectrum.E
            self.form_factor.dE = float(torch.diff(self.Earr).mean() / nu.eV) * nu.eV

            dRdne = torch.trapezoid(spectrum.dR_dE * prob_fn_tiled, x=spectrum.E, axis=1)
            dRdnEs[m, :] = dRdne
            debug_last_dRdE = spectrum.dR_dE

        if debug:
            return dRdnEs.T, debug_last_dRdE, prob_fn_tiled
        return dRdnEs.T

    def calculate_qcdark2_spectrum(
        self,
        mX,
        halo_model,
        FDMn,
        screening=None,
        variant=None,
        isoangle=None,
        halo_id_params=None,
        useVerne=False,
        calcErrors=None,
        mediator_spin='vector',
        sigma_e=None,
        modulated_source=None,
        srdm_base_data_dir=None,
    ):
        """Return native QCDark2 dR/dE spectrum for a single mass point."""
        modulated_source = _validate_srdm_modulated_call(halo_model, isoangle, modulated_source)
        if halo_model in _SRDM_HALO_MODELS:
            from .srdm.mediators import normalize_mediator_spin
            mediator_spin = normalize_mediator_spin(mediator_spin)
        if screening is None:
            raise ValueError(
                "QCDark2 calculations require an explicit screening choice. "
                "Pass screening='rpa' or screening='none'."
            )
        if variant is None:
            variant = getattr(self, 'qcdark2_variant', 'composite')
        rhoX_eV_per_cm3 = float(self.rhoX * nu.c0**2 / (nu.eV / nu.cm**3))
        sigma_e_cm2 = float(self.cross_section / nu.cm**2) if sigma_e is None else float(sigma_e)
        dielectric = self._qcdark2_dielectric_for_variant(variant)
        eta_provider = None
        if halo_id_params is not None:
            eta_provider = lambda vmins, params=halo_id_params: self.get_halo_data(
                vmins, halo_model, halo_id_params=params
            )
        elif halo_model not in {"imb", "srdm", "srdm_modulated"}:
            self.setup_halo_data(
                mX,
                FDMn,
                halo_model,
                isoangle=isoangle,
                useVerne=useVerne,
                calcErrors=calcErrors,
            )
            eta_provider = lambda vmins: self.get_halo_data(vmins, halo_model)

        return compute_qcdark2_dRdE(
            material=self.material,
            mX_eV=float(mX) * 1e6,
            FDMn=FDMn,
            halo_model=halo_model,
            screening=screening,
            variant=variant,
            sigma_e_cm2=sigma_e_cm2,
            mediator_spin=mediator_spin,
            ring_index=isoangle,
            modulated_source=modulated_source,
            srdm_base_data_dir=srdm_base_data_dir,
            rhoX_eV_per_cm3=rhoX_eV_per_cm3,
            halo_distribution=self.DM_Halo,
            eta_provider=eta_provider,
            dielectric=dielectric,
        ).spectrum

    def qcdark2_above_threshold_rate(self, mX, halo_model, FDMn, screening=None, variant=None, threshold_eV=None):
        """Integrate the QCDark2 dR/dE spectrum above an energy threshold."""
        import torch

        spectrum = self.calculate_qcdark2_spectrum(
            mX=mX,
            halo_model=halo_model,
            FDMn=FDMn,
            screening=screening,
            variant=variant,
        )
        if threshold_eV is None:
            threshold = QCDARK2_BANDGAPS[canonical_qcdark2_material(self.material)]
        else:
            threshold = self._qcdark2_pair_energy_to_nu(threshold_eV)
        mask = spectrum.E >= threshold
        if not torch.any(mask):
            return 0.0 * (1 / (nu.kg * nu.year))
        return torch.trapezoid(spectrum.dR_dE[mask], x=spectrum.E[mask], axis=0)
    
    def calculate_semiconductor_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False,screening=None,variant=None,pair_energy=None,mediator_spin='vector',sigma_e=None,lindhard_eta_eV=DEFAULT_LINDHARD_ETA_EV,modulated_source=None,srdm_base_data_dir=None,modulation_source=None):
        """Calculate rates for semiconductor crystals (Si, Ge).
        
        Args:
            mX_array: Array of DM masses
            halo_model: Velocity distribution model
            FDMn: Form factor model
            ne: Number of electron-hole pairs
            integrate: Perform numerical integration vs using preintegrated version
            DoScreen: Apply screening effects
            screening: Explicit screening mode for QEDark/QCDark1 or QCDark2
            lindhard_eta_eV: Lindhard imaginary broadening in eV
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low') (only for DaMaSCUS)
            debug: Return debug information
            
        Returns:
            Calculated rates
        """
        import torch
        import numpy
        useVerne = _resolve_ordinary_modulation_source(
            halo_model,
            useVerne,
            modulation_source,
        )
        modulated_source = _validate_srdm_modulated_call(halo_model, isoangle, modulated_source)
        if halo_model in _SRDM_HALO_MODELS:
            from .srdm.mediators import normalize_mediator_spin
            mediator_spin = normalize_mediator_spin(mediator_spin)
        if self.form_factor_type == 'qcdark2':
            return self.calculate_qcdark2_rates(
                mX_array=mX_array,
                halo_model=halo_model,
                FDMn=FDMn,
                ne=ne,
                screening=screening,
                variant=variant,
                pair_energy=pair_energy,
                isoangle=isoangle,
                halo_id_params=halo_id_params,
                useVerne=useVerne,
                calcErrors=calcErrors,
                debug=debug,
                mediator_spin=mediator_spin,
                sigma_e=sigma_e,
                modulated_source=modulated_source,
                srdm_base_data_dir=srdm_base_data_dir,
                modulation_source=modulation_source,
            )
        if self.material == 'Ge':
            if self.ionization_func is not self.step_probabilities:
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
        sigma_e_cm2 = float(self.cross_section / nu.cm**2) if sigma_e is None else float(sigma_e)

        debug_last_dRdE = None
        for m,mX in enumerate(mX_array):
            if halo_model in _SRDM_HALO_MODELS:
                spectrum = compute_form_factor_spectrum(
                    material=self.material,
                    mX=mX,
                    FDMn=FDMn,
                    halo_model=halo_model,
                    DoScreen=DoScreen,
                    integrate=integrate,
                    QEDark=self.QEDark,
                    form_factor=self.form_factor,
                    qArr=self.qArr,
                    Earr=self.Earr,
                    Ei_array=self.Ei_array,
                    dtype_str=self.dtype_str,
                    reduced_mass_fn=self.reduced_mass,
                    fdm_fn=self.FDM,
                    get_parametrized_eta_fn=self.get_parametrized_eta,
                    vmin_tensor_fn=self.vMin_tensor,
                    tfscreening_fn=self.TFscreening,
                    thomas_fermi_screening_fn=self.thomas_fermi_screening,
                    halo_id_params=halo_id_params,
                    sigma_e_cm2=sigma_e_cm2,
                    mediator_spin=mediator_spin,
                    screening=screening,
                    lindhard_eta_eV=lindhard_eta_eV,
                    ring_index=isoangle,
                    modulated_source=modulated_source,
                    srdm_base_data_dir=srdm_base_data_dir,
                )
                self.Earr = spectrum.E
                dRdne = semiconductor_rates_from_spectrum(
                    spectrum,
                    prob_fn_tiled,
                    integrate=integrate,
                )
                debug_last_dRdE = spectrum.dR_dE
            else:
                self.setup_halo_data(mX,FDMn,halo_model,isoangle=isoangle,useVerne=useVerne,calcErrors=calcErrors)
                dRdE = self.vectorized_dRdE(
                    mX,
                    FDMn,
                    halo_model,
                    DoScreen=DoScreen,
                    halo_id_params=halo_id_params,
                    integrate=integrate,
                    screening=screening,
                    lindhard_eta_eV=lindhard_eta_eV,
                ) #this is in 1 /kg/year/eV , but units are still implicit
                debug_last_dRdE = dRdE

                if integrate:
                    #TODO
                    #maybe change this to simpsons rule too
                    dRdne = torch.trapezoid(dRdE*prob_fn_tiled,x=self.Earr, axis = 1)
                else:
                    # Legacy QEDark summed these grid values directly; since the
                    # semiconductor form-factor grids use dE=0.1 eV, dE*10 supplies
                    # the missing energy unit as 1 eV without changing that convention.
                    dRdne = torch.sum(dRdE*prob_fn_tiled*self.form_factor.dE*10, axis = 1)

            dRdnEs[m,:] = dRdne
        if debug:
            return dRdnEs.T,debug_last_dRdE,prob_fn_tiled
        return dRdnEs.T #should be in kg/year
    
    def rate_dme_shell(self,mX,FDMn,halo_model,shell_key,halo_id_params=None,debug=False,unitize=False):
        """Calculate rate for specific electron shell in noble elements.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            shell_key: Electron shell identifier
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Rate for specified shell
        """
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
        """Calculate differential rates for all shells in noble elements.
        
        Args:
            mX: DM mass
            FDMn: Form factor model
            halo_model: Velocity distribution model
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            debug: Return debug information
            unitize: Convert to standard units
            
        Returns:
            Dictionary of differential rates by shell
        """

        mX = mX*nu.MeV / nu.c0**2
        drs = dict()
        for key in self.form_factor.keys:
            if key in skip_keys[self.material]:
                continue
            dr = self.rate_dme_shell(mX,FDMn,halo_model,key,halo_id_params=halo_id_params,debug=False,unitize=unitize)
            drs[key] = dr
        return drs
    

    def energy_to_ne_pmf(self,rates,shell,nes_tensor,p_primary,p_secondary):
        """Convert energy differential rates to electron count probabilities.
        
        Args:
            rates: Differential rates
            shell: Electron shell
            nes_tensor: Electron count values
            p_primary: Primary probability
            p_secondary: Secondary probability
            
        Returns:
            Probability mass function values
        """
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
        """Vectorized version of energy_to_ne_pmf.
        
        Args:
            rates: Differential rates
            shell: Electron shell
            nes_tensor: Electron count values
            p_primary: Primary probability
            p_secondary: Secondary probability
            
        Returns:
            Probability mass function values
        """
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
        """Convert differential rates to electron count rates.
        
        Args:
            drs: Dictionary of differential rates by shell
            nes: Electron values to calculate for (list/int)
            p_primary: Primary probability
            p_secondary: Secondary probability
            swap_4s4p: Special handling for Xenon shells because the form factor data was swapped
            
        Returns:
            Dictionary of rates by shell
        """
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
            
                



            







    def calculate_nobleGas_rates(self,mX_array,halo_model,FDMn,ne,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False,returnShells=False,sigma_e=None,mediator_spin='vector',modulated_source=None,srdm_base_data_dir=None,modulation_source=None):
        """Calculate rates for noble gas targets (Xe, Ar).
        
            Args:
                mX_array: Array of DM masses
                halo_model: Velocity distribution model
                FDMn: Form factor model
                ne: Number of electrons (list,int)
                isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
                halo_id_params: Parameters for step function model (used for halo independent analysis)
                useVerne: Use Verne distribution
                calcErrors: Calculate errors ('High'/'Low')
                debug: Return debug information
                returnShells: Return shell-by-shell breakdown
                sigma_e: SRDM cross section in cm^2; defaults to self.cross_section
                mediator_spin: SRDM mediator mode. Noble SRDM supports 'vector'.
                
            Returns:
                Calculated rates (and shell breakdown if requested)
        """
        import torch
        import numpy
        useVerne = _resolve_ordinary_modulation_source(
            halo_model,
            useVerne,
            modulation_source,
        )
        modulated_source = _validate_srdm_modulated_call(halo_model, isoangle, modulated_source)
        if halo_model in _SRDM_HALO_MODELS:
            from .srdm.mediators import normalize_mediator_spin
            mediator_spin = normalize_mediator_spin(mediator_spin)

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

        sigma_e_cm2 = float(self.cross_section / nu.cm**2) if sigma_e is None else float(sigma_e)
        
        for m,mX in enumerate(mX_array):
            if halo_model in _SRDM_HALO_MODELS:
                spectrum = noble_srdm_dRdE_spectrum(
                    material=self.material,
                    mX=mX,
                    FDMn=FDMn,
                    mediator_spin=mediator_spin,
                    sigma_e_cm2=sigma_e_cm2,
                    form_factor=self.form_factor,
                    qArrdict=self.qArrdict,
                    Earr=self.Earr,
                    reduced_mass_fn=self.reduced_mass,
                    fdm_fn=self.FDM,
                    vmin_tensor_fn=self.vMin_tensor,
                    halo_model=halo_model,
                    modulated_source=modulated_source,
                    ring_index=isoangle,
                    srdm_base_data_dir=srdm_base_data_dir,
                )
                drs = spectrum.shell_spectra
            else:
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











    def calculate_rates(self,mX_array,halo_model,FDMn,ne,integrate=True,DoScreen=True,isoangle=None,halo_id_params=None,useVerne=False,calcErrors=None,debug=False,screening=None,variant=None,pair_energy=None,mediator_spin='vector',sigma_e=None,lindhard_eta_eV=DEFAULT_LINDHARD_ETA_EV,modulated_source=None,srdm_base_data_dir=None,modulation_source=None):
        """Main rate calculation method that routes to appropriate implementation.
        
        Args:
            mX_array: Array of DM masses
            halo_model: Velocity distribution model
            FDMn: Form factor model
            ne: Number of electrons/electron-hole pairs (list/int)
            integrate: Perform numerical integration
            DoScreen: Apply screening effects
            screening: Explicit screening mode
            lindhard_eta_eV: Lindhard imaginary broadening in eV
            isoangle: isoangle (for modulated distributions) (number from 0 -35 with the true isoangle being 5x that value)
            halo_id_params: Parameters for step function model (used for halo independent analysis)
            modulated_source: SRDMBeam upstream source ('Verne', 'DaMaSCUS', or 'SRDMBeam').
                Only valid with halo_model='srdm_modulated'. Not to be confused with
                modulation_source, which selects the ordinary halo-DM modulation source.
            modulation_source: Ordinary halo modulation source ('Verne' or 'DaMaSCUS').
                Only valid with halo_model='modulated' or 'summer'. Overrides useVerne.
            useVerne: Use Verne distribution
            calcErrors: Calculate errors ('High'/'Low')
            debug: Return debug information
            
        Returns:
            Calculated rates
        """
        useVerne = _resolve_ordinary_modulation_source(
            halo_model,
            useVerne,
            modulation_source,
        )
        modulated_source = _validate_srdm_modulated_call(halo_model, isoangle, modulated_source)
        if halo_model in _SRDM_HALO_MODELS:
            from .srdm.mediators import normalize_mediator_spin
            mediator_spin = normalize_mediator_spin(mediator_spin)
        if self.form_factor_type == 'qcdark2':
            return self.calculate_semiconductor_rates(
                mX_array,
                halo_model,
                FDMn,
                ne,
                integrate,
                DoScreen,
                isoangle=isoangle,
                halo_id_params=halo_id_params,
                useVerne=useVerne,
                calcErrors=calcErrors,
                debug=debug,
                screening=screening,
                variant=variant,
                pair_energy=pair_energy,
                mediator_spin=mediator_spin,
                sigma_e=sigma_e,
                lindhard_eta_eV=lindhard_eta_eV,
                modulated_source=modulated_source,
                srdm_base_data_dir=srdm_base_data_dir,
                modulation_source=modulation_source,
            )
        if self.material == 'Si' or self.material =='Ge':
            return self.calculate_semiconductor_rates(
                mX_array,
                halo_model,
                FDMn,
                ne,
                integrate,
                DoScreen,
                isoangle=isoangle,
                halo_id_params=halo_id_params,
                useVerne=useVerne,
                calcErrors=calcErrors,
                debug=debug,
                screening=screening,
                variant=variant,
                pair_energy=pair_energy,
                mediator_spin=mediator_spin,
                sigma_e=sigma_e,
                lindhard_eta_eV=lindhard_eta_eV,
                modulated_source=modulated_source,
                srdm_base_data_dir=srdm_base_data_dir,
                modulation_source=modulation_source,
            )
        if self.material == 'Xe' or self.material == 'Ar':
            return self.calculate_nobleGas_rates(
                mX_array,
                halo_model,
                FDMn,
                ne,
                isoangle=isoangle,
                halo_id_params=halo_id_params,
                useVerne=useVerne,
                calcErrors=calcErrors,
                debug=debug,
                returnShells=False,
                sigma_e=sigma_e,
                mediator_spin=mediator_spin,
                modulated_source=modulated_source,
                srdm_base_data_dir=srdm_base_data_dir,
                modulation_source=modulation_source,
            )
    
    
    

    def generate_dat(self,dm_masses,ne_bins,fdm,dm_halo_model,DoScreen=False,write=True,tag=""):
        """Generate rate data files for storage and analysis.
        
        Args:
            dm_masses: Array of DM masses
            ne_bins: Electron count bins
            fdm: Form factor model 
            dm_halo_model: Velocity distribution model
            DoScreen: Apply screening effects
            write: Save to file
            tag: Additional filename tag
            
        Returns:
            Array of calculated rates
        """

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
            g_day = gday.detach().cpu().numpy()
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
            f.close()
            self._write_sidecar(filename, halo_model=dm_halo_model, FDMn=fdm, DoScreen=DoScreen, integrate=integrate)
        return data

    def _write_sidecar(self, dat_path: str, halo_model: str, FDMn: int, DoScreen: bool, integrate: bool):
        import yaml
        import DMeRates as _pkg
        sidecar_path = dat_path + ".yaml"
        config = {
            'package_version': _pkg.__version__,
            'backend':  self.form_factor_type,
            'material': self.material,
            'physics': {
                'v0_km_s':      float(self.v0      / (nu.km / nu.s)),
                'vEarth_km_s':  float(self.vEarth  / (nu.km / nu.s)),
                'vEscape_km_s': float(self.vEscape / (nu.km / nu.s)),
                'rhoX_GeV_cm3': float(self.rhoX    / (1e9 * nu.eV / nu.c0**2 / nu.cm**3)),
                'sigma_e_cm2':  float(self.cross_section / nu.cm**2),
            },
            'halo_model': halo_model,
            'FDMn':      FDMn,
            'screening': DoScreen,
            'integrate': integrate,
        }
        qcdark2_variant = getattr(self, 'qcdark2_variant', None)
        if qcdark2_variant is not None:
            config['qcdark2_variant'] = qcdark2_variant
        with open(sidecar_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        

    
