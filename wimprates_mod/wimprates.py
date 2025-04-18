import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from bremsstrahlung import *
from elastic_nr import *
from electron import *
from halo import *
# from migdal import *
from utils import *
# from summary import *
import numericalunits as nu


#seems like you only nneed to have it return a function 
from scipy.stats import binom

additional_quanta = {
    '4s': 3,
    '4p': 6,
    '4d': 4,
    '5s': 0,
    '5p': 0
}


def rates_to_ne(e_er, drs,
                W=None, max_n_el=16,
                p_primary=1, p_secondary=0.83,
                swap_4s4p=False,argon=False):
    """Return (n_electrons, {shell: rate / (kg day) for each electron count})
    
    :param W: Work function (energy need to produce a quantum)
    :param max_n_el: Maximum number of electrons to consider.
    :param p_primary: Probability that primary electron survives
    :param p_secondary: Probability that secondary quanta survive as electrons
    :param swap_4s4p: If True, swap differential rates of 4s and 4p
    """
    if W is None:
        W = 13.8 * nu.eV
        
    n_el = np.arange(max_n_el + 1, dtype=int)
    result = dict()    
    
    # We need an "energy bin size" to multiply with (or do some fancy integration)
    # I'll use the differences between the points at which the differential 
    # rates were computed.
    # To ensure this doesn't give a bias, nearby bins can't differ too much 
    # (e.g. use a linspace or a high-n logspace/geomspace)
    binsizes = np.array(np.diff(e_er).tolist() + [e_er[-1] - e_er[-2]])
    
    for shell, rates in drs.items():
        if swap_4s4p and not argon:
            # Somehow we can reproduce 1703.00910
            # if we swap 4s <-> 4p here??
            if shell == '4s':
                rates = drs['4p']
            elif shell == '4p':
                rates = drs['4s']

        # Convert to from energy to n_electrons
        r_n = np.zeros(len(n_el))
        for e, r in zip(e_er, rates * binsizes):
            if argon:
                fact = 0
            else:
                fact = additional_quanta[shell]
            n_secondary = int(np.floor(e / W)) + fact
            r_n += r * (
                p_primary * binom.pmf(n_el - 1, n=n_secondary, p=p_secondary)
                + (1 - p_primary) * binom.pmf(n_el, n=n_secondary, p=p_secondary))

        # We can't see "0-electron events"
        # Set their rate to 0 so we don't sum them accidentally
        r_n[0] = 0
        
        result[shell] = r_n
        
    return n_el, result

class ModulatedHalo:
    def __init__(self,data_dir = None,isoangle=None,rho_dm=None,v_esc = None,v_0 = None):
        from halo import _HALO_DEFAULTS
        self.v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
        self.v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = './'

        if isoangle is None:
            self.isoangle = 0
        else:
            self.isoangle = isoangle
        self.rho_dm = _HALO_DEFAULTS['rho_dm'] * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm
        self.file = self.data_dir + f'DM_Eta_theta_{isoangle}.txt'

    def velocity_dist(self,v):
        import numericalunits as nu
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator,interp1d   
        import numpy as np
        data = np.loadtxt(self.data_dir)
        gammas = data[:,0]
        vmins = data[:,1] 
        fv = data[:,2] 
        fvref = data[:,3]

        isoangles = np.unique(gammas)
        indices = np.where(gammas == isoangles[self.isoangle])
        velocity_func = fv + fvref 
        velocity_func = velocity_func[indices] 
        vmins = vmins[indices] 
        f_of_v = Akima1DInterpolator(vmins,velocity_func)
        result = f_of_v(v)
        # print('****************')
        # print('VMINS',vmins)
        # print('****************')
        # print('FV',velocity_func)
        v/= (nu.km/nu.s)
        if v > np.max(vmins):
            result = 0
        if np.isnan(result):
            print(v,result)
        
        return result
    def eta(self,vmin):
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator,interp1d
        
        data = np.loadtxt(self.file,delimiter='\t')  
        file_etas = data[:,1]
        file_vmins = data[:,0]
        eta_func = interp1d(
        file_vmins,
        file_etas,
        # If we don't have 0 < v_min < v_max, we want to return 0
        # so the integrand vanishes
        fill_value=0, bounds_error=False)
        return eta_func(vmin)


        
me_eV = 5.1099894e5
mP_eV = 938.27208816 *1e6
 
def mu_Xe(mX):
    """
    DM-electron reduced mass
    """
    return mX*me_eV/(mX+me_eV)


def mu_XP(mX):
    """
    DM-proton reduced mass
    """
    return mX*mP_eV/(mX+mP_eV)

#    #for eta     
# datadir = '../QEDark/halo_data/modulated/Verne_Scr/mDM_10_MeV_sigmaE_1e-31_cm2/'
# mod_halo_test = ModulatedHalo(data_dir=datadir,isoangle=0)

#    #for velocity distribution     
def sigmaE_to_sigmaP(sigmaE,mX):
    import numpy as np
    mX*=1e6 #eV
    sigmaP = sigmaE*(mu_XP(mX)/mu_Xe(mX))**2
    # sigmaP = np.round(sigmaP,3)
    return sigmaP


def sigmaP_to_sigmaE(sigmaP,mX):
    import numpy as np
    mX*=1e6 #eV
    sigmaE = sigmaP*(mu_Xe(mX)/mu_XP(mX))**2
    # sigmaP = np.round(sigmaP,3)
    return sigmaE



def dR(m_MeV,sigmaE,fdm,isoangle = None,mat = 'Xe',useVerne = True,lnE = False,largeGrid= False):
    if useVerne:
        dir_string = 'Verne'
    else:
        dir_string = 'Parameter_Scan'
    e_er = np.geomspace(1, 400, 100)
    if largeGrid:
        e_er = np.geomspace(1, 1000, 500)
    if fdm == 0:
        fdm_str = 'Scr'
        fdms = '1'
    else:
        fdm_str = 'LM'
        fdms = '1_q2'
    mass_str = str(np.round(float(m_MeV),3)).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    m_gev = m_MeV*1e-3
    s_cm2 = sigmaE
    # s_cm2 = sigmaE_to_sigmaP(sigmaE,m_MeV)
    datadir = f'../QEDark/halo_data/modulated/{dir_string}_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'
    if isoangle is not None:
        halo = ModulatedHalo(data_dir=datadir,isoangle=isoangle)
        
    else:
        halo = None

    if mat == 'Xe':
        dmeshells = dme_shells
        argon = False
    elif mat == 'Ar':
        dmeshells  = argon_dme_shells
        argon = True
    drs = dict()

    for n, l in dmeshells:

        dr = rate_dme(
        e_er * nu.eV, 
        n, l, 
        mw=m_gev * nu.GeV/nu.c0**2, 
        sigma_dme=s_cm2 * nu.cm**2,
        halo_model = halo,
        verbose=False,
        etafromfile=True,
        f_dm=fdms,mat=mat,lnE = lnE)

        if argon:
            dr *=5.963307622096163e-05 #seems to match results from https://arxiv.org/pdf/1802.06998 with this factor 

        drs[shell_str(n,l,argon=argon)] = dr
    return drs

    
def dRdnE(m_MeV,sigmaE,fdm,isoangle=None,mat = 'Xe',ne='All',return_shells = False,useVerne = True,largeGrid = False):
    drs =dR(m_MeV,sigmaE,fdm,isoangle = isoangle,mat = mat,useVerne = useVerne,largeGrid=largeGrid)
    if mat == "Ar":
        W = 19.5*nu.eV
        fe = 0.83
        fR = 0
        argon = True
    elif mat == 'Xe':
        W = 13.8 *nu.eV
        fe = 0.83
        fR = 0
        argon = False

    e_er = np.geomspace(1, 400, 100)
    n_el, drsn = rates_to_ne(e_er * nu.eV, drs, 
                            swap_4s4p=True,W = W,p_primary=1-fR,p_secondary=fe,argon=argon)
    
    for shell, rn in drsn.items():
        rn *= (nu.kg * nu.year) #turn into rate / kg / year

    
    total_rates = np.sum(list(drsn.values()),axis=0)

    if return_shells:
        return n_el,drsn
    if ne == 'All': 
        return n_el,total_rates
    else:
        return total_rates[ne]

    
    


def plot_dRdne(n_el,drsn,m_gev,s_cm2,tonyear=False,day =False):
    s_cm2 = float(format(s_cm2, '.3g'))
    if tonyear:
        factor = 1000
    else:
        factor = 1
    if day:
        factor /=365
    
    for shell, rn in drsn.items():
        plt.plot(n_el, np.array(rn)*factor, drawstyle='steps-mid', label=shell)

    plt.plot(n_el, np.sum(np.array(list(drsn.values()))*factor, axis=0),
            label='Total',
            drawstyle='steps-mid', 
            linestyle='--', 
            c='k')

    title = "$m_\chi = %s$ $\mathrm{GeV}/c^2$, $\sigma_p =$ %s $\mathrm{cm}^2$, $F_\mathrm{DM} = 1$" % (m_gev, s_cm2)

    plt.title(title)# + (' -- SWAP 4s<->4p' if True else ''))
    # plt.legend(loc='upper right', ncol=2)

    plt.xticks(np.arange(1, 16))
    plt.xlim(0.5, 15.5)
    plt.xlabel("N electrons")

    plt.yscale('log')
    # plt.ylim(1e-5, .45)
    if tonyear:
        extra_str = 'ton'
    else:
        extra_str = ''
    if day:
        day_str = 'day'
    else:
        day_str = 'year'
    plt.ylabel(f"Rate [events / {extra_str} kg /{day_str}]")

    plt.show()
    plt.close()

    
def plot_dR(drs,m_gev,s_cm2,mat= 'Xe',lnE = False,largeGrid = False,plotTotal = False):

    e_er = np.geomspace(1, 400, 100)
    if largeGrid:
        e_er = np.geomspace(1, 1000, 500)
    fig,ax = plt.subplots(figsize=(15,15))
    s_cm2 = float(format(s_cm2, '.3g'))
    if lnE:
        energy_fac = 1
        energy_str = ''
        log_str = 'ln'
    else:
        energy_fac = nu.keV
        energy_str = 'keV'
        log_str = ''
    if mat == 'Xe':
        dmshells = dme_shells
        factor = nu.kg * energy_fac * nu.day
        argon = False
    elif mat == 'Ar':
        dmshells = argon_dme_shells
        factor =  nu.kg * nu.day  * energy_fac
        argon = True
    
    for n, l in dmshells:
        dr = drs[shell_str(n,l,argon=argon)] 

        ax.plot(e_er, dr * factor,
                label=f'{shell_str(n, l,argon)}',ls=':')
    if plotTotal:
        ax.plot(e_er, np.sum(list(drs.values()), axis=0),
            label='Total', c='black', linestyle=':',alpha=1)
            
    title = "$m_\chi = %s$ $\mathrm{GeV}/c^2$, $\sigma_p =$ %s $\mathrm{cm}^2$, $F_\mathrm{DM} = 1$" % (m_gev, s_cm2)
    fig.suptitle(title)
    # ax.legend(loc='upper right', ncol=2)
    ax.legend()
    ax.set_xlabel("$E_R$ [eV]")
    # plt.xlim(0, 400)

    ax.set_yscale('log')

    ax.set_ylabel(f"dR/d{log_str}E [1 / (kg day {energy_str})]")
    if lnE:
        ax.set_xscale('log')
        ax.set_xlim(1,np.max(e_er))

    # plt.ylim(1e-10, 1e-4)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


        


def get_modulated_rates(m_mev,sigmaE,fdm,useVerne =True,mat='Xe',ne=1):
    import os
    mass_str = str(np.round(float(m_mev),3)).replace('.','_')

    sigmaE = float(format(sigmaE, '.3g'))
    if fdm == 0:
        fdm_str = 'Scr'
        fdms = '1'
    else:
        fdm_str = 'LM'
        fdms = '1_q2'
    datadir = f'../QEDark/halo_data/modulated/Verne_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'

    num_angles = len(os.listdir(datadir))
    rates = []
    for i in range(num_angles):
        rate = dRdnE(m_mev,sigmaE,fdm,isoangle=i,mat = mat,ne=ne,return_shells = False,useVerne = useVerne)
        rates.append(rate)
    rates = np.array(rates)
    return num_angles,rates



def generate_wimprates(FDMn,useVerne = True,overwrite=False,material='Xe'):

    import csv
    import re
    import numpy as np
    import os
    from tqdm.autonotebook import tqdm
    if FDMn == 0:
        fdm_str = 'Scr'
    else:
        fdm_str = 'LM'

    if useVerne: 
        dir = f'../QEDark/halo_data/modulated/Verne_{fdm_str}/'

    else:
        dir = f'../QEDark/halo_data/modulated/Parameter_Scan_{fdm_str}/'
    directories = os.listdir(dir)
    massesP = []
    massesE = []

    cross_sections = []
    sigmaEs = []

    write_dir= f'damascus_modulated_wimprates'

    if useVerne:
        write_dir= f'verne_modulated_wimprates'
    
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    for i in tqdm(range(len(directories))):
        d = directories[i]
        if 'Store' in d:
            continue
        mass_str = re.findall('DM_.*_MeV',d)[0][3:-4]
        mass = mass_str.replace('_','.')
        mass = float(mass)

        if 'sigmaE' in d:
            cross_sectionE = re.findall('E_.*cm',d)[0][2:-3].replace('_','.')
            cross_sectionE = float(cross_sectionE)

            outfile = write_dir+f'/mX_{mass_str}_MeV_sigmaE_{cross_sectionE}_FDM{FDMn}.csv'

            if os.path.isfile(outfile) and not overwrite:
                print("This rate is generated, continuing")
                continue

            nangles,rates = get_modulated_rates(mass,cross_sectionE,FDMn,useVerne =useVerne,mat=material,ne=1)

            #rates is in kg-years  
            rates /=1000
            rates/=365 #gdays


            isoangles= np.linspace(0,180,nangles)
            with open(outfile,'w') as f:
                writer = csv.writer(f,delimiter=',')
                writer.writerows(zip(isoangles,rates))
    return





