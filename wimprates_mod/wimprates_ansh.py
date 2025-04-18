
#constants
me_eV = 5.1099894e5
mP_eV = 938.27208816 *1e6

kg_per_amu = 1.660540199e-27
c0 = lightSpeed = 299792458 # m / s
alphaFS = 1/137 #fine structure constant
lightSpeed_kmpers = lightSpeed *1e-3
ry = 13.605693122990 #eV
ccms = lightSpeed*1e2
secperday = 60*60*24
secperyear =secperday * 365.25

dayspersecond = 1/secperday
km_per_cm = 1e-5

years_per_day = 1/365.25

ATOMIC_WEIGHT = dict(
    Xe=131.293,
    Ar=39.948,
    Ge=72.64,
    Si=28.0855
)
# vEscape = 544 #km/s
# vEarth = 250.2 #km/s
# v0 = 238.0 #km/s
# rho_DM = 0.4 #GeV/cm^3

# def eta_MB(vMin):    #same as SHM but the indefinite integration is done so this is faster. In units of inverse vMin
#     import numpy as np
#     from scipy.special import erf
#     if (vMin < vEscape - vEarth):
#         val = -4.0*vEarth*np.exp(-(vEscape/v0)**2) + np.sqrt(np.pi)*.0*(erf((vMin+vEarth)/v0) - erf((vMin - vEarth)/v0))
#     elif (vMin < vEscape + vEarth):
#         val = -2.0*(vEarth+vEscape-vMin)*np.exp(-(vEscape/v0)**2) + np.sqrt(np.pi)*v0*(erf(vEscape/v0) - erf((vMin - vEarth)/v0))
#     else:
#         val = 0.0
#     K = (v0**3)*(-2.0*np.pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (np.pi**1.5)*erf(vEscape/v0))
#     return (v0**2)*np.pi/(2.0*vEarth*K)*val




def get_halo_data(mX,sigmaE,fdm,isoangle=None,useVerne=True,oldParams=False,calcErrors=None):
    #assumes mX is in eV
    import numpy as np
    from scipy.interpolate import interp1d,PchipInterpolator,Akima1DInterpolator
    import os
    if isoangle is None:
        # return eta_MB

        file = 'halo_data/shm_v0238.0_vE250.2_vEsc544.0_rhoX0.3.txt'
        if oldParams:
            file = 'halo_data/shm_v0220.0_vE244.0_vEsc544.0_rhoX0.4.txt'
        if not os.path.isfile(file):
            # print(useVerne,dir)
            print(file)
            raise FileNotFoundError('halo file not found')
        data = np.loadtxt(file)
        file_etas = data[:,1]
        file_vmins = data[:,0]
        eta_func = interp1d(file_vmins,file_etas,fill_value=0,bounds_error=False)
        # eta_func = interp1d(file_vmins,file_etas,fill_value='extrapolate',bounds_error=False)
        return eta_func,file_vmins
        

    else:
        mass_string = mX*1e-6 #turn into MeV

        sigmaE = float(format(sigmaE, '.3g'))

        mass_string = float(mass_string)
        mass_string = np.round(mass_string,3)
        mass_string = str(mass_string).replace('.',"_")

        if fdm == '1':
            fdm_str = 'Scr'
        else:
            fdm_str = 'LM'
        if useVerne:
            dir = f'../halo_data/modulated/Verne_{fdm_str}/'
        else:
            dir = f'../halo_data/modulated/Parameter_Scan_{fdm_str}/'
        
        file = f'{dir}mDM_{mass_string}_MeV_sigmaE_{sigmaE}_cm2/DM_Eta_theta_{isoangle}.txt'
        if not os.path.isfile(file):
            # print(useVerne,dir)
            print(file)
            raise FileNotFoundError('halo file not found')
        data = np.loadtxt(file,delimiter='\t')
        file_etas = data[:,1]
        if useVerne is False and calcErrors is not None:
            if calcErrors == 'High':
                file_eta_err = data[:,2]
                file_etas += file_eta_err
            if calcErrors == 'Low':
                file_eta_err = data[:,2]
                file_etas -= file_eta_err
        
        file_vmins = data[:,0]
        # eta_func = PchipInterpolator(file_vmins,file_etas)
        eta_func = interp1d(file_vmins,file_etas,fill_value=0,bounds_error=False)
        # eta_func = interp1d(file_vmins,file_etas,fill_value='extrapolate',bounds_error=False)


        return eta_func,file_vmins




def mn(material='Xe'):
    """Mass of nucleus (not nucleon!)"""
    return ATOMIC_WEIGHT[material] * kg_per_amu #kg

 
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





additional_quanta = {
    'Xe':{
        '4s': 3,
        '4p': 6,
        '4d': 4,
        '5s': 0,
        '5p': 0,
        '3p': 0, #not sure if this is right..
        '3d': 0 #not sure if this is right..
        },
    'Ar': {
        '3s': 0,
        '3p12': 0,
        '3p32': 0,
    }}


binding_es = {
    'Xe':{
        '4s': 213.8,
        '4p': 163.5,
        '4d': 75.6,
        '5s': 25.7,
        '5p': 12.4
        },
    # 'Ar': {
    #     '3s': 29.3,
    #     '3p12': 15.9,
    #     '3p32': 15.7,
    #     }
    'Ar': { #the darkside versions
        '3s': 34.76,
        '3p12': 16.08,
        '3p32': 16.08,
        }
}

skip_keys = {
    'Xe': ['3s','3p','3d'],
    'Ar': ['3p32'],
}

def get_binding_es(mat,key):
    return binding_es[mat][key] #eV


def get_shelL_data(material):
    from scipy.interpolate import RegularGridInterpolator, interp1d
    import numpy as np

    import pickle

    if material == 'Xe':
        # try:
            # file = 'data/dme/dme_ionization_ff.pkl'
        # except FileNotFoundError:
        file = 'wimprates_mod/data/dme/dme_ionization_ff.pkl'
    elif material == 'Ar':
        # try:
        #     file = 'data/dme/dme_ionization_ff_argon.pkl'
        # except FileNotFoundError:
        file = 'wimprates_mod/data/dme/dme_ionization_ff_argon.pkl'
    
    with open(file, mode='rb') as f:
        shell_data = pickle.load(f)
    keys = list(shell_data.keys())
    for _shell_, _sd_ in shell_data.items():
        _sd_['log10ffsquared_itp'] = RegularGridInterpolator(
            (_sd_['lnks'], _sd_['lnqs']),
            np.log10(_sd_['ffsquared']),
            bounds_error=False, fill_value=-float('inf'),)
    return shell_data,keys

# @export
def dme_ionization_ff(indexed_shell_data,e_er, q):
    import numpy as np
    """Return dark matter electron scattering ionization form factor

    Outside the parametrized range, the form factor is assumed 0
    to give conservative results.

    :param shell: Name of atomic shell, e.g. '4p'
        Note not all shells are included in the data.
    :param e_er: Electronic recoil energy
    :param q: Momentun transfer
    """


    lnq = np.log(q / (me_eV * alphaFS))
    # From Mathematica: (*ER*) (2 lnkvalues[[j]])/Log[10]
    # log10 (E/Ry) = 2 lnk / ln10
    # lnk = log10(E/Ry) * ln10 / 2
    #     = lng(E/Ry) / 2
    # Ry = rydberg = 13.6 eV

    lnk = np.log(e_er / ry) / 2
    # print(lnk,lnq)
    try:
        index =  np.concatenate([lnk, lnq]).T
    except ValueError:
        index =  np.vstack([lnk, lnq]).T

    return 10**(indexed_shell_data['log10ffsquared_itp'](index
       ))




        
def v_min_dme(eb, erec, q, mX):
    """Minimal DM velocity for DM-electron scattering
    :param eb: binding energy of shell
    :param erec: electronic recoil energy
    :param q: momentum transfer
    :param mX: DM mass in eV
    """
    return (erec + eb) / q + q / (2 * mX)


# def velocity_integral_without_time(halo_model=None):
#     from scipy.integrate import quad
#     from scipy.interpolate import interp1d
#     import numpy as np
#     halo_model = StandardHaloModel() if halo_model is None else halo_model
#     _v_mins = np.linspace(0, 1, 1000) * v_max(None, halo_model.v_esc)
#     _ims = np.array([
#         quad(lambda v: 1 / v * halo_model.velocity_dist(v,None),
#             _v_min,
#              v_max(None, halo_model.v_esc ))[0]
#         for _v_min in _v_mins])
    
#     # Store interpolator in km/s rather than unit-dependent numbers
#     # so we don't have to recalculate them when nu.reset_units() is called
#     inverse_mean_speed_kms = interp1d(
#         _v_mins,
#         _ims,
#         # If we don't have 0 < v_min < v_max, we want to return 0
#         # so the integrand vanishes
#         fill_value=0, bounds_error=False)
#     return inverse_mean_speed_kms

# inverse_mean_speed_kms = velocity_integral_without_time()

def rate_dme_sum(e_er,shell_key,mX,sigma_dme,f_dm='1',isoangle=None,debug=False,mat = 'Xe',useVerne=True,lnE = False,oldParams=False,numqs = 1000,calcErrors=None):
    import numpy as np
    eta,file_vmins = get_halo_data(mX,sigma_dme,f_dm,isoangle=isoangle,useVerne= useVerne,oldParams=oldParams,calcErrors=calcErrors)
    
    eb = get_binding_es(mat,shell_key)
    f_dm = {
        '1': lambda q: np.ones_like(q),
        '1_q': lambda q:  alphaFS * me_eV / q,
        '1_q2': lambda q: (alphaFS * me_eV  / q)**2
    }[f_dm]

    shell_data,keys = get_shelL_data(mat)
    if shell_key not in keys:
        print('you have given an invalid or unimplemented shell for this material')
        return -1
    
    indexed_shell_data = shell_data[shell_key]
    qmax = (np.exp(shell_data[shell_key]['lnqs'].max()))
    # print('QMAX',qmax)#eV))
    qmax *= me_eV * alphaFS 

    qmin = (np.exp(shell_data[shell_key]['lnqs'].min()))
    #10000000
    qs,dQ = np.linspace(qmin,qmax,numqs,retstep=True)
    

    qs_tiled = np.tile(qs,(len(e_er),1))
    e_er_tiled = np.tile(e_er,(len(qs),1)).T


    lnqi = np.log(qs / (me_eV * alphaFS))
    lnk  = np.log(e_er / ry) / 2

    ffdata = np.zeros_like(e_er_tiled)
    for i,k in enumerate(lnk):
        ffdata[i,:] = indexed_shell_data['log10ffsquared_itp']((k,lnqi))

    ffdata = 10**ffdata 



    vmins = v_min_dme(eb, e_er_tiled, qs_tiled, mX) * lightSpeed_kmpers
    
    etas = eta(vmins)
    # etas[vmins > file_vmins[-1]] = 0
    # etas[vmins > 750] = 0

    
    f_dms_tiled = f_dm(qs_tiled)**2
   

    result =  qs_tiled * ffdata *f_dms_tiled**2 *etas
    result = np.sum(result * dQ,axis=1) #sum over all q


            # print("checking each component to see why I don't match")
            # print('q')
            # print(qs)
            # print('qmin')
            # print(qmin)
            # print('e_er')
            # print(e_er)
            # print('ff')
            # print(ffdata)
            # print('fdm')
            # print(f_dms_tiled)
            # print('eta')
            # print(etas)
            # print('result')
            # print(result)
            # print('length')
            # print(len(qs))



    if oldParams:
        rho_DM = 0.4
    else:
        rho_DM = 0.3

    if lnE:
        div_factor = 1
    else:
        div_factor = (1/e_er)  * 1000 #eV/keV -> 1/kg * day *keV


    mXGeV = mX *1e-9
    mu_e = mu_Xe(mX)

    units = ((rho_DM)/mXGeV) * (1 / mn(mat)) * sigma_dme / (8 * mu_e ** 2)* km_per_cm* ccms**2 * secperyear  * div_factor

    if debug:
        prefactor =  1 /( 8 * (mu_e ** 2) * mn(mat)) 
        print('returning debug output:')
        print('result,prefactor,ffdata,f_dms_tiled,etas,qs,qmin,qmax')
        return result*units,prefactor,ffdata,f_dms_tiled,etas,qs,qmin,qmax
    return result * units












    #     vmins = v_min_dme(eb, erec, qs, mX) * lightSpeed_kmpers
    #     ffdata = dme_ionization_ff(indexed_shell_data,erec, qs)
    #     f_dm_res =  f_dm(qs)**2
    #     etas = eta(vmins)
    #     result = qs*ffdata*f_dm_res*etas


# from utils import vectorize_first
# @vectorize_first
def rate_dme(erec, shell_key, mX, sigma_dme,
             f_dm='1',isoangle=None,verbose=False,mat = 'Xe',useVerne=True,lnE = False,oldParams=False,calcErrors=None):
    
    """Return differential rate of dark matter electron scattering vs energy
    (i.e. dr/dE, not dr/dlogE)
    :param erec: Electronic recoil energy
    :param n: Principal quantum numbers of the shell that is hit
    :param l: Angular momentum quantum number of the shell that is hit
    :param mw: DM mass [GeV]
    :param sigma_dme: DM-free electron scattering cross-section at fixed
    momentum transfer q=0
    :param f_dm: One of the following:
     '1':     |F_DM|^2 = 1, contact interaction / heavy mediator (default)
     '1_q':   |F_DM|^2 = (\alpha m_e c / q), dipole moment
     '1_q2': |F_DM|^2 = (\alpha m_e c / q)^2, ultralight mediator
    :param t: A J2000.0 timestamp.
    If not given, a conservative velocity distribution is used.
    :param halo_model: class (default to standard halo model) containing velocity distribution
    """
    from scipy.integrate import quad, dblquad
    import numpy as np




    eta,file_vmins = get_halo_data(mX,sigma_dme,f_dm,isoangle=isoangle,useVerne= useVerne,oldParams=oldParams,calcErrors=calcErrors)
    
    
    eb = get_binding_es(mat,shell_key)

    f_dm = {
        '1': lambda q: 1,
        '1_q': lambda q:  alphaFS * me_eV / q,
        '1_q2': lambda q: (alphaFS * me_eV  / q)**2
    }[f_dm]

    # No bounds are given for the q integral
    # but the form factors are only specified in a limited range of q
    shell_data,keys = get_shelL_data(mat)
    if shell_key not in keys:
        print('you have given an invalid or unimplemented shell for this material')
        return -1
    
    indexed_shell_data = shell_data[shell_key]
    qmax = (np.exp(shell_data[shell_key]['lnqs'].max()))
    # print('QMAX',qmax)#eV))
    qmax *= me_eV * alphaFS 

    # if t is None and etafromfile is False:
        # Use precomputed inverse mean speed,
        # so we only have to do a single integral

    # if integrate:
    def diff_xsec(q):
        vmin = v_min_dme(eb, erec, q, mX) * lightSpeed_kmpers
        # print('vmin',vmin)
        # print('inputs to dme_ionization_ff: erec, q')

        # print(erec,q)
        result = q * dme_ionization_ff(indexed_shell_data,erec, q) * f_dm(q)**2 * eta(vmin)
        if verbose:
            print("checking each component to see why I don't match")
            print('q')
            print(q)
            print('ff')
            print(dme_ionization_ff(indexed_shell_data,erec, q))
            print('fdm')
            print(f_dm(q)**2)
            print('eta')
            print(eta(vmin))
        # Note the interpolator is in kms, not unit-carrying numbers
        # see above
        # result *= eta(vmin) #should be in s/km
        # result *= 1e-5 #km/cm
        # result /= lightSpeed_kmpers #unitless
        test = result[0]
        
        return result
    r = quad(diff_xsec, 0, qmax)[0]# s/km * ev^2 / c^2
    if verbose:
        print('result')
        print(r)

    # else:
    #     #do an approximation with summing rather than doing a slow quadrature integral
    #     qmin = (np.exp(shell_data[shell_key]['lnqs'].min()))
    #     qs = np.geomspace(qmin,qmax,100)
    #     vmins = v_min_dme(eb, erec, qs, mX) * lightSpeed_kmpers
    #     ffdata = dme_ionization_ff(indexed_shell_data,erec, qs)
    #     f_dm_res =  f_dm(qs)**2
    #     etas = eta(vmins)
    #     result = qs*ffdata*f_dm_res*etas

    #     r = np.sum(result,axis=1)


    #     #dme_ionization_ff(indexed_shell_data,erec, q) * f_dm(q)**2 * eta(vmin)





    # * rho /m [cm/s][1/cm^3]->[1/cm^2/s] 
    #* n_t number density [1/kg][1/cm] ->[1/kg/cm^2/s]
    #sigma_dme [cm^2] -> [1/kg*s] 
    #secperday
    # r [unitless?]

    if oldParams:
        rho_DM = 0.4
    else:
        rho_DM = 0.3


    mXGeV = mX *1e-9
    mu_e = mu_Xe(mX)
    if lnE:
        return (
            # Convert cross-section to rate, as usual
            ((rho_DM)/mXGeV) * (1 / mn(mat)) # 1 / kg cm^3
            # Prefactors in cross-section
            * sigma_dme / (8 * mu_e ** 2) #  cm^2 * c^2 / eV^2 -> 1/ cm kg * c^2 /ev^2
            * r # s/km * ev^2 / c^2 -> s/ cm *km kg 
            * km_per_cm #km/cm -> s/cm/kg
            * ccms**2 # cm^2 / s^2 -> 1 / s/kg
            * secperyear # 1/kg/year
            )
    
            # 

    else:
        return (
            
            # Convert cross-section to rate, as usual
            ((rho_DM)/mXGeV) * (1 / mn(mat)) # 1 / kg cm^3
            # d/lnE -> d/E
            * (1 / (erec)) # 1/eV -> 1 / kg cm^3 eV
            # Prefactors in cross-section
             * sigma_dme / (8 * mu_e ** 2) #  cm^2 * c^2 / eV^2 -> 1/ cm kg eV * c^2 /ev^2
            * r # s/km * ev^2 / c^2 -> s/ cm *km kg eV 
            * km_per_cm #km/cm -> s/cm/kg/eV
            * ccms**2 # cm^2 / s^2 -> 1 / s/kg /eV
            * secperyear # 1/kg year eV
            * 1000 #eV/keV -> 1/kg * day *keV
            )



def rates_to_ne(e_er, drs,
                W=None, max_n_el=16,
                p_primary=1, p_secondary=0.83,
                swap_4s4p=False,mat='Xe'):
    import numpy as np
    from scipy.stats import binom
    """Return (n_electrons, {shell: rate / (kg year) for each electron count})
    
    :param W: Work function (energy need to produce a quantum)
    :param max_n_el: Maximum number of electrons to consider.
    :param p_primary: Probability that primary electron survives
    :param p_secondary: Probability that secondary quanta survive as electrons
    :param swap_4s4p: If True, swap differential rates of 4s and 4p
    """
    if mat == 'Xe':
        W = 13.8 #eV
    elif mat == 'Ar':
         W = 19.5  #eV
    else:
        raise ValueError("Invalid Material")
    
        
    n_el = np.arange(max_n_el + 1, dtype=int)
    result = dict()    
    
    # We need an "energy bin size" to multiply with (or do some fancy integration)
    # I'll use the differences between the points at which the differential 
    # rates were computed.
    # To ensure this doesn't give a bias, nearby bins can't differ too much 
    # (e.g. use a linspace or a high-n logspace/geomspace)
    binsizes = np.array(np.diff(e_er).tolist() + [e_er[-1] - e_er[-2]])
    
    for shell, rates in drs.items():
        if swap_4s4p and mat == 'Xe':
            # Somehow we can reproduce 1703.00910
            # if we swap 4s <-> 4p here??
            if shell == '4s':
                rates = drs['4p']
            elif shell == '4p':
                rates = drs['4s']

        # Convert to from energy to n_electrons
        r_n = np.zeros(len(n_el))
        for e, r in zip(e_er, rates * binsizes):

            fact = additional_quanta[mat][shell]
            n_secondary = int(np.floor(e / W)) + fact
            r_n += r * (
                p_primary * binom.pmf(n_el - 1, n=n_secondary, p=p_secondary)
                + (1 - p_primary) * binom.pmf(n_el, n=n_secondary, p=p_secondary))

        # We can't see "0-electron events"
        # Set their rate to 0 so we don't sum them accidentally
        r_n[0] = 0
        
        result[shell] = r_n
        
    return n_el, result



def dRdE(m_MeV,sigmaE,fdm,e_er = None,isoangle = None,mat = 'Xe',useVerne = True,verbose=False,oldParams = False,integrate=True,calcErrors=None):
    import numpy as np
    if e_er is None:
        e_er = np.geomspace(1, 400, 100)

    if fdm == 0:
        fdm_str = 'Scr'
        fdms = '1'
    else:
        fdm_str = 'LM'
        fdms = '1_q2'
    mass_str = str(np.round(float(m_MeV),3)).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    m_eV = m_MeV*1e6

    shell_data,keys = get_shelL_data(mat)
    drs = dict()
    for key in keys:
        if key in skip_keys[mat]:
            continue
        # if mat == 'Xe' and '3' in key: #somehow dont have info for these so here is my  clunky way of skipping
        #     continue
        # if mat == 'Ar' and '3p32' in key: #testing this 
        #     continue 
        if integrate:
            dr = rate_dme(e_er, key, m_eV, sigmaE,
                    f_dm=fdms,isoangle=isoangle,debug=verbose,mat = mat,useVerne=useVerne,lnE = False,oldParams=oldParams,calcErrors=calcErrors)
        else:
            dr = rate_dme_sum(e_er, key, m_eV, sigmaE,
                    f_dm=fdms,isoangle=isoangle,debug=verbose,mat = mat,useVerne=useVerne,lnE = False,oldParams=oldParams,calcErrors=calcErrors)
        drs[key] = dr
    return drs


def dRdlnE(m_MeV,sigmaE,fdm,e_er = None,isoangle = None,mat = 'Xe',useVerne = True,verbose=False,oldParams = False,integrate=True):
    import numpy as np
    if e_er is None:
        e_er = np.geomspace(1, 400, 100)

    if fdm == 0:
        fdm_str = 'Scr'
        fdms = '1'
    else:
        fdm_str = 'LM'
        fdms = '1_q2'
    mass_str = str(np.round(float(m_MeV),3)).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    m_eV = m_MeV*1e6

    shell_data,keys = get_shelL_data(mat)
    drs = dict()
    for key in keys:
        if key in skip_keys[mat]:
            continue
        if integrate:
            dr = rate_dme(e_er, key, m_eV, sigmaE,
                    f_dm=fdms,isoangle=isoangle,verbose=verbose,mat = mat,useVerne=useVerne,lnE = True,oldParams=oldParams)
        else:
            dr = rate_dme_sum(e_er, key, m_eV, sigmaE,
                    f_dm=fdms,isoangle=isoangle,verbose=verbose,mat = mat,useVerne=useVerne,lnE = True,oldParams=oldParams)
        drs[key] = dr
    return drs


def dRdne(m_MeV,sigmaE,fdm,e_er = None,isoangle=None,material='Xe',maxne = 16,ne='All',return_shells=False,useVerne=True,verbose=False,oldParams = False,integrate=True,calcErrors=None):
    import numpy as np
    if e_er is None:
        e_er = np.geomspace(1, 400, 100)
    drs = dRdE(m_MeV,sigmaE,fdm,e_er = e_er,isoangle = isoangle,mat = material,useVerne = useVerne,verbose=verbose,oldParams=oldParams,integrate=integrate,calcErrors=calcErrors)

    n_el,drsn = rates_to_ne(e_er, drs,max_n_el=maxne,p_primary=1, p_secondary=0.83,swap_4s4p=True,mat=material) #should be in 1/kg/year

    
    # for shell, rn in drsn.items():
    #     rn *= 365  #turn into rate / kg / year

    total_rates = np.sum(list(drsn.values()),axis=0) #should be in 1/kg/year
 
    if return_shells:
        return n_el,drsn
    if ne == 'All': 
        return n_el,total_rates
    elif type(ne) == int:
        return total_rates[ne]
    else:
        min_ne = ne[0]
        max_ne = ne[-1]
        return total_rates[min_ne:max_ne+1]
    

def plot_dRdne(n_el,drsn,mX_MeV,sigmaE,fdm,tonyear=False,day =False):
    import matplotlib.pyplot as plt
    import numpy as np
    if fdm == 0:
        fdm_str = 1
    else:
        fdm_str = '1/q$^2$'
    sigmaE = float(format(sigmaE, '.3g'))
    if tonyear:
        factor =1 #keV and ton year factors cancel
    else:
        factor = 1/1000 #to cancel keV factor
    if day:
        factor *=years_per_day
    
    for shell, rn in drsn.items():
        plt.plot(n_el, np.array(rn)*factor, drawstyle='steps-mid', label=shell)

    plt.plot(n_el, np.sum(np.array(list(drsn.values()))*factor, axis=0),
            label='Total',
            drawstyle='steps-mid', 
            linestyle='--', 
            c='k')

    title = f"$m_\chi$ = {mX_MeV} MeV/c$^2$, $\sigma_e =$ {sigmaE} cm$^2$, FDM = {fdm_str}" 

    plt.title(title)# + (' -- SWAP 4s<->4p' if True else ''))
    # plt.legend(loc='upper right', ncol=2)

    plt.xticks(np.arange(1, 16))
    plt.xlim(0.5, 15.5)
    plt.xlabel("N electrons")
    plt.legend()
    plt.yscale('log')
    # plt.ylim(1e-5, .45)
    if tonyear:
        extra_str = '1000'
    else:
        extra_str = ''
    if day:
        day_str = 'day'
    else:
        day_str = 'year'
    plt.ylabel(f"Rate [events / {extra_str} kg /{day_str}]")

    plt.show()
    plt.close()
    return


def plot_dRdE(drs,mX_MeV,sigmaE,e_er=None,mat= 'Xe',lnE = False,plotTotal = False,day=True):
    import numpy as np
    import matplotlib.pyplot as plt
    if e_er is None:
        e_er = np.geomspace(1, 400, 100)

    sigmaE = float(format(sigmaE, '.3g'))
    if lnE:
        energy_str = ''
        log_str = 'ln'
    else:
        energy_str = 'keV'
        log_str = ''
    # if mat == 'Xe':
    #     dmshells = dme_shells
    #     factor = nu.kg * energy_fac * nu.day
    #     argon = False
    # elif mat == 'Ar':
    #     dmshells = argon_dme_shells
    #     factor =  nu.kg * nu.day  * energy_fac
    #     argon = True
    for key in drs.keys():
        dr = drs[key] 
        if day:
            plot_dr = dr* years_per_day
            total = np.sum(list(drs.values()),axis=0) * years_per_day
        else:
            plot_dr = dr
            total = np.sum(list(drs.values()),axis=0) 

        plt.plot(e_er, plot_dr,
                label=key)
    if plotTotal:
        plt.plot(e_er, total,
            label='Total', c='black',alpha=1)
            
    title = f"{mat} " + "$m_\chi = %s$ $\mathrm{MeV}/c^2$, $\sigma_e =$ %s $\mathrm{cm}^2$, $F_\mathrm{DM} = 1$" % (mX_MeV, sigmaE)
    plt.title(title)
    # ax.legend(loc='upper right', ncol=2)
    plt.legend()
    plt.xlabel("$E_R$ [eV]")
    # plt.xlim(0, 400)

    plt.yscale('log')

    plt.ylabel(f"dR/d{log_str}E [1 / (kg day {energy_str})]")
    if lnE:
        plt.xscale('log')
        plt.xlim(1,np.max(e_er))

    # plt.ylim(1e-10, 1e-4)
    plt.tight_layout()
    plt.show()
    plt.close()



def get_modulated_rates(m_mev,sigmaE,fdm,useVerne =True,mat='Xe',ne=1,maxne=None,integrate=False,calcError=None):
    import os
    import numpy as np
    mass_str = str(np.round(float(m_mev),3)).replace('.','_')

    sigmaE = float(format(sigmaE, '.3g'))
    if fdm == 0:
        fdm_str = 'Scr'
        fdms = '1'
    else:
        fdm_str = 'LM'
        fdms = '1_q2'
    if useVerne:
        datadir = f'../halo_data/modulated/Verne_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'
    else:
        datadir = f'../halo_data/modulated/Parameter_Scan_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'

    num_angles = len(os.listdir(datadir))
    rates = []
    for i in range(num_angles):
        kgyear = dRdne(m_mev,sigmaE,fdm,isoangle=i,material=mat,ne=ne,return_shells = False,useVerne = useVerne,integrate=integrate,calcErrors=calcError)
                # print(kgyear)
        gday = kgyear/1000/365
        rates.append(gday)
    
    rates = np.array(rates)
    return np.linspace(0,180,num_angles),rates


# generate_modulated_wimprates(0,useVerne = True,overwrite=False,material='Xe')
# generate_modulated_wimprates(2,useVerne = True,overwrite=True,material='Xe')
# generate_modulated_wimprates(0,useVerne = True,overwrite=True,material='Ar')
# generate_modulated_wimprates(2,useVerne = True,overwrite=True,material='Ar')
def generate_modulated_wimprates(FDMn,useVerne = True,overwrite=False,material='Xe',integrate=False,verbose=False,ne=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]):

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
        dir = f'../halo_data/modulated/Verne_{fdm_str}/'

    else:
        dir = f'../halo_data/modulated/Parameter_Scan_{fdm_str}/'
    directories = os.listdir(dir)

    write_dir= f'damascus_modulated_rates_{material}'

    if useVerne:
        write_dir= f'verne_modulated_rates_{material}'
    
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
                if verbose:
                    print("This rate is generated, continuing")
                continue

            isoangles,rates = get_modulated_rates(mass,cross_sectionE,FDMn,useVerne =useVerne,mat=material,ne=ne,integrate=integrate)


            # isoangles= np.linspace(0,180,nangles)
            combined= np.vstack((isoangles,rates.T))
            combined = combined.T
            np.savetxt(outfile,combined,delimiter=',')
    return



# def fitted_rates(angles,rates):
#     import sys
#     sys.path.append('../QEDark/')
#     from modulation_contour import rate_fitting
#     import numpy as np
#     func_stuff  = rate_fitting(angles,rates)
#     if func_stuff == -1:
#         print('This rate does not exist or something else went wrong')
#         return -1
#     if len(func_stuff) > 2: 
#         fitFailed = False
#         fit_func = func_stuff[0]
#         a = func_stuff[1]
#         b = func_stuff[2]
#         c = func_stuff[3]
#         d = func_stuff[4]
#         sf = func_stuff[5]
        

#     else:
#         fitFailed = True
#         fit_func = func_stuff[0]
#         sf = func_stuff[1]
#         a = 1
#         b = 1
#         c = 1
#         d = 1

#     angles = np.linspace(0,180,500)

#     def fit(fitFailed,fit_func,angle,a,b,c,d):
#             if fitFailed:
#                 return fit_func(angle) #do interpolation I guess
#             else:
#                 return fit_func(angle,a,b,c,d)
            
#     rate_per_angle = fit(fitFailed,fit_func,angles,a,b,c,d)*sf
#     return angles,rate_per_angle