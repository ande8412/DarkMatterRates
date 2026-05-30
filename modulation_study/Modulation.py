import numericalunits as nu

#This contains all plotting/analysis code used for the modulation study we conducted.
#For all relevant figures please see the modulation_figures.ipynb notebook. 
northcolor='#2E88D1'
southcolor='#D1772E'


def _ordinary_modulation_source_to_use_verne(modulation_source):
    if modulation_source is None:
        return None
    key = str(modulation_source).strip().lower().replace("-", "_")
    if key == "verne":
        return True
    if key in ("damascus", "damascus_shm"):
        return False
    raise ValueError(
        f"Unsupported ordinary modulation_source={modulation_source!r}; "
        "expected 'Verne' or 'DaMaSCUS'"
    )


def _srdmbeam_representative_angles_for_indices(metadata, ring_indices):
    """Return SRDMBeam representative angles for interpolation and plotting."""
    import warnings
    import numpy as np

    ring_count = int(metadata["ring_count"])
    representative_angles = metadata.get("angle_representative_deg")
    if representative_angles is None:
        representative_angles = metadata.get("file_isoangle_deg")
    if representative_angles is None:
        raw_metadata = metadata.get("raw_metadata", {})
        if "file_isoangle_deg" in raw_metadata:
            representative_angles = raw_metadata["file_isoangle_deg"]
        elif metadata.get("angle_grid_type") == "bin_average":
            bin_edges = metadata.get("angle_bin_edges_deg")
            if bin_edges is None:
                bin_edges = np.linspace(0.0, 180.0, ring_count + 1)
            bin_edges = np.asarray(bin_edges, dtype=float)
            representative_angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if representative_angles is None:
        warnings.warn(
            "SRDMBeam metadata has no angle_representative_deg; using the "
            "legacy lower-edge angle grid idx * 180 / ring_count for "
            "interpolation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.array(
            [int(idx) * 180.0 / ring_count for idx in ring_indices],
            dtype=float,
        )

    angle_grid = np.asarray(representative_angles, dtype=float)
    if angle_grid.ndim != 1 or angle_grid.size != ring_count:
        raise ValueError(
            "SRDMBeam angle_representative_deg must contain one angle per ring: "
            f"expected {ring_count}, got shape {angle_grid.shape}"
        )

    selected = np.array([angle_grid[int(idx)] for idx in ring_indices], dtype=float)
    if not np.all(np.isfinite(selected)):
        raise ValueError("SRDMBeam angle_representative_deg contains non-finite values")
    return selected


def _create_dmrate(material, useQCDark=True):
    """Create DMeRate with engine-compatible form-factor selection."""
    import DMeRates
    import DMeRates.DMeRate as DMeRate

    noble_materials = {"XE", "AR"}
    material_key = str(material).upper()
    if material_key in noble_materials:
        # Noble targets always use wimprates in the new architecture.
        return DMeRate.DMeRate(material, form_factor_type='wimprates')

    ff_type = 'qcdark' if useQCDark else 'qedark'
    return DMeRate.DMeRate(material, form_factor_type=ff_type)

def set_default_plotting_params(fontsize=40):
    """Set default matplotlib plotting parameters for consistent figure styling.
    
    Args:
        fontsize (int): Base font size to use for all text elements (default: 40)
        
    Sets rcParams for:
        - LaTeX text rendering
        - Font family and size
        - Figure autolayout
        - Figure size
        - Unicode minus sign handling
    """

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    params = {'text.usetex' : True,
            'font.size' : fontsize,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=fontsize
    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden
    plt.rcParams['figure.figsize']=(16,12)
    return

def round_to_sig_figs(x, sig_figs=3):
    import math
    """
    Rounds a number to a specified number of significant figures.

    Args:
        x (float): The number to round.
        sig_figs (int): The desired number of significant figures.
    
    Returns:
        float: The number rounded to the specified significant figures.
    """
    if x == 0:
        return 0.0
    
    # Calculate the exponent based on the number of significant figures
    exponent = sig_figs - int(math.floor(math.log10(abs(x)))) - 1
    
    # Use the standard round() function with the calculated exponent
    # Note: Python's round() uses "round half to even" for .5 cases.
    return round(x, exponent)

def get_modulated_rates(material,mX,sigmaE,fdm,ne,useVerne=True,calcError=None,useQCDark=True,DoScreen = True,verbose = False,flat=False,dmRateObject = None,summer=False,modulation_source=None):
    """Calculate modulated DM-electron scattering rates for given parameters.
    
    Args:
        material (str): Target material ('Si', 'Xe', 'Ar')
        mX (float): DM mass in MeV
        sigmaE (float): DM-electron cross section in cm^2
        fdm (int): Form factor model (0=FDM1, 2=FDMq2)
        ne (int/list): Electron bin(s) to calculate rates for
        modulation_source (str/None): Explicit ordinary modulation source,
            "Verne" or "DaMaSCUS". Overrides legacy useVerne when supplied.
        useVerne (bool): Legacy selector; True means Verne, False means DaMaSCUS.
        calcError (str/None): Calculate error (only applies to DaMaSCUS (useVerne=False)) ('High'/'Low') (default None)
        useQCDark (bool): Use QCDark form factor (default True)
        DoScreen (bool): Include screening effects (default True)
        verbose (bool): Print debug info (default False)
        flat (bool): Return results based on shm velocity distribution without modulation (default False)
        dmRateObject: Pre-initialized DMeRate object (default None)
        summer (bool): Use summer velocity distribution (default False)
        
    Returns:
        tuple: (isoangles, rate_per_angle) where:
            isoangles: Array of isotropy angles in degrees
            rate_per_angle: 2D array of rates (angles × electron bins)
    """

    import os
    import torch
    import sys
    sys.path.append('..')

    explicit_use_verne = _ordinary_modulation_source_to_use_verne(modulation_source)
    if explicit_use_verne is not None:
        useVerne = explicit_use_verne

    if dmRateObject is not None:
        dmrates = dmRateObject
    else:
        dmrates = _create_dmrate(material, useQCDark=useQCDark)

    if useQCDark:
        integrate = True
    else:
        integrate = False

    fdm_dict = {0: "FDM1", 2: "FDMq2"}
    
    calc_method_dict = {True: "Verne", False: "DaMaSCUS"}    

    dmrates.update_crosssection(sigmaE)

    halo_model = 'modulated' if not summer else 'summer'
    summer_str = '' if not summer else '_summer'

    fdm_str = fdm_dict[fdm]
    calc_str = calc_method_dict[useVerne]
    mass_str = str(mX).replace('.','_')
    loc_dir = f'../halo_data/modulated/{fdm_str}/{calc_str}{summer_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'

    if type(ne) == int:
        ne = [ne]
    # else:
    #     loc_dir = f'./halo_data/modulated/Parameter_Scan_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'


    if os.path.isdir(loc_dir) and len(os.listdir(loc_dir)) > 0:
        dir_contents = [
            i for i in os.listdir(loc_dir)
            if i.startswith("DM_Eta_theta_") and i.endswith(".txt")
        ]
        num_angles = len(dir_contents)
        
        # isoangles = np.arange(num_angles) * (180 / num_angles)
        # if useVerne:
        isoangles = torch.linspace(0,180,num_angles)
        if verbose:
            print(f'data is generated, num_angles = {num_angles}')
        rate_per_angle = torch.zeros((num_angles,len(ne)))
        for isoangle in range(0,num_angles,1):
            try:
                if flat:
                    result = dmrates.calculate_rates(mX,'shm',fdm,ne,integrate=integrate,DoScreen=DoScreen,isoangle=None,useVerne=useVerne,calcErrors=calcError).flatten()
                else:
                    result = dmrates.calculate_rates(mX,halo_model,fdm,ne,integrate=integrate,DoScreen=DoScreen,isoangle=isoangle,useVerne=useVerne,calcErrors=calcError,modulation_source=calc_str).flatten()
                # if kgday:
                #     result*= nu.kg *nu.day
                # else:
        
                #     result*= nu.g *nu.day
            
            except ValueError:
                continue

            rate_per_angle[isoangle,:]= result
        isoangles = isoangles.cpu()
        rate_per_angle = rate_per_angle.cpu()
        return isoangles,rate_per_angle
    else:
        print('data not found')
        
        print(loc_dir)
        return


def get_srdm_modulated_rates(
    material,
    mX,
    sigmaE,
    fdm,
    ne,
    modulated_source="SRDMBeam",
    useQCDark=True,
    DoScreen=True,
    screening=None,
    variant=None,
    pair_energy=None,
    mediator_spin="vector",
    dmRateObject=None,
    base_data_dir=None,
    verbose=False,
):
    """Calculate SRDMBeam rates for each available file-facing beam ring.

    Returns ``(srdm_isoangles_deg, rate_per_angle)``. The angle values are the
    representative SRDMBeam angles from metadata, with 0 deg overhead and
    180 deg nadir. For Verne point-sampled files these are the emitted
    ``file_isoangle_deg`` nodes; for DaMaSCUS ring-averaged files they are bin
    centers. This helper is SRDMBeam-specific and does not use the ordinary
    ThetaIso() halo-wind modulation path.
    """
    import numpy as np
    import torch

    from DMeRates.srdm.flux_loader import (
        available_srdmbeam_ring_indices,
        load_srdmbeam_metadata,
    )

    metadata = load_srdmbeam_metadata(
        mX,
        sigmaE,
        fdm,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    ring_count = int(metadata["ring_count"])
    ring_indices = available_srdmbeam_ring_indices(
        mX,
        sigmaE,
        fdm,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    if not ring_indices:
        ring_indices = list(range(ring_count))

    if dmRateObject is not None:
        dmrates = dmRateObject
    else:
        dmrates = _create_dmrate(material, useQCDark=useQCDark)

    if hasattr(dmrates, "update_crosssection"):
        dmrates.update_crosssection(sigmaE)

    if isinstance(ne, (int, np.integer)):
        ne_values = [ne]
    else:
        ne_values = list(ne)

    srdm_isoangles_np = _srdmbeam_representative_angles_for_indices(metadata, ring_indices)
    angle_order = np.argsort(srdm_isoangles_np)
    srdm_isoangles_np = srdm_isoangles_np[angle_order]
    ring_indices = [ring_indices[int(i)] for i in angle_order]

    integrate = bool(useQCDark)
    srdm_isoangles_deg = torch.tensor(
        srdm_isoangles_np,
        dtype=torch.float64,
    )
    rate_rows = []
    for ring_index in ring_indices:
        if verbose:
            print(f"calculating SRDMBeam ring_index={ring_index}")
        result = dmrates.calculate_rates(
            mX,
            "srdm_modulated",
            fdm,
            ne_values,
            integrate=integrate,
            DoScreen=DoScreen,
            isoangle=int(ring_index),
            modulated_source=modulated_source,
            sigma_e=sigmaE,
            screening=screening,
            variant=variant,
            pair_energy=pair_energy,
            mediator_spin=mediator_spin,
            srdm_base_data_dir=base_data_dir,
        )
        rate_rows.append(torch.as_tensor(result).flatten().cpu())

    if rate_rows:
        rate_per_angle = torch.stack(rate_rows)
    else:
        rate_per_angle = torch.empty((0, len(ne_values)), dtype=torch.float64)
    return srdm_isoangles_deg.cpu(), rate_per_angle.cpu()


def get_srdm_daily_rates(
    material,
    mX,
    sigmaE,
    fdm,
    ne,
    location,
    date,
    cadence_minutes=10,
    **kwargs,
):
    """Return a daily SRDMBeam solar-angle time series and interpolated rates.

    Angles outside the ring grid are extrapolated by holding the nearest ring
    rate constant. This is accurate for Verne's endpoint-nudged grids (first
    ring near 0 deg, last ring near 180 deg) but may introduce ~1-2% error
    for coarse DaMaSCUS grids where the first bin center is far from 0 deg.
    """
    import numpy as np

    from modulation_study.isoangle import solar_daily_angle_series

    ring_angles, rate_per_angle = get_srdm_modulated_rates(
        material,
        mX,
        sigmaE,
        fdm,
        ne,
        **kwargs,
    )
    times, srdm_isoangles_deg = solar_daily_angle_series(
        location,
        date,
        cadence_minutes=cadence_minutes,
    )

    angle_grid = np.asarray(ring_angles, dtype=float)
    rates_grid = np.asarray(rate_per_angle, dtype=float)
    if angle_grid.size == 0:
        raise ValueError("No SRDMBeam ring rates were available for daily interpolation")
    if rates_grid.ndim == 1:
        rates_grid = rates_grid[:, None]

    order = np.argsort(angle_grid)
    angle_grid = angle_grid[order]
    rates_grid = rates_grid[order, :]
    if angle_grid.size > 1 and np.any(np.diff(angle_grid) <= 0.0):
        raise ValueError("SRDMBeam interpolation angles must be strictly increasing")

    interp_angles = angle_grid
    interp_rates = rates_grid
    if angle_grid[0] > 0.0:
        interp_angles = np.concatenate([[0.0], interp_angles])
        interp_rates = np.vstack([rates_grid[0, :], interp_rates])
    if angle_grid[-1] < 180.0:
        interp_angles = np.concatenate([interp_angles, [180.0]])
        interp_rates = np.vstack([interp_rates, rates_grid[-1, :]])

    daily_rates = np.vstack(
        [
            np.interp(
                srdm_isoangles_deg,
                interp_angles,
                interp_rates[:, col],
                left=interp_rates[0, col],
                right=interp_rates[-1, col],
            )
            for col in range(interp_rates.shape[1])
        ]
    ).T
    return times, srdm_isoangles_deg, daily_rates


def get_srdm_daily_modulation_amplitude(*args, **kwargs):
    """Return min/max and fractional daily modulation for SRDMBeam rates."""
    import numpy as np

    times, srdm_isoangles_deg, daily_rates = get_srdm_daily_rates(*args, **kwargs)
    rates = np.asarray(daily_rates, dtype=float)
    rate_min = np.min(rates, axis=0)
    rate_max = np.max(rates, axis=0)
    mean = 0.5 * (rate_max + rate_min)
    fractional = np.divide(
        rate_max - rate_min,
        mean,
        out=np.zeros_like(rate_max),
        where=mean != 0,
    )
    return {
        "times": times,
        "srdm_isoangles_deg": srdm_isoangles_deg,
        "daily_rates": rates,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "fractional_modulation": fractional,
    }

    
    
def generate_modulated_rates(material,FDMn,useQCDark = True,useVerne=True,calcError=None,doScreen=True,overwrite=False,verbose=False,save=True,summer=False,outdir= './'):
    """Generate and save modulated rate data for a range of masses and cross-sections.
    
    Args:
        material (str): Target material ('Si', 'Xe', 'Ar')
        FDMn (int): Form factor model (0=FDM1, 2=FDMq2)
        useQCDark (bool): Use QCDark form factor (default True)
        useVerne (bool): Use Verne velocity distribution (default True)
        calcError (str/None): Calculate error ('High'/'Low') (default None, only applies if UseVerne=False)
        doScreen (bool): Include screening effects (default True)
        overwrite (bool): Overwrite existing files (default False)
        verbose (bool): Print debug info (default False)
        save (bool): Save results to files (default True)
        summer (bool): Use summer velocity distribution (default False)
    """
    import csv
    import re
    import numpy as np
    import os
    import sys
    sys.path.append('..')
    from tqdm.autonotebook import tqdm
    import numericalunits as nu
    dmrates = _create_dmrate(material, useQCDark=useQCDark)




    fdm_dict = {0: "FDM1", 2: "FDMq2"}    

    calc_method_dict = {True: "Verne", False: "DaMaSCUS"}  
    summer_str = '_summer' if summer else ''
    fdm_str = fdm_dict[FDMn]
    calc_str = calc_method_dict[useVerne] + summer_str
    if summer:
        print("WARNING: You are generating rates for summer, not for March (where vE is at average). If this is not what you meant to do, pleas turn summer to False.")
        print("Using files in " + f'../halo_data/modulated/{fdm_str}/{calc_str}/')
    scr_dict = {True: "_screened", False: "_unscreened"}  

    qedict = {True: "_qcdark",False: "_qedark"}
    
    

    nes = [1,2,3,4,5,6,7,8,9,10]

    scr_str = scr_dict[doScreen] if material == 'Si' else ""
    qestr = qedict[useQCDark]
    if material != 'Si':
        qestr = ''

    



    write_dir= outdir + f'damascus_modulated_rates{scr_str}{qestr}_{material}{summer_str}'
    

    if useVerne:
        write_dir= outdir + f'verne_modulated_rates{scr_str}{qestr}_{material}{summer_str}'
        calcError=None



    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    module_dir = os.path.dirname(__file__)
    halodir = os.path.join(module_dir,f'../halo_data/modulated/{fdm_str}/{calc_str}/')

    # dir = f'../halo_data/modulated/{calc_str}_{fdm_str}/'


    directories = os.listdir(halodir)

    for celery in tqdm(range(len(directories))):
        d = directories[celery]
        if 'Store' in d:
            continue
        mass_str = re.findall('DM_.*_MeV',d)[0][3:-4]
        mX = mass_str.replace('_','.')
        mX = float(mX)

        if 'sigmaE' in d:
            sigmaE = re.findall('E_.*cm',d)[0][2:-3].replace('_','.')
            sigmaE = float(sigmaE)

        outfile = write_dir+f'/mX_{mass_str}_MeV_sigmaE_{sigmaE}_FDM{FDMn}.csv'
        if os.path.isfile(outfile) and not overwrite:
            if verbose:
                print(f'this rate is generated, continuing: {outfile}')
            continue
        if verbose:
            print(mX,sigmaE,d)

        try:
            isoangles,rate_per_angle = get_modulated_rates(material,mX,sigmaE,FDMn,nes,useVerne=useVerne,calcError=calcError,useQCDark=useQCDark,DoScreen = doScreen,verbose = verbose,flat=False, dmRateObject = dmrates,summer=summer)
        except TypeError:
            print('continuing, since data not found')
            continue
        rate_per_angle = rate_per_angle.cpu() * nu.g * nu.day
        isoangles = isoangles.cpu()

        if save:
            combined= np.vstack((isoangles,rate_per_angle.T))
            combined = combined.T
            np.savetxt(outfile,combined,delimiter=',')
                # with open(outfile,'w') as f:
                #     print(isoangles,rate_per_angle)
                #     writer = csv.writer(f,delimiter=',')
                #     writer.writerows(zip(isoangles,rate_per_angle))
                
    return

def generate_damascus_rates_with_error(ne,material,FDMn,useQCDark = True,DoScreen=True,overwrite=False,verbose=False,save=True,fit=False,summer=False):
    """Generate modulated rates with error estimates for a specific electron bin.
    
    Args:
        ne (int): Electron bin to calculate
        material (str): Target material ('Si', 'Xe', 'Ar')
        FDMn (int): Form factor model (0=FDM1, 2=FDMq2)
        useQCDark (bool): Use QCDark form factor (default True)
        DoScreen (bool): Include screening effects (default True)
        overwrite (bool): Overwrite existing files (default False)
        verbose (bool): Print debug info (default False)
        save (bool): Save results to files (default True)
        fit (bool): Fit rates to functional form (default False)
        summer (bool): Use summer velocity distribution (default False)
    """

    import csv
    import re
    import numpy as np
    import os
    import sys
    sys.path.append('..')
    from tqdm.autonotebook import tqdm
    import numericalunits as nu
    dmrates = _create_dmrate(material, useQCDark=useQCDark)




    fdm_dict = {0: "FDM1", 2: "FDMq2"}    

    scr_dict = {True: "_screened", False: "_unscreened"}  

    qedict = {True: "_qcdark",False: "_qedark"}
    
    summer_str = 'summer' if summer else ''

    scr_str = scr_dict[DoScreen] if material == 'Si' else ""
    qestr = qedict[useQCDark]
    if material != 'Si':
        qestr = ''

    fdm_str = fdm_dict[FDMn]



    write_dir= f'damascus_modulated_rates_{ne}e{scr_str}{qestr}_{material}{summer_str}'
    
    write_dir_fit= f'fitted_damascus_modulated_rates_{ne}e{scr_str}{qestr}_{material}{summer_str}'



    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    if not os.path.isdir(write_dir_fit):
        os.mkdir(write_dir_fit)

    module_dir = os.path.dirname(__file__)
    halodir = os.path.join(module_dir,f'../halo_data/modulated/{fdm_str}/DaMaSCUS{summer_str}/')


    directories = os.listdir(halodir)

    for celery in tqdm(range(len(directories))):
        d = directories[celery]
        if 'Store' in d:
            continue
        mass_str = re.findall('DM_.*_MeV',d)[0][3:-4]
        mX = mass_str.replace('_','.')
        mX = float(mX)

        if 'sigmaE' in d:
            sigmaE = re.findall('E_.*cm',d)[0][2:-3].replace('_','.')
            sigmaE = float(sigmaE)

        outfile = write_dir+f'/mX_{mass_str}_MeV_sigmaE_{sigmaE}_FDM{FDMn}.csv'
        outfile_fit = write_dir_fit+f'/mX_{mass_str}_MeV_sigmaE_{sigmaE}_FDM{FDMn}.csv'

        if os.path.isfile(outfile) and not overwrite:
            if verbose:
                print(f'this rate is generated, continuing: {outfile}')
            continue
        if verbose:
            print(mX,sigmaE,d)

        
        isoangles,rate_per_angle = get_modulated_rates(material,mX,sigmaE,FDMn,ne,useVerne=False,calcError=None,useQCDark=useQCDark,DoScreen = DoScreen,verbose = verbose,flat=False, dmRateObject = dmrates)
        isoangles,rate_per_angle_high = get_modulated_rates(material,mX,sigmaE,FDMn,ne,useVerne=False,calcError='High',useQCDark=useQCDark,DoScreen = DoScreen,verbose = verbose,flat=False, dmRateObject = dmrates)
        isoangles = isoangles.cpu().numpy()
        rate_per_angle = rate_per_angle.flatten().cpu().numpy() * nu.g * nu.day
        rate_per_angle_high = rate_per_angle_high.flatten().cpu().numpy() * nu.g * nu.day

        rate_err = rate_per_angle_high - rate_per_angle
        if fit:
            angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate_per_angle,rate_err)
            rate_fit = fit_vector[0]


        if save:
            combined= np.vstack((isoangles,rate_per_angle,rate_err))
            combined = combined.T
            np.savetxt(outfile,combined,delimiter=',')
            if fit:
                combined_fit = np.vstack((angle_grid,rate_fit))
                combined_fit = combined_fit.T
                np.savetxt(outfile_fit,combined_fit,delimiter=',')

                # with open(outfile,'w') as f:
                #     print(isoangles,rate_per_angle)
                #     writer = csv.writer(f,delimiter=',')
                #     writer.writerows(zip(isoangles,rate_per_angle))
                
    return


def to_pretty_scientific_notation(num_str):
    """Convert a number string to pretty-printed scientific notation for plots.
    
    Args:
        num_str (str/number): Input number to format
        
    Returns:
        str: Formatted string with proper LaTeX math notation
    """
    import numpy as np
    num = float(num_str)
    coeff, exp = f"{num:.2e}".split("e")
    exp = int(exp)
    coeff = float(coeff)
    if int(coeff) == 1:
        coeff = ''
    else:
        coeff = str(np.round(coeff,2)) + ' $*$ '
    # superscript = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))

    return coeff + f"$10^{{{exp}}}$"


def hyp_tan_ff(theta,a,theta_0,theta_s,ff):
    """Hyperbolic tangent function for fitting modulated rates.
    
    Args:
        theta: Angle values
        a: Amplitude parameter
        theta_0: Transition angle parameter  
        theta_s: Slope parameter
        ff: Offset parameter
        
    Returns:
        Array of function values
    """
    import numpy as np
    #rbar is mean
    #a = amplitude
    #theta is angle
    #theta_0 is transition angle
    #theta_s is slope fit
    return (a/2)*np.tanh((theta-theta_0)/theta_s) + ff

def fitted_rates(angles,rates,rates_err=None,linear=False):
    """Fit modulated rate data to functional form (hyperbolic tangent or linear).
    
    Args:
        angles: Array of angle values
        rates: Array of rate values
        rates_err: Array of rate errors (optional)
        linear: Force linear fit (default False)
        
    Returns:
        tuple: (angle_grid, fit_vector, parameters, errors) containing:
            angle_grid: Fine grid of angles
            fit_vector: Fit results and metrics
            parameters: Fit parameters
            errors: Parameter errors
    """

    import numpy as np
    from scipy.stats import linregress
    

    from scipy.optimize import curve_fit
    rbar = np.mean(rates)
    rates_to_fit = rates / rbar
    if rates_err is not None:
        rates_fit_err = rates_err/rbar
    angle_grid = np.linspace(0,180,len(angles))
    if not linear:
        if rates_err is not None:
            try:
                parameters,covariance = curve_fit(hyp_tan_ff,angles,rates_to_fit,bounds=([-np.inf,0,0,-np.inf],[0,180,np.inf,np.inf]),sigma=rates_fit_err)
            except ValueError:
                parameters,covariance = curve_fit(hyp_tan_ff,angles,rates_to_fit,bounds=([-np.inf,0,0,-np.inf],[0,180,np.inf,np.inf]))
        else:
            parameters,covariance = curve_fit(hyp_tan_ff,angles,rates_to_fit,bounds=([-np.inf,0,0,-np.inf],[0,180,np.inf,np.inf]))
            
        amplitude = parameters[0]
        inflection = parameters[1]
        slope_angle = parameters[2]
        shift = parameters[3]
        errors = np.sqrt(np.diag(covariance))
        fit = (hyp_tan_ff(angle_grid,amplitude,inflection,slope_angle,shift))*rbar
        fit_upper = (hyp_tan_ff(angle_grid,amplitude-errors[0],inflection,slope_angle+errors[2],shift+errors[3]))*rbar 
        fit_lower = (hyp_tan_ff(angle_grid,amplitude+errors[0],inflection,slope_angle-errors[2],shift-errors[3]))*rbar 

        


        result = linregress(angles,rates)
        slope = result.slope
        intercept = result.intercept
        r = result.rvalue
        p = result.pvalue
        std_err = result.stderr
        intercept_stderr = result.intercept_stderr 

        linear_fit = slope*angle_grid + intercept
        linear_fit_upper = (slope + std_err)*angle_grid + intercept + intercept_stderr
        linear_fit_lower = (slope - std_err)*angle_grid + intercept - intercept_stderr


        mse_sigmoid = np.mean((rates - fit)**2)
        rmse_sigmoid = np.sqrt(mse_sigmoid)
        ssr_sigmoid =  ((rates - fit) ** 2).sum()
        
        mse_linear = np.mean((rates - linear_fit)**2)
        rmse_linear = np.sqrt(mse_linear)
        ssr_linear =  ((rates - linear_fit) ** 2).sum()


        # print(f'Sigmoid Fit RMSE: {rmse_sigmoid}')
        # print(f'Linear Fit RMSE: {rmse_linear}')

        # print(f'Sigmoid Fit SSR: {ssr_sigmoid}')
        # print(f'Linear Fit SSR: {ssr_linear}')

        # if rmse_linear < rmse_sigmoid and ssr_linear < ssr_sigmoid:
        #     fit_vector = [linear_fit,linear_fit_upper,linear_fit_lower,mse_linear,rmse_linear,ssr_linear,"Linear"]
        # else:
        fit_vector = [fit,fit_upper,fit_lower,mse_sigmoid,rmse_sigmoid,ssr_sigmoid,"Sigmoid"]
    if linear:
        #fit failed, do linear regression
        
        # from scipy.interpolate import PchipInterpolator


        result = linregress(angles,rates)
        slope = result.slope
        intercept = result.intercept
        r = result.rvalue
        p = result.pvalue
        std_err = result.stderr
        intercept_stderr = result.intercept_stderr 
    
        fit = slope*angle_grid + intercept
        fit_upper = (slope + std_err)*angle_grid + intercept + intercept_stderr
        fit_lower = (slope - std_err)*angle_grid + intercept - intercept_stderr

        mse = np.mean((rates - fit)**2)
        rmse = np.sqrt(mse)
        ssr =  ((rates - fit) ** 2).sum()


        parameters = [slope,intercept,r]
        errors = [p,std_err,intercept_stderr]
        fit_vector = [fit,fit_upper,fit_lower,mse,rmse,ssr]




        # rate_interp = PchipInterpolator(angles,rates)
        # fit = rate_interp(angle_grid)
        # parameters = [False,False,False,False]
        # errors = [0,0,0,0]
        # inflection = 0 #default, not actually real
        # inflection_err =180




    return angle_grid,fit_vector,parameters,errors







def plot_damascus_output(test_mX,FDMn,cross_section,long=True,savefig=False,cmap_name='viridis'):
    """Plot eta(vmin) for different angles from DaMaSCUS output (for demonstration purposes).
    
    Args:
        test_mX: DM mass in MeV
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        cross_section: Cross section in cm^2
        long: Use long simulation data (180 points instead of 36) (default True)
        savefig: Save figure to file (default False)
        cmap_name: Colormap name (default 'viridis')
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    set_default_plotting_params()

    plt.figure()

    fdm_dict = {0: "FDM1", 2: "FDMq2"}    
    mediator = fdm_dict[FDMn]

    mX_str = float(test_mX)
    mX_str = np.round(mX_str,3)

    mX_str = str(mX_str)
    mX_str = mX_str.replace('.',"_")
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate



    dmrates = DMeRate.DMeRate('Si')

   
    dmrates.update_params(220,232,544,0.3e9,1e-36)
    vhigh = 3*(dmrates.vEarth + dmrates.vEscape)
    vMins = np.linspace(0,vhigh,1000)

    
    shm_etas = []
    for v in vMins:
        shm_eta = dmrates.DM_Halo.etaSHM(v)
        shm_etas.append(shm_eta)
    shm_etas = np.array(shm_etas)
    shm_etas /= (nu.s / nu.km) #in s/km
    vMins/=(nu.km/nu.s)

    cmap = plt.get_cmap(cmap_name, 180) 
    if long:
        long_str = '_long'
    else:
        long_str = ''
    halo_dir = f'../halo_data/modulated/{mediator}/DaMaSCUS{long_str}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2{long_str}/'
    steps = len([f for f in os.listdir(halo_dir) if f.endswith('.txt')])
    actual_angle = np.linspace(0,180,steps)
    for isoangle in range(steps):
        ai = actual_angle[isoangle]
        ai = round(ai)
        # print(isoangle,ai,cmap(ai))

        fname = halo_dir + f'DM_Eta_theta_{isoangle}.txt'
        # fname_DAMASCUS = f'./DaMaSCUS/results/5MeV_test_histograms/eta.{isoangle}'
        fdata = np.loadtxt(fname,delimiter='\t')
        vmin = fdata[:,0]
        eta = fdata[:,1]

        plt.plot(vmin,eta,color=cmap(ai))


    
    plt.xlim([0, 700])

    ax = plt.gca()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=180) 

    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm) 

    plt.plot(vMins,shm_etas,linewidth=4,ls=':',color='black',label='SHM')
    plt.legend(prop={'size': 32},loc=3)
    plt.xlabel('$v_{\mathrm{min}}$ [km/s]')
    plt.ylabel('$\eta$ [s/km]')
    ticks = np.linspace(0,180,19)[::2]
    clb = plt.colorbar(sm,ax=ax,ticks=ticks)
    clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)


    if FDMn == 2:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    else:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    cs_str = r'${} \times 10^{{{}}}$'.format(*str(cross_section).split('e')) + 'cm$^2$'
    plt.text(0.99,0.87,'$\overline{\sigma}_{e} =$ ' + cs_str,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.80,'$m_\chi=$ ' + f'{test_mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)



    if savefig:
        plt.savefig(f'figures/Misc/Eta_{test_mX}MeV_{cross_section}sigmaE_FDM{FDMn}.pdf')
    plt.show()
    plt.close
    return



def plot_damascus_figure(test_mX,cross_section,long=True,savefig=False,cmap_name='viridis'):
    """Comparative plot of eta(vmin) for both form factors.
    
    Args:
        test_mX: DM mass in MeV
        cross_section: Cross section in cm^2
        long: Use long simulation data (default True)
        savefig: Save figure to file (default False)
        cmap_name: Colormap name (default 'viridis')
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    set_default_plotting_params()



    
    fig,ax = plt.subplots(1,2,figsize=(24,10))
    # mediator = "LM"
    fdm_dict = {0: "FDM1", 2: "FDMq2"}    
    

    mX_str = float(test_mX)
    mX_str = np.round(mX_str,3)

    mX_str = str(mX_str)
    mX_str = mX_str.replace('.',"_")
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate



    dmrates = DMeRate.DMeRate('Si')

   
    dmrates.update_params(220,232,544,0.3e9,1e-36)
    vhigh = 3*(dmrates.vEarth + dmrates.vEscape)
    vMins = np.linspace(0,vhigh,1000)

    
    shm_etas = []
    for v in vMins:
        shm_eta = dmrates.DM_Halo.etaSHM(v)
        shm_etas.append(shm_eta)
    shm_etas = np.array(shm_etas)
    shm_etas /= (nu.s / nu.km) #in s/km
    vMins/=(nu.km/nu.s)

    cmap = plt.get_cmap(cmap_name, 180) 
    if long:
        long_str = '_long'
        dirend = long_str
    else:
        long_str = ''
        dirend = long_str

    for i,FDMn in enumerate([0,2]):
        mediator = fdm_dict[FDMn]
        halo_dir = f'../halo_data/modulated/{mediator}/DaMaSCUS{dirend}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2{long_str}/'
        steps = len([f for f in os.listdir(halo_dir) if f.endswith('.txt')])
        actual_angle = np.linspace(0,180,steps)
        for isoangle in range(steps):
            ai = actual_angle[isoangle]
            ai = round(ai)
            # print(isoangle,ai,cmap(ai))

            fname = halo_dir + f'DM_Eta_theta_{isoangle}.txt'
            # fname_DAMASCUS = f'./DaMaSCUS/results/5MeV_test_histograms/eta.{isoangle}'
            fdata = np.loadtxt(fname,delimiter='\t')
            vmin = fdata[:,0]
            eta = fdata[:,1]

            ax[i].plot(vmin,eta,color=cmap(ai))


        
        ax[i].set_xlim([0, 700])


        norm = matplotlib.colors.Normalize(vmin=0, vmax=180) 

        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm) 

        ax[i].plot(vMins,shm_etas,linewidth=4,ls=':',color='black',label='SHM')
        ax[i].legend(prop={'size': 32},loc=3)
        ax[i].set_xlabel('$v_{\mathrm{min}}$ [km/s]')
        ax[i].set_ylabel('$\eta$ [s/km]')
        ticks = np.linspace(0,180,19)[::2]
        clb = plt.colorbar(sm,ax=ax[i],ticks=ticks)
        clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)


        if FDMn == 2:
            ax[i].text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[i].transAxes)
        else:
            ax[i].text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[i].transAxes)
        cs_str = r'${} \times 10^{{{}}}$'.format(*str(cross_section).split('e')) + 'cm$^2$'
        ax[i].text(0.99,0.86,'$\overline{\sigma}_{e} =$ ' + cs_str,color='black',horizontalalignment='right',verticalalignment='center',transform = ax[i].transAxes)
        ax[i].text(0.99,0.77,'$m_\chi=$ ' + f'{test_mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[i].transAxes)



    if savefig:
        plt.savefig(f'figures/Misc/DaMaSCUS_Eta_{test_mX}MeV_{cross_section}sigmaE.pdf')
    plt.tight_layout()
    plt.show()
    plt.close
    return



def mu_Xe(mX):
    """Calculate reduced mass for DM-electron system.
    
    Args:
        mX: DM mass in MeV
        
    Returns:
        Reduced mass in eV
    """
    me_eV = 5.1099894e5
    """
    DM-electron reduced mass
    """
    return mX*me_eV/(mX+me_eV)

def mu_XP(mX):
    """Calculate reduced mass for DM-proton system.
    
    Args:
        mX: DM mass in MeV
        
    Returns:
        Reduced mass in eV
    """
    mP_eV = 938.27208816 *1e6
    return mX*mP_eV/(mX+mP_eV)

def sigmaE_to_sigmaP(sigmaE,mX):
    """Convert DM-electron to DM-proton cross section.
    
    Args:
        sigmaE: DM-electron cross section in cm^2
        mX: DM mass in MeV
        
    Returns:
        DM-proton cross section in cm^2
    """
    import numpy as np
    mX*=1e6 #eV
    sigmaP = sigmaE*(mu_XP(mX)/mu_Xe(mX))**2
    # sigmaP = np.round(sigmaP,3)
    return sigmaP

def sigmaP_to_sigmaE(sigmaP,mX):
    """Convert DM-proton to DM-electron cross section.
    
    Args:
        sigmaP: DM-proton cross section in cm^2
        mX: DM mass in MeV
        
    Returns:
        DM-electron cross section in cm^2
    """

    import numpy as np
    mX*=1e6 #eV
    sigmaE = sigmaP*(mu_Xe(mX)/mu_XP(mX))**2
    # sigmaP = np.round(sigmaP,3)
    return sigmaE

def get_damascus_output(mX,sigmaE,FDMn):
    """Load eta(vmin) data from DaMaSCUS output files.
    
    Args:
        mX: DM mass in MeV
        sigmaE: Cross section in cm^2
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        
    Returns:
        List of (vmin, eta) arrays for each angle
    """

    import numpy as np
    import os
    if FDMn == 0:
        dir_stir = 'FDM1'
    else:
        dir_stir = 'FDMq2'
    # mX= np.round(float(mX),2)
    mX_str = str(mX).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    ddir = f'halo_data/modulated/{dir_stir}/DaMaSCUS/mDM_{mX_str}_MeV_sigmaE_{sigmaE}_cm2/'
    data = []
    num_angles = len([f for f in os.listdir(ddir) if f.endswith('.txt')])
    for i in range(num_angles):
        file = ddir + f'DM_Eta_theta_{i}.txt'
        filedata = np.loadtxt(file,delimiter='\t')
        file_vmin = filedata[:,0]
        file_eta = filedata[:,1]
        data.append([file_vmin,file_eta])
    return data

def get_angle_limits(loc, date=None):
    """Calculate min/max isotropy angles for a location on given date.

    Args:
        loc: Location name ('SNOLAB', 'Bariloche', etc.)
        date: [day, month, year] (default [8, 8, 2024])

    Returns:
        tuple: (min_angle, max_angle) in degrees
    """

    import numpy as np
    from scipy.interpolate import CubicSpline
    from isoangle import (
        ThetaIso,
        FracDays,
        SITE_ALIASES,
        SITE_COORDS,
        normalize_site_key,
        normalize_thetaiso_loc,
    )

    if date is None:
        date = [8, 8, 2024]

    if isinstance(loc, str):
        loc_key = normalize_site_key(loc)
        if loc_key not in SITE_COORDS:
            valid_aliases = sorted(SITE_ALIASES.keys())
            valid_keys = sorted(SITE_COORDS.keys())
            raise ValueError(
                f"Unknown location '{loc}'. "
                f"Valid aliases: {valid_aliases}. "
                f"Valid site keys: {valid_keys}."
            )
        site = normalize_thetaiso_loc(loc_key)
    else:
        site = normalize_thetaiso_loc(loc)

    # Include h=24 so the spline is not extrapolating at the upper endpoint.
    hours = np.arange(25)

    nlist = [
        FracDays(np.array(date), np.array([h, 0, 0]))
        for h in hours
    ]

    y = np.array([
        np.rad2deg(ThetaIso(site, n))
        for n in nlist
    ])

    xnew = np.linspace(0, 24, num=1000)
    spl = CubicSpline(hours, y)
    ynew = spl(xnew)

    min_angle = np.min(ynew)
    max_angle = np.max(ynew)

    return min_angle, max_angle

def get_amplitude(mX,sigmaE,FDMn,material,min_angle,max_angle,ne=1,fractional=False,useVerne=False,verbose=False,fromFile=False,returnaverage=False,useQCDark=True,fit=None,summer=False):
    """Calculate modulation amplitude between min/max angles.
    
    Args:
        mX: DM mass in MeV
        sigmaE: Cross section in cm^2
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        material: Target material
        min_angle: Minimum isotropy angle in degrees
        max_angle: Maximum isotropy angle in degrees
        ne: Electron bin (default 1)
        fractional: Return fractional amplitude (default False)
        useVerne: Use Verne distribution (default False)
        verbose: Print debug info (default False)
        fromFile: Load rates from file (default False)
        returnaverage: Return average rate instead of amplitude (default False)
        useQCDark: Use QCDark form factor (default True)
        fit: Fit rates (default None=auto)
        summer: Use summer distribution (default False)
        
    Returns:
        Modulation amplitude (or fractional amplitude if specified)
    """

    if fit is None:
        if useVerne:
            fit = False
        else:
            fit = True

    import numpy as np
    # try:

    qedict = {True: "_qcdark",False: "_qedark"}

    qestr = qedict[useQCDark] if material == 'Si' else ""

    if fromFile:
        if useVerne:
            type_str = 'verne'
        else:
            type_str = 'damascus'
        # mass_str = str(np.round(mX,2)).replace('.','_')
        mass_str = str(mX).replace('.','_')
        screenstr = '_screened' if material == 'Si' else ""
        summerstr = '_summer' if summer else ''

        file = f'./{type_str}_modulated_rates{screenstr}{qestr}_{material}{summerstr}/mX_{mass_str}_MeV_sigmaE_{sigmaE}_FDM{FDMn}.csv'
        

        fdata = np.loadtxt(file,delimiter=',')
        try:
            isoangles = fdata[:,0]
            rate = fdata[:,ne]/ nu.g / nu.day
            # if kgday:
            #     rate *=1000

        except IndexError:
            print(mX,sigmaE,FDMn,material)
            raise IndexError("Something wrong with this file, perhaps it needs a redo?")

        


    else:
        isoangles,rate = get_modulated_rates(material,mX,sigmaE,FDMn,ne=ne,useVerne=useVerne,calcError=None,useQCDark=useQCDark,summer=summer)
        if not useVerne:
            isoangles_h,rate_high = get_modulated_rates(material,mX,sigmaE,FDMn,ne=ne,useVerne=useVerne,calcError='High',summer=summer)
                
                

        if not useVerne:
            rate_err = rate_high - rate



    if fit:
        fitFailed_w_errors = False
        fitFailed = False
        if np.sum(rate) == 0:
            return 0
        if useVerne:
            try:
                angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=None)
            except:
                #fitfailed
                fitFailed = True
        else:
            # try:
            #     angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=rate_err)
                
            # except:
            try:
                angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=None)
                fitFailed_w_errors = True
            except:
                fitFailed = True
                

        if fitFailed:
            print(f'Warning, fit failed for this point mX = {mX} sigmaE = {sigmaE}')
            return np.nan

       

        lab_angles = np.linspace(min_angle,max_angle,100)
        try:
            lab_rate = hyp_tan_ff(lab_angles,*parameters)*np.mean(rate)
        except:
            print('fit returned a linear fit i think')
            return np.nan

    else: # interpolate
        import numpy as np
        lab_angles = np.linspace(min_angle,max_angle,100)

        lab_rate = np.interp(lab_angles,isoangles,rate)



    amplitude = (np.max(lab_rate) - np.min(lab_rate))/ 2
    average = np.mean(lab_rate)
    fractional_amplitude = amplitude / average
    


    if fractional:
        if average == 0:
            fractional_amplitude = 0
        if fractional_amplitude < 0:
            fractional_amplitude = 0
        return fractional_amplitude
    elif returnaverage:
        return average
    else:
        return amplitude
    

def plot_modulation_ne_bins(mX1,mX2,sigmaE1,sigmaE2,material,FDMn,location1='SNOLAB',location2='SUPL',fractional=True,useVerne=False,verbose=False,fromFile=False,nes=[1,2,3,4,5,6,7,8,9,10],save=False,ybounds=None,useQCDark = True):
    """Plot modulation amplitudes across electron bins for two parameter sets.
    
    Args:
        mX1, mX2: DM masses in MeV to compare
        sigmaE1, sigmaE2: Cross sections in cm^2 to compare  
        material: Target material ('Si', 'Xe', 'Ar')
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        location1, location2: Locations to compare (default 'SNOLAB' vs 'SUPL')
        fractional: Plot fractional amplitudes (default True)
        useVerne: Use Verne distribution (default False)
        verbose: Print debug info (default False) 
        fromFile: Load rates from file (default False)
        nes: List of electron bins to plot (default 1-10)
        save: Save figure to file (default False)
        ybounds: Custom y-axis bounds (default None)
        useQCDark: Use form factor from QCDark (default True)
    """

    qedict = {True: "_qcdark",False: "_qedark"}
    matnamedict = {
        'Si': "Silicon",
        'Xe': "Xenon",
        'Ar': 'Argon',
    }
    qestr = qedict[useQCDark]

    min_angle_1,max_angle_1 = get_angle_limits(location1)
    min_angle_2,max_angle_2 = get_angle_limits(location2)
    import numpy as np
    frac_amps_pt1_loc1 = np.zeros(len(nes))
    frac_amps_pt1_loc2 = np.zeros(len(nes))
    frac_amps_pt2_loc1 = np.zeros(len(nes))
    frac_amps_pt2_loc2 = np.zeros(len(nes))
    for i,ne in enumerate(nes):
        frac_amps_pt1_loc1[i] = get_amplitude(mX1,sigmaE1,FDMn,material,min_angle_1,max_angle_1,ne=int(ne),fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,useQCDark=useQCDark)
        frac_amps_pt1_loc2[i] = get_amplitude(mX1,sigmaE1,FDMn,material,min_angle_2,max_angle_2,ne=int(ne),fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,useQCDark=useQCDark)

        frac_amps_pt2_loc1[i] = get_amplitude(mX2,sigmaE2,FDMn,material,min_angle_1,max_angle_1,ne=int(ne),fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,useQCDark=useQCDark)
        frac_amps_pt2_loc2[i] = get_amplitude(mX2,sigmaE2,FDMn,material,min_angle_2,max_angle_2,ne=int(ne),fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,useQCDark=useQCDark)


    # plotting specifications
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    large = 28
    small = 18
    medium = 22
    set_default_plotting_params(fontsize=large)
    #Options
    
    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue

    plot_nes = np.arange(1,len(nes) +2) - 0.5

    xticks = np.arange(1,len(nes)+1)
    fig = plt.figure(layout='constrained',figsize=(9,8))
    ax = plt.gca()
    plt.xlabel("Q")
    
    colorlist = [northcolor,southcolor]
    plt.stairs(frac_amps_pt1_loc1,plot_nes,color=colorlist[0],lw=3)
    plt.stairs(frac_amps_pt1_loc2,plot_nes,color=colorlist[1],lw=3)
    plt.stairs(frac_amps_pt2_loc1,plot_nes,ls='--',color=colorlist[0],lw=3)
    plt.stairs(frac_amps_pt2_loc2,plot_nes,ls='--',color=colorlist[1],lw=3)

    if FDMn == 0:
        fdm_str = '$F_{\mathrm{DM}} = 1$'
    else:
        fdm_str = '$F_{\mathrm{DM}} = (\\alpha m_e/q)^2$'

    if (FDMn ==0) and (material == 'Xe' or material == 'Ar'):
        x = 0.96
        y = 0.88
        # y =0.96
    elif ((FDMn == 0) and material == 'Si' and not useVerne):
        x = 0.96
        y = 0.88


    else:
        x = 0.96
        y = 0.96
        
    plt.text(x, y, material,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c='Black',fontsize=large)

    plt.text(x, y-0.06, f'{fdm_str}',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c='Black',fontsize=medium)

    sigmaE1_print = to_pretty_scientific_notation(sigmaE1)
    sigmaE2_print = to_pretty_scientific_notation(sigmaE2)

    firstPointStr = '$m_\chi = $' + f'{mX1}  MeV' + ' $\overline{\sigma}_e =$ ' + f'{sigmaE1_print}' + ' cm$^2$'
    secondPointStr = '$m_\chi = $' + f'{mX2} MeV' + ' $\overline{\sigma}_e =$ ' + f'{sigmaE2_print}' + ' cm$^2$'
    plt.text(x, y-0.06-0.06, location1,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c=colorlist[0],fontsize=medium)
    plt.text(x, y-0.06-0.06-0.06, location2,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c=colorlist[1],fontsize=medium)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='-',lw=3),
                    Line2D([0], [0], color='black', linestyle='--',lw=3)]

    if material == 'Si':
        if FDMn == 0:
            xshift = 0.5
        else:
            xshift = 0.5
    else:
        if FDMn == 0:
            xshift = 0.52
        else:
            xshift = 0.55
    legend = plt.legend(custom_lines, [firstPointStr,secondPointStr],loc=(x-xshift,y-0.33),fontsize=small,frameon=False,framealpha=0)
    legend.get_texts()[0].set_horizontalalignment('right')
    legend.get_texts()[1].set_horizontalalignment('right')
    plt.xticks(xticks)
    plt.xlim(np.min(xticks)-0.5,np.max(nes)+0.5)
    if useVerne:
        titlestr = 'Verne'
    else:
         titlestr = 'DaMaSCUS'
    if fractional:
        plt.title(f'Fractional Modulation Comparison')
        plt.ylabel("$f_{\mathrm{mod}}$")
    else:
        plt.title(f'{titlestr} Modulation Amplitude Comparison',fontsize=large)
        plt.ylabel("Amplitude [events/g/day]")
        plt.yscale('log')

    if ybounds is not None:
        plt.ylim(ybounds[0],ybounds[1])
    
    if save:
        savedir = f'figures/{matnamedict[material]}/'
        if fractional:
            frac_str = 'fractional_'
        else:
            frac_str = ''
        if useVerne:
            verne_str=  'verne'
        else:
            verne_str= 'damascus'
        name = f'{frac_str}modulation_amp_ne_bins_{location1}_vs_{location2}_FDM{FDMn}_{verne_str}{qestr}.pdf'
        plt.savefig(savedir+name,bbox_inches='tight')
    plt.show()

    plt.close()

    


def getModulationAmplitudes(material,FDMn,location,fractional=False,useVerne=True,fromFile=True,verbose=False,ne=1,returnaverage=False,useQCDark=True,summer=False):
    """Get modulation amplitudes for all available masses/cross-sections.
    
    Args:
        material: Target material ('Si', 'Xe', 'Ar')
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        location: Location name ('SNOLAB', etc.)
        fractional: Return fractional amplitudes (default False)
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        verbose: Print debug info (default False)
        ne: Electron bin (default 1)
        returnaverage: Return average rate instead of amplitude (default False)
        useQCDark: Use form factor from QCDark (default True)
        summer: Use summer distribution (default False)
        
    Returns:
        tuple: (masses, sigmaEs, amplitudes) arrays for all available data
    """

    from tqdm.autonotebook import tqdm
    import re
    import numpy as np
    import os
    
    min_angle,max_angle = get_angle_limits(location)
    print(f"Angle Limits for {location}: {min_angle,max_angle}")
    calc_method_dict = {True: "verne", False: "damascus"}   

    qedict = {True: "_qcdark",False: "_qedark"}

    qestr = qedict[useQCDark] if material == 'Si' else ""

    halo_type = calc_method_dict[useVerne]

    screenstr = '_screened' if material == 'Si' else ""
    summerstr = '_summer' if summer else ''
    
    halo_dir = f'./{halo_type}_modulated_rates{screenstr}{qestr}_{material}{summerstr}/'


    amplitudes = []
    masses = []
    sigmaEs = []
    file_list = os.listdir(halo_dir)
    for f in tqdm(range(len(file_list)),desc="Fetching Modulation Data"):
        file = file_list[f]
        if 'mX' not in file:
            continue
        mass_str = re.findall('mX_.+MeV',file)[0][3:-4]
        mX = float(mass_str.replace('_','.'))

        sigmaE = re.findall('sigmaE_.+_FD',file)[0][7:-3]

        sigmaE = float(sigmaE)
        Fdm= int(re.findall('FDM.+.csv',file)[0][3:-4])
        if Fdm != FDMn:
            continue

        amp = get_amplitude(mX,sigmaE,FDMn,material,min_angle,max_angle,fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,ne=ne,returnaverage=returnaverage,useQCDark=useQCDark,summer=summer)


        amplitudes.append(amp)
        sigmaEs.append(sigmaE)
        masses.append(mX)


    sigmaEs = np.array(sigmaEs)
    masses = np.array(masses)

    amplitudes = np.array(amplitudes)

    return masses,sigmaEs,amplitudes



def getContourData(material,FDMn,location,fractional=False,useVerne=True,fromFile=True,verbose=False,getAll=True,masses=None,sigmaEs=None,ne=1,returnaverage=False,useQCDark=True,unitize=False,summer=False):
    """Prepare modulation data for contour plotting on mass-cross section grid.
    
    Args:
        material: Target material ('Si', 'Xe', 'Ar')
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        location: Location name ('SNOLAB', etc.)
        fractional: Use fractional amplitudes (default False)
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        verbose: Print debug info (default False)
        getAll: Get all available data (default True)
        masses: Custom mass grid (default None)
        sigmaEs: Custom cross section grid (default None)
        ne: Electron bin (default 1)
        returnaverage: Return average rates (default False)
        useQCDark: Use form factor from QCDark (default True)
        unitize: Convert units to kg*day (default False)
        summer: Use summer distribution (default False)
        
    Returns:
        tuple: (mass_grid, cs_grid, amplitude_grid) interpolated data
    """
    import numpy as np
    from scipy.interpolate import griddata
    masses,cross_sections,amplitudes = getModulationAmplitudes(material,FDMn,location,fractional=fractional,useVerne=useVerne,fromFile=fromFile,verbose=verbose,ne=ne,returnaverage=returnaverage,useQCDark=useQCDark,summer=summer)
    log_masses = np.log10(masses)
    
    log_cross_sections = np.log10(cross_sections)
    log_mass_grid = np.linspace(log_masses.min(), log_masses.max(), 1000)
    log_cs_grid = np.linspace(log_cross_sections.min(), log_cross_sections.max(), 1000)

    mass_grid = 10**log_mass_grid
    cs_grid = 10**log_cs_grid

    log_mass_grid, log_cs_grid = np.meshgrid(log_mass_grid, log_cs_grid)

    amplitude_grid = griddata(
    points=(log_masses, log_cross_sections),
    values=amplitudes,
    xi=(log_mass_grid, log_cs_grid),
    method='linear'
)


    if unitize:
        #change units here if you want it in a different unit
        amplitude_grid *= nu.kg * nu.day
        
    return mass_grid,cs_grid,amplitude_grid
    
   

                    
def find_exp(number) -> int:
    """Find base 10 exponent of a number.
    
    Args:
        number: Input number
        
    Returns:
        int: Floor of base 10 logarithm
    """
    from math import log10, floor
    base10 = log10(number)
    return floor(base10)




      
    
def modify_colormap(cmap_name,divisor=2, white_at_bottom=True):
    
    """Modify colormap by replacing part with white.
    
    Args:
        cmap_name: Name of colormap to modify
        divisor: Fraction of colormap to replace (default 2)
        white_at_bottom: Replace bottom (True) or top (False) (default True)
        
    Returns:
        Modified colormap
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    cmap = plt.cm.get_cmap(cmap_name)
    cmap_list = cmap(np.linspace(0, 1, cmap.N))
    
    if white_at_bottom:
        cmap_list[:cmap.N//divisor, :] = [1, 1, 1, 1]  # Set bottom half to white
    else:
         cmap_list[cmap.N//divisor:, :] = [1, 1, 1, 1]  # Set top half to white

    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'modified_{cmap_name}', cmap_list)
    return new_cmap



                    

def plotMaterialSignifianceFigure(loc,material='Si',plotConstraints=True,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs=None,ne=1,shadeMFP=True,savefig=False,standardizeGrid = False,useQCDark=True,showProjection=False):
    """Create multi-panel figure showing modulation significance.
    
    Args:
        loc: Location name ('SNOLAB', etc.)
        material: Target material (default 'Si')
        plotConstraints: Plot experimental constraints (default True)
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        verbose: Print debug info (default False)
        masses: Custom mass grid (default None)
        sigmaEs: Custom cross section grid (default None)
        ne: Electron bin (default 1)
        shadeMFP: Shade mean free path region (default True)
        savefig: Save figure to file (default False)
        standardizeGrid: Use standardized grid (default False)
        useQCDark: Use form factor from QCDark (default True)
        showProjection: Show experimental projections (default False)
    """
    from tqdm.autonotebook import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors

    #Options

    large = 48
    small = 36
    medium = 40
    smaller = 30
    smallest=16
    set_default_plotting_params(fontsize=medium)
   
    nrows = 3
    ncols = 2

    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,layout='constrained',figsize=(26,26))
    ebinstr = f'{ne}' + '$e^-$ bin'
    
    matdit = {
        'Si': 'Silicon',
        "Xe": 'Xenon',
        'Ar': "Argon"
    }
    fig.suptitle(f"{matdit[material]} Sensitivity at {loc}",fontsize=large,y=1.03)


    temp_amps = []

    masses_list = []
    cs_list = []
    famp_list = []
    amp_list = []
    if masses is None and sigmaEs is None:
        getAll = True
    else:
        getAll = False


       

    for fdm in [0,2]:

        Masses,CrossSections,FractionalAmplitudes = getContourData(material,fdm,loc,fractional=True,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,useQCDark=useQCDark)
        Masses,CrossSections,Amplitudes = getContourData(material,fdm,loc,fractional=False,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,returnaverage=True,useQCDark=useQCDark)
        FractionalAmplitudes[np.isnan(FractionalAmplitudes)] = 0

        masses_list.append(Masses)
        cs_list.append(CrossSections)
        amp_list.append(Amplitudes)
        famp_list.append(FractionalAmplitudes)

        


    exposure_dict = {
        'Si': np.array([1 * nu.kg * nu.day,30 * nu.kg * nu.day,30 * nu.kg * nu.year]), #kg day, kg month, 30 kg year
        'Xe': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day ,1 * nu.tonne * nu.year]),#tonne day, tonne month, 1 tonne year
        'Ar': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day, 1 * nu.tonne * nu.year])#tonne day, tonne month, ~17.4 tonne year
    }

    time_units = {
        'Si': [nu.kg*nu.day,nu.kg*nu.day,nu.kg*nu.year],
        'Xe': [nu.tonne*nu.day,nu.tonne*nu.day,nu.tonne*nu.year],
        'Ar': [nu.tonne*nu.day,nu.tonne*nu.day,nu.tonne*nu.year],

    }
    time_unit_strs = ['day','month','year']
    exposures = exposure_dict[material]
    for i in range(nrows):
        exposure = exposures[i]
        for j in range(ncols):
            if j == 0 or j==2:
               first_index = 0 #fdm-
               fdm = 0
               fdmstr = '$F_{\mathrm{DM}} = 1$'
            else:
                first_index = 1 #SUPL
                fdm = 2
                fdmstr = '$F_{\mathrm{DM}} = (\\alpha m_e/q)^2$'
           
            current_ax = axes[i,j]

           
           
            if material == 'Si':
                if ne == 1:
                    pix_1e = 1.39e-5 #e^-/pix/day
                    background_rate = pix_1e / (3.485*1e-7) #e- /gram/day
                    background_rate = background_rate / nu.g / nu.day
                    # background_rate *= 1000 #e / kg/ day
                elif ne == 2:
                    # #snolab 2e rate
                    exp_2e =46.61  * nu.g * nu.day
                    # exp_2e /=1000 #kg days
                    counts_2e = 55
                    background_rate = counts_2e / exp_2e #e / kg /day
                else:
                    background_rate = 0

            elif material == 'Xe':
                #taken from https://arxiv.org/pdf/2411.15289
                if ne == 1:
                    background_rate = 3  / nu.kg / nu.day 
                elif ne == 2:
                    background_rate = 0.1 / nu.kg / nu.day 
                elif ne == 3:
                    background_rate = 0.02 / nu.kg / nu.day 
                elif ne == 4:
                    background_rate = 0.01 / nu.kg / nu.day 


            elif material == 'Ar':
                #values from https://arxiv.org/pdf/2407.05813
                argon_2e_background= 0.1 #events / 0.25 *kg / day
                argon_3e_background= 5e-3 #events / 0.25 *kg / day
                argon_4e_background= 1e-3 #events / 0.25 *kg / day
                if ne == 1:
                    raise ValueError('No 1e background rate for Argon')
                elif ne == 2:
                    background_rate = argon_2e_background/0.25 / nu.kg / nu.day 
                elif ne == 3:
                    background_rate = argon_3e_background/0.25 / nu.kg / nu.day 
                elif ne == 4:
                    background_rate = argon_4e_background/0.25 / nu.kg / nu.day 



            FractionalAmplitudes = famp_list[first_index]#[second_index]
            Amplitudes = amp_list[first_index]#[second_index]

            Significance = (FractionalAmplitudes*Amplitudes)*exposure / np.sqrt((Amplitudes + background_rate)*exposure)

            Significance[np.isnan(Significance)] = 0

            Significance[Significance > 5] = 5.5

            Amplitudes = Significance

            if masses is not None and sigmaEs is not None:
                Amplitudes = Amplitudes.T

            temp_amps.append(np.nanmax(Amplitudes))

            current_ax.set_xscale('log')
            current_ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            current_ax.set_yscale('log')
            if masses is not None:
                mass_low = np.min(masses)
                mass_high = np.max(masses)
            else:
                mass_low = np.min(Masses)
                mass_high = 1000 #np.max(Masses)
            cs_low = np.min(CrossSections)
            cs_high = np.max(CrossSections)
            if fdm == 2:
                cs_low = 1e-40
                cs_high = 1e-30

            xlow = mass_low
            xhigh = mass_high
            ylow = cs_low
            yhigh = cs_high
            if material == 'Si':
                if fdm == 0:
                    xy = (40,2e-41)
                    xysolar = (1.5,3e-37)
                    if ne == 1 or ne ==2:
                        xlow = mass_low
                        xhigh = 10
                        yhigh = 1e-35
                        ylow = 1e-38
                        # xlow = xlow
                        # xhigh = xhigh
                        # yhigh = yhigh
                        # ylow = ylow

                    
                elif fdm == 2:
                    xy = (40,1e-36)
                    xysolar = (2,9e-34)
                    if ne == 1 or ne == 2:
                        xlow= mass_low
                        xhigh = 10

                        yhigh=1e-33
                        ylow=1e-37
            
                    
                


            elif material == 'Xe':
                if fdm == 0:
                    xy = (40,2e-41)
                    xysolar = (2,1e-36)
                    if ne == 1:
                        xlow= 3
                        xhigh = mass_high
                        yhigh=5e-37
                        ylow=1e-42
                    elif ne == 2:
                        xlow= 5
                        xhigh = mass_high
                        yhigh=1e-38
                        ylow=1e-42

                    elif ne == 3 or ne == 4:
                        xlow = 10
                        xhigh = mass_high
                        yhigh = 1e-38
                        ylow = 1e-42

                if fdm == 2:
                    xy = (40,1e-36)
                    xysolar = (2,9e-34)
                    if ne == 1:
                        xlow= 3
                        xhigh = mass_high
                        yhigh=1e-34
                        ylow=1e-38
                    elif ne == 2:
                        xlow= 10
                        xhigh = mass_high
                        yhigh=1e-34
                        ylow=1e-38

                    elif ne == 3 or ne == 4:
                        xlow = 10
                        xhigh = mass_high
                        yhigh = 1e-34
                        ylow = 1e-38
                   
                    

            elif material == 'Ar':

                if fdm == 0:
                    xy = (40,2e-41)
                    xysolar = (2,1e-36)
                    if ne == 2:
                        xlow= 10
                        xhigh = mass_high
                        yhigh=1e-38
                        ylow=1e-42
                    elif ne == 3 or ne ==4:
                        xlow= 10
                        xhigh = mass_high
                        yhigh=1e-38
                        ylow=1e-42

                if fdm == 2:
                    xy = (40,1e-36)
                    xysolar = (2,9e-34)
                    if ne == 2 or ne == 3:
                        xlow= 10
                        xhigh = mass_high
                        yhigh=1e-32
                        ylow=1e-38
                    elif ne == 4:
                        xlow= 10
                        xhigh = mass_high
                        yhigh = 1e-32
                        ylow = 1e-38
                

         



        
            
                
            

            yhighexp = find_exp(yhigh)
            ylowexp = find_exp(ylow)
            # print(yhighexp,ylowexp)
            yticks = np.arange(-50,-28,1)
            # print(yticks)
            yticks = np.power(10.,yticks)
            # print(yticks)
            n = 2
            current_ax.set_yticks(yticks)
            # [l.set_visible(False) for (i,l) in enumerate(current_ax.yaxis.get_ticklabels()) if i % n != 0]

            current_ax.set_xlim(xlow,xhigh)
            current_ax.set_ylim(ylow,yhigh)


            current_ax.tick_params('x', top=True, labeltop=False)
            current_ax.tick_params('y', right=True, labelright=False)
            current_ax.set_xlabel('$m_\chi$ [MeV]',fontsize=small)
            current_ax.set_ylabel('$\overline{\sigma}_e$ [cm$^2$]',fontsize=small)
            
            vmin = 0
            vmax = 5
            max_sig = np.max(Significance)
            min_sig = np.min(Significance)
            norm =colors.Normalize(vmin=vmin,vmax=vmax)
            # if max_sig > 2:
            #     levs = np.arange(0,int(max_sig),1)
            # else:
            levs = np.arange(0,5.5,0.5)
            extend = 'max'
            cmap = 'Reds'#diverging
            original_cmap = cmap
            cmap = modify_colormap(original_cmap,divisor=10)


            CT1 =  current_ax.contourf(Masses,CrossSections,Amplitudes,levs,norm=norm,cmap=cmap,extend=extend)
            
           
            if plotConstraints:
                import sys
                # sys.path.append('../../../limits/other_experiments/')
                sys.path.append('../limits/')
                from Constraints import plot_constraints
                
                xsol,ysol = plot_constraints('Solar',fdm)
                current_ax.plot(xsol,ysol,color='black',lw=3,ls='--')
                # upper_boundary_sol=np.ones_like(ysol)*yhigh
                # current_ax.fill_between(xsol,ysol,upper_boundary_sol,alpha=0.3, color='grey')

                x,y = plot_constraints('All',fdm)
                current_ax.plot(x,y,color='black',lw=3)
                if fdm == 0:
                    x,y = plot_constraints('Migdal',fdm)
                    current_ax.plot(x,y,color='black',lw=3,ls=':')



                from scipy.interpolate import interp1d
                constraint_interp = interp1d(x,y,bounds_error=False,fill_value=np.nan)
                solar_constraint_interp = interp1d(xsol,ysol,bounds_error=False,fill_value=np.nan)
                grid = np.geomspace(xlow,xhigh,50)
                ylower = []
                for m in grid:
                    ylower.append(np.nanmin(np.array([constraint_interp(m),solar_constraint_interp(m)])))
                
                ylower = np.array(ylower)

                upper_boundary=np.ones_like(grid)*yhigh
                current_ax.fill_between(grid,ylower,upper_boundary,alpha=0.3, color='grey')

            if showProjection:
                 if i == 2:
                    if material == 'Si':
                        oscura_heavy = '../sensitivity_projections/oscura_heavy.csv' #exposure 30 kg year
                        oscura_light = '../sensitivity_projections/oscura_light.csv' #exposure 30 kg year
                        
                        f = oscura_heavy if fdm == 0 else oscura_light
                    elif material == 'Ar':
                    
                        darkside20k_heavy = '../sensitivity_projections/Darkside20k_heavy.csv' #exposure 17.4 ton·year for one year of data
                        darkside20k_light = '../sensitivity_projections/Darkside20k_light.csv' #exposure 17.4 ton·year for one year of data

                        f = darkside20k_heavy if fdm == 0 else darkside20k_light
                    fdata = np.loadtxt(f,delimiter=',')
                    current_ax.plot(fdata[:,0],fdata[:,1],color='blue',lw=2,label='direct sensitivity')


                




            import matplotlib.patches as patches

            if fdm == 0:
                box_xpos,box_ypos = 0.14,0.06
                e_xpos,e_ypos = box_xpos,box_ypos+0.07
                rect = patches.Rectangle((0.01, 0.02), 0.25, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
            else:
                box_xpos,box_ypos = 0.2,0.06
                e_xpos,e_ypos = box_xpos,box_ypos+0.07
                rect = patches.Rectangle((0.01, 0.02), 0.38, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
            # xw = 0.4 if fdm == 2 else 0.25
            
            current_ax.add_patch(rect)
            current_ax.text(0.95, 0.93, material,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=medium,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))
            current_ax.text(box_xpos, box_ypos, fdmstr,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=smaller,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            e_bin_str = f'{ne}' + '$e^-$ bin'
            current_ax.text(e_xpos, e_ypos, e_bin_str,
            horizontalalignment='center',
            verticalalignment='center', 
            transform = current_ax.transAxes,c='Black',fontsize=smaller,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))


            time_unit = time_unit_strs[i]
            exp_str = exposures[i] / time_units[material][i]
            exp_str = np.round(exp_str)
            if i == 1:
                exp_str /=30

            if material=='Si':
                mass_unit_str = 'kg'
            else:
                mass_unit_str = 'tonne'


            

        
            exp_str = int(exp_str)
            exposure_str = f'{exp_str} {mass_unit_str}-{time_unit}'

            current_ax.text(0.98, 0.05, exposure_str,
            horizontalalignment='right',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=small,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            
            if shadeMFP:
                import sys
                from tqdm.autonotebook import tqdm
                sys.path.append('../DaMaSCUS/')
                from MeanFreePath import Earth_Density_Layer_NU


                mX_grid_mfp = np.geomspace(mass_low,mass_high,100)
                sigmaE_grid_mpf = np.geomspace(cs_low,cs_high,100)

                #np.arange(0.1,1500,0.1)
                EDLNU = Earth_Density_Layer_NU()
                r_test = 0.8*EDLNU.EarthRadius #choose mantle
                vMax = 300 * EDLNU.km / EDLNU.sec

                MFP = []
                for s in range(len(sigmaE_grid_mpf)):
                    MFP_small = []
                    for m in range(len(mX_grid_mfp)):
                        mX = mX_grid_mfp[m]*1e-3 #GeV
                        sigmaP= sigmaE_grid_mpf[s] * (EDLNU.muXElem(mX,EDLNU.mProton) / EDLNU.muXElem(mX,EDLNU.mElectron))**2

                        mfp = EDLNU.Mean_Free_Path(r_test,mX,sigmaP,vMax,fdm,doScreen=True)
                        MFP_small.append(mfp)
                    MFP_small = np.array(MFP_small)
                    MFP.append(MFP_small)
                MFP = np.array(MFP)

               
                X,Y = np.meshgrid(mX_grid_mfp,sigmaE_grid_mpf)
                shade = np.ma.masked_where(MFP > 1, MFP)

                current_ax.pcolor(X, Y, shade,hatch='/',alpha=0)

                

            


    cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])

    cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)

    cbar.ax.tick_params(labelsize=large)
    if savefig:
        mat_str_dict = {
            'Si': 'Silicon',
            'Xe': 'Xenon',
            'Ar': 'Argon',
        }
        matstr = mat_str_dict[material]
        
        savedir = f'figures/{matstr}'

        tag = 'verne' if useVerne else 'damascus'
        plt.savefig(f'{savedir}/Mod_Sensitivity_{material}_CombinedFig_{ne}ebin_loc{loc}_{tag}.jpg')

    plt.show()
    plt.close()
    return 


def plotMaterialSeasonalSignificanceComparison(fdm,loc,material='Si',plotConstraints=True,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs=None,ne=1,shadeMFP=True,savefig=False,standardizeGrid = False,useQCDark=True,showProjection=False):
    """Compare modulation significance between March and June.
    
    Args:
        fdm: Form factor model (0=FDM1, 2=FDMq2)
        loc: Location name ('SNOLAB', etc.)
        material: Target material (default 'Si')
        plotConstraints: Plot experimental constraints (default True)
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        verbose: Print debug info (default False)
        masses: Custom mass grid (default None)
        sigmaEs: Custom cross section grid (default None)
        ne: Electron bin (default 1)
        shadeMFP: Shade mean free path region where Verne is expected to be less accurate (default True)
        savefig: Save figure to file (default False)
        standardizeGrid: Use standardized grid (default False)
        useQCDark: Use form factor from QCDark (default True)
        showProjection: Show experimental projections (default False)
    """
    from tqdm.autonotebook import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors

    #Options

    large = 48
    small = 36
    medium = 40
    smaller = 36
    smallest=16
    set_default_plotting_params(fontsize=medium)
   
    nrows = 1
    ncols = 1

    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,layout='constrained',figsize=(16,14))
    ebinstr = f'{ne}' + '$e^-$ bin'

    fdmstr = '$F_{\mathrm{DM}} = (\\alpha m_e/q)^2$' if fdm == 2 else '$F_{\mathrm{DM}} = 1$'
    
    matdit = {
        'Si': 'Silicon',
        "Xe": 'Xenon',
        'Ar': "Argon"
    }

    if material == 'Si':
        if ne == 1:
            pix_1e = 1.39e-5 #e^-/pix/day
            background_rate = pix_1e / (3.485*1e-7) #e- /gram/day
            background_rate = background_rate / nu.g / nu.day
            # background_rate *= 1000 #e / kg/ day
        elif ne == 2:
            # #snolab 2e rate
            exp_2e =46.61  * nu.g * nu.day
            # exp_2e /=1000 #kg days
            counts_2e = 55
            background_rate = counts_2e / exp_2e #e / kg /day
        else:
            background_rate = 0

    elif material == 'Xe':
        #taken from https://arxiv.org/pdf/2411.15289
        if ne == 1:
            background_rate = 3  / nu.kg / nu.day 
        elif ne == 2:
            background_rate = 0.1 / nu.kg / nu.day 
        elif ne == 3:
            background_rate = 0.02 / nu.kg / nu.day 
        elif ne == 4:
            background_rate = 0.01 / nu.kg / nu.day 


    elif material == 'Ar':
        #values from https://arxiv.org/pdf/2407.05813
        argon_2e_background= 0.1 #events / 0.25 *kg / day
        argon_3e_background= 5e-3 #events / 0.25 *kg / day
        argon_4e_background= 1e-3 #events / 0.25 *kg / day
        if ne == 1:
            raise ValueError('No 1e background rate for Argon')
        elif ne == 2:
            background_rate = argon_2e_background/0.25 / nu.kg / nu.day 
        elif ne == 3:
            background_rate = argon_3e_background/0.25 / nu.kg / nu.day 
        elif ne == 4:
            background_rate = argon_4e_background/0.25 / nu.kg / nu.day 
    
    exposure_dict = {
        'Si': np.array([1 * nu.kg * nu.day,30 * nu.kg * nu.day,30 * nu.kg * nu.year]), #kg day, kg month, 30 kg year
        'Xe': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day ,1 * nu.tonne * nu.year]),#tonne day, tonne month, 1 tonne year
        'Ar': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day, 1 * nu.tonne * nu.year])#tonne day, tonne month, ~17.4 tonne year
    }
    time_units = {
        'Si': [nu.kg*nu.day,nu.kg*nu.day,nu.kg*nu.year],
        'Xe': [nu.tonne*nu.day,nu.tonne*nu.day,nu.tonne*nu.year],
        'Ar': [nu.tonne*nu.day,nu.tonne*nu.day,nu.tonne*nu.year],

    }
            
    time_unit_strs = ['day','month','year']
    exposures = exposure_dict[material]
    fig.suptitle(f"{matdit[material]} Sensitivity at {loc} March vs June",fontsize=large,y=1.04)



    masses_list = []
    cs_list = []
    famp_list = []
    amp_list = []
    sig_list = []
    if masses is None and sigmaEs is None:
        getAll = True
    else:
        getAll = False


    i = -1
    exposure = exposures[i]
    exp_str = exposures[i] / time_units[material][i]
    exp_str = np.round(exp_str)
    if i == 1:
        exp_str /=30
    exp_str = int(exp_str)
    time_unit = time_unit_strs[i]
        
       

    if material=='Si':
        mass_unit_str = 'kg'
    else:
        mass_unit_str = 'tonne'
    exposure_str = f'{exp_str} {mass_unit_str}-{time_unit}'
    for i,s in enumerate([False,True]):

        Masses,CrossSections,FractionalAmplitudes = getContourData(material,fdm,loc,fractional=True,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,useQCDark=useQCDark,summer=s)
        Masses,CrossSections,Amplitudes = getContourData(material,fdm,loc,fractional=False,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,returnaverage=True,useQCDark=useQCDark,summer=s)
        FractionalAmplitudes[np.isnan(FractionalAmplitudes)] = 0
        Significance = (FractionalAmplitudes*Amplitudes) / np.sqrt((Amplitudes + background_rate))

        if i == 0:
            ls= '-'
        else:
            ls = '--'

   
        current_ax = axes
    
        
        if s ==True:
            summer_str = 'Summer'

        else:
            summer_str = 'Average'

           
            



        Significance = Significance*(exposure / np.sqrt(exposure))

        Significance[np.isnan(Significance)] = 0

        Significance[Significance > 5] = 5.5

        if masses is not None and sigmaEs is not None:
            Significance = Significance.T


        current_ax.set_xscale('log')
        current_ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        current_ax.set_yscale('log')
        if masses is not None:
            mass_low = np.min(masses)
            mass_high = np.max(masses)
        else:
            mass_low = np.min(Masses)
            mass_high = np.max(Masses)
        cs_low = np.min(CrossSections)
        cs_high = np.max(CrossSections)
        if fdm == 2:
            cs_low = 1e-40
            cs_high = 1e-30

        xlow = mass_low
        xhigh = mass_high
        ylow = cs_low
        yhigh = cs_high
        if material == 'Si':
            if fdm == 0:
                xy = (40,2e-41)
                xysolar = (1.5,3e-37)
                if ne == 1 or ne ==2:
                    xlow = mass_low
                    xhigh = 10
                    yhigh = 1e-35
                    ylow = 1e-38
                    # xlow = xlow
                    # xhigh = xhigh
                    # yhigh = yhigh
                    # ylow = ylow

                
            elif fdm == 2:
                xy = (40,1e-36)
                xysolar = (2,9e-34)
                if ne == 1 or ne == 2:
                    xlow= mass_low
                    xhigh = 10

                    yhigh=1e-33
                    ylow=1e-37
        
                
            


        elif material == 'Xe':
            if fdm == 0:
                xy = (40,2e-41)
                xysolar = (2,1e-36)
                if ne == 1:
                    xlow= 3
                    xhigh = mass_high
                    yhigh=5e-37
                    ylow=1e-42
                elif ne == 2:
                    xlow= 5
                    xhigh = mass_high
                    yhigh=1e-38
                    ylow=1e-42

                elif ne == 3 or ne == 4:
                    xlow = 10
                    xhigh = mass_high
                    yhigh = 1e-38
                    ylow = 1e-42

            if fdm == 2:
                xy = (40,1e-36)
                xysolar = (2,9e-34)
                if ne == 1:
                    xlow= 3
                    xhigh = mass_high
                    yhigh=1e-34
                    ylow=1e-38
                elif ne == 2:
                    xlow= 10
                    xhigh = mass_high
                    yhigh=1e-34
                    ylow=1e-38

                elif ne == 3 or ne == 4:
                    xlow = 10
                    xhigh = mass_high
                    yhigh = 1e-34
                    ylow = 1e-38
                
                

        elif material == 'Ar':

            if fdm == 0:
                xy = (40,2e-41)
                xysolar = (2,1e-36)
                if ne == 2:
                    xlow= 10
                    xhigh = mass_high
                    yhigh=1e-38
                    ylow=1e-42
                elif ne == 3 or ne ==4:
                    xlow= 10
                    xhigh = mass_high
                    yhigh=1e-38
                    ylow=1e-42

            if fdm == 2:
                xy = (40,1e-36)
                xysolar = (2,9e-34)
                if ne == 2 or ne == 3:
                    xlow= 10
                    xhigh = mass_high
                    yhigh=1e-32
                    ylow=1e-38
                elif ne == 4:
                    xlow= 10
                    xhigh = mass_high
                    yhigh = 1e-32
                    ylow = 1e-38
            





        
            
        

        yhighexp = find_exp(yhigh)
        ylowexp = find_exp(ylow)
        # print(yhighexp,ylowexp)
        yticks = np.arange(-50,-28,1)
        # print(yticks)
        yticks = np.power(10.,yticks)
        # print(yticks)
        n = 2
        current_ax.set_yticks(yticks)
        # [l.set_visible(False) for (i,l) in enumerate(current_ax.yaxis.get_ticklabels()) if i % n != 0]

        current_ax.set_xlim(xlow,xhigh)
        current_ax.set_ylim(ylow,yhigh)


        current_ax.tick_params('x', top=True, labeltop=False)
        current_ax.tick_params('y', right=True, labelright=False)
        current_ax.set_xlabel('$m_\chi$ [MeV]',fontsize=small)
        current_ax.set_ylabel('$\overline{\sigma}_e$ [cm$^2$]',fontsize=small)
        vmin = 0
        vmax = 5
        max_sig = np.max(Significance)
        min_sig = np.min(Significance)
        
        # if max_sig > 2:
        #     levs = np.arange(0,int(max_sig),1)
        # else:
        levs = np.arange(0,5.5,1)
        divisor=len(levs)
        
        
        norm =colors.Normalize(vmin=vmin,vmax=vmax)
        extend = 'max'
        cmap = 'Reds'#diverging
        original_cmap = cmap
        cmap = modify_colormap(original_cmap,divisor=divisor)


     
        CT1 =  current_ax.contourf(Masses,CrossSections,Significance,levs,norm=norm,cmap=cmap,extend=extend,alpha=0)
        two_sigma = CT1.allsegs[2][0]
        current_ax.plot(two_sigma[:, 0][280:-800], two_sigma[:, 1][280:-800], color=northcolor,lw=3,ls=ls,label='2$\sigma$')
        five_sigma = CT1.allsegs[-1][0]
        current_ax.plot(five_sigma[:, 0], five_sigma[:, 1], color='black',lw=3,ls=ls,label='5$\sigma$')
    import matplotlib.lines as mlines

    red = mlines.Line2D([], [], color='black', label='5$\sigma$')
    blue = mlines.Line2D([], [], color=northcolor, label='2$\sigma$')

    legend1 = current_ax.legend(handles=[red,blue],loc='center left',fontsize=medium,frameon=False)
    current_ax.add_artist(legend1)

    dashed = mlines.Line2D([], [], color='black', linestyle='--', label='June')
    solid = mlines.Line2D([], [], color='black', label='March')
    legend2 = current_ax.legend(handles=[solid, dashed], fontsize=medium, loc='center right',frameon=False)

    
    if plotConstraints:
        import sys
        # sys.path.append('../../../limits/other_experiments/')
        sys.path.append('../limits/')
        from Constraints import plot_constraints
        
        xsol,ysol = plot_constraints('Solar',fdm)
        current_ax.plot(xsol,ysol,color='black',lw=3,ls='--')
        # upper_boundary_sol=np.ones_like(ysol)*yhigh
        # current_ax.fill_between(xsol,ysol,upper_boundary_sol,alpha=0.3, color='grey')

        x,y = plot_constraints('All',fdm)
        current_ax.plot(x,y,color='black',lw=3)
        if fdm == 0:
            x,y = plot_constraints('Migdal',fdm)
            current_ax.plot(x,y,color='black',lw=3,ls=':')


        from scipy.interpolate import interp1d
        constraint_interp = interp1d(x,y,bounds_error=False,fill_value=np.nan)
        solar_constraint_interp = interp1d(xsol,ysol,bounds_error=False,fill_value=np.nan)
        grid = np.geomspace(xlow,xhigh,50)
        ylower = []
        for m in grid:
            ylower.append(np.nanmin(np.array([constraint_interp(m),solar_constraint_interp(m)])))
        
        ylower = np.array(ylower)

        upper_boundary=np.ones_like(grid)*yhigh
        current_ax.fill_between(grid,ylower,upper_boundary,alpha=0.3, color='grey')

    if showProjection:
        if material == 'Si':
            oscura_heavy = '../sensitivity_projections/oscura_heavy.csv' #exposure 30 kg year
            oscura_light = '../sensitivity_projections/oscura_light.csv' #exposure 30 kg year
            
            f = oscura_heavy if fdm == 0 else oscura_light
        elif material == 'Ar':
        
            darkside20k_heavy = '../sensitivity_projections/Darkside20k_heavy.csv' #exposure 17.4 ton·year for one year of data
            darkside20k_light = '../sensitivity_projections/Darkside20k_light.csv' #exposure 17.4 ton·year for one year of data

            f = darkside20k_heavy if fdm == 0 else darkside20k_light
        fdata = np.loadtxt(f,delimiter=',')
        current_ax.plot(fdata[:,0],fdata[:,1],color='blue',lw=2,label='direct sensitivity')


        

    import matplotlib.patches as patches

    if fdm == 0:
        box_xpos,box_ypos = 0.14,0.06
        e_xpos,e_ypos = box_xpos,box_ypos+0.07
        rect = patches.Rectangle((0.01, 0.02), 0.25, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
    else:
        box_xpos,box_ypos = 0.2,0.06
        e_xpos,e_ypos = box_xpos,box_ypos+0.07
        rect = patches.Rectangle((0.01, 0.02), 0.38, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
    xw = 0.4 if fdm == 2 else 0.25
    
    current_ax.add_patch(rect)
    current_ax.text(0.95, 0.93, material,
    horizontalalignment='center',
    verticalalignment='center',
    transform = current_ax.transAxes,c='Black',fontsize=medium,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

    current_ax.text(box_xpos, box_ypos, fdmstr,
    horizontalalignment='center',
    verticalalignment='center',
    transform = current_ax.transAxes,c='Black',fontsize=smaller,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))
   
    e_bin_str = f'{ne}' + '$e^-$ bin'
    current_ax.text(e_xpos, e_ypos, e_bin_str,
    horizontalalignment='center',
    verticalalignment='center', 
    transform = current_ax.transAxes,c='Black',fontsize=smaller,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

  
    current_ax.text(0.98, 0.05, exposure_str,
    horizontalalignment='right',
    verticalalignment='center',
    transform = current_ax.transAxes,c='Black',fontsize=small,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))




    
    if shadeMFP:
        import sys
        from tqdm.autonotebook import tqdm
        sys.path.append('../DaMaSCUS/')
        from MeanFreePath import Earth_Density_Layer_NU


        mX_grid_mfp = np.geomspace(mass_low,mass_high,100)
        sigmaE_grid_mpf = np.geomspace(cs_low,cs_high,100)

        #np.arange(0.1,1500,0.1)
        EDLNU = Earth_Density_Layer_NU()
        r_test = 0.8*EDLNU.EarthRadius #choose mantle
        vMax = 300 * EDLNU.km / EDLNU.sec

        MFP = []
        for s in range(len(sigmaE_grid_mpf)):
            MFP_small = []
            for m in range(len(mX_grid_mfp)):
                mX = mX_grid_mfp[m]*1e-3 #GeV
                sigmaP= sigmaE_grid_mpf[s] * (EDLNU.muXElem(mX,EDLNU.mProton) / EDLNU.muXElem(mX,EDLNU.mElectron))**2

                mfp = EDLNU.Mean_Free_Path(r_test,mX,sigmaP,vMax,fdm,doScreen=True)
                MFP_small.append(mfp)
            MFP_small = np.array(MFP_small)
            MFP.append(MFP_small)
        MFP = np.array(MFP)

        # shade = np.zeros_like(MFP,dtype=bool)
        # shade[MFP < 1] = True
        # shade[MFP>=1] = False
        X,Y = np.meshgrid(mX_grid_mfp,sigmaE_grid_mpf)
        shade = np.ma.masked_where(MFP > 1, MFP)

        current_ax.pcolor(X, Y, shade,hatch='/',alpha=0)

        

    if savefig:
        mat_str_dict = {
            'Si': 'Silicon',
            'Xe': 'Xenon',
            'Ar': 'Argon',
        }
        matstr = mat_str_dict[material]
        
        savedir = f'figures/{matstr}'

        tag = 'verne' if useVerne else 'damascus'
        print(f'saved in {savedir}')
        plt.savefig(f'{savedir}/Seasonal_Mod_Sensitivity_{material}_CombinedFig_{ne}ebin_loc{loc}_FDM{fdm}_{tag}.jpg',bbox_inches='tight')

    # plt.tight_layout()
    plt.show()
    plt.close()
    return CT1


def plotModulationFigure(fdm,fractional=False,plotConstraints=True,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs=None,ne=1,shadeMFP=True,savefig=False,kgday=True,useQCDark=True,logfractional=True,summer=False):
    """Create multi-panel figure showing modulation amplitudes.
    
    Args:
        fdm: Form factor model (0=FDM1, 2=FDMq2)
        fractional: Plot fractional amplitudes (default False)
        plotConstraints: Plot experimental constraints (default True)
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        verbose: Print debug info (default False)
        masses: Custom mass grid (default None)
        sigmaEs: Custom cross section grid (default None)
        ne: Electron bin (default 1)
        shadeMFP: Shade mean free path region where Verne is less accurate (default True)
        savefig: Save figure to file (default False)
        useQCDark: Use form factor from QCDark (default True)
        logfractional: Use log scale for fractional amps (default True)
        summer: Use summer distribution (default False)
    """
    from tqdm.autonotebook import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors

    #Options

    large = 48
    small = 36
    medium = 40
    smaller = 32
    smallest=16
    set_default_plotting_params(fontsize=large)
    
    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,layout='constrained',figsize=(26,26))
    fracstr = "Fractional" if fractional else ""
    fdmstr = '$F_{\mathrm{DM}} = (\\alpha m_e/q)^2$' if fdm == 2 else '$F_{\mathrm{DM}} = 1$'
    ebinstr = f'{ne}' + '$e^-$ bin'
    fig.suptitle(f"{fracstr} Modulation Amplitude {fdmstr}",fontsize=large)
    materials = ['Si','Xe','Ar']
    temp_amps = []
    for i in range(nrows):
        mat = materials[i]
        for j in range(ncols):
            if j == 0 or j==2:
                loc = 'SNOLAB'
            else:
                loc = 'SUPL'
            # if j == 0 or j == 1:
            #     fdm = 0
            # else:
            #     fdm = 2
            current_ax = axes[i,j]

            if masses is None and sigmaEs is None:
                getAll = True
            else:
                getAll = False
            unitize = False if fractional else True
            Masses,CrossSections,Amplitudes = getContourData(mat,fdm,loc,fractional=fractional,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,unitize=unitize,useQCDark=useQCDark,summer=summer)
            if fractional:
                Amplitudes[np.isnan(Amplitudes)] = np.nanmin(Amplitudes)
                Amplitudes[np.isnan(Amplitudes)] =np.nanmin(Amplitudes)
                Amplitudes[Amplitudes == np.inf] = np.nanmax(Amplitudes)
                Amplitudes[Amplitudes == -np.inf] = np.nanmin(Amplitudes)
            else:
                # Amplitudes[np.isnan(Amplitudes)] =0
                Amplitudes[Amplitudes==0] = 1e-10
           
            if masses is not None and sigmaEs is not None:
                Amplitudes = Amplitudes.T

            temp_amps.append(np.nanmax(Amplitudes))

            current_ax.set_xscale('log')
            current_ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            current_ax.set_yscale('log')
            if masses is not None:
                mass_low = np.min(masses)
                mass_high = np.max(masses)
            else:
                mass_low = np.min(Masses)#0.6
                mass_high = 1000 #np.max(Masses) #1000



            if fdm == 0:
                cs_low = 1e-42
                # cs_low = np.min(CrossSections)
                cs_high = 1e-37

            elif fdm == 2:
                mass_low = 0.5
                if fractional:
                    cs_low = 1e-37
                else:
                    cs_low  = 1e-40
                cs_high = 1e-32
            else:
                cs_low = np.min(CrossSections)
                cs_high = np.max(CrossSections)
            xlow = mass_low
            xhigh = mass_high
            ylow = cs_low
            yhigh = cs_high
           
            yhighexp = find_exp(yhigh)
            ylowexp = find_exp(ylow)
            yticks = np.arange(-50,-28,1)
            # print(yticks)
            yticks = np.power(10.,yticks)
            # print(yticks)
            n = 2
            current_ax.set_yticks(yticks)
            [l.set_visible(False) for (i,l) in enumerate(current_ax.yaxis.get_ticklabels()) if i % n != 0]


            current_ax.set_xlim(xlow,xhigh)
            current_ax.set_ylim(ylow,yhigh)




            current_ax.tick_params('x', top=True, labeltop=False)
            current_ax.tick_params('y', right=True, labelright=False)
            current_ax.set_xlabel('$m_\chi$ [MeV]',fontsize=small)
            current_ax.set_ylabel('$\overline{\sigma}_e$ [cm$^2$]',fontsize=small)
            if fractional:
                vmin = 0
                
                max_amp = np.nanmax(Amplitudes) + np.mean(Amplitudes)
                min_amp = np.nanmin(Amplitudes)
                
                
                
                if logfractional:
                    #try log scale
                    vminexp = -2
                    vmaxexp = 0.25
                    vmax = np.power(10.,vmaxexp)
                    vmin = np.power(10.,vminexp)
                    vmaxminus1 =  np.power(10,vmaxexp-1)
                    levs = np.arange(vminexp,vmaxexp,0.25) 
                    levs = np.power(10.,levs)
                    divisor = len(levs)
                    norm =colors.LogNorm(vmin=vmin,vmax=vmax)
                    extend = 'both'
                else:
                    # levs = np.round(levs,3)
                    if max_amp < 2:
                        vmax = 2
                    else:
                        vmax = 10

                    vmax = 2
                    levs = np.arange(0,2.1,0.1)
                    norm =colors.Normalize(vmin=vmin,vmax=vmax)
                    extend = 'max'
                    divisor = len(levs)
                

            else:
                Amplitudes[Amplitudes == np.inf] = np.nanmax(Amplitudes[np.isfinite(Amplitudes)])
                Amplitudes[Amplitudes == -np.inf] = np.nanmin(Amplitudes[np.isfinite(Amplitudes)])
                max_amp = np.nanmax(Amplitudes)
                
                min_amp = np.nanmin(Amplitudes)
                low = -10 #find_exp(min_amp) 
                high = find_exp(max_amp) + 2
                if high <= 1:
                    high = 3

                
    
                
                vminexp = -6
                vmaxexp = 6
                vmax = np.power(10.,vmaxexp)
                vmin = np.power(10.,vminexp)
                vmaxminus1 =  np.power(10,vmaxexp-1)
                levs = np.arange(vminexp,vmaxexp) 
                levs = np.power(10.,levs)
                divisor = len(levs)
    

                norm =colors.LogNorm(vmin=vmin,vmax=vmax)
                extend = 'both'
            cmap = 'Spectral_r'#diverging
            cmap = 'Reds'#sequential
            original_cmap = cmap
 

           
            background_filling = np.ones_like(Amplitudes)*1e-10

            CT1 =  current_ax.contourf(Masses,CrossSections,Amplitudes,levs,norm=norm,cmap=cmap,extend=extend)
            CT1.cmap.set_under(color='white')


          

            if plotConstraints:
                import sys
                sys.path.append('../limits/')
                from Constraints import plot_constraints
                
                xsol,ysol = plot_constraints('Solar',fdm)
                current_ax.plot(xsol,ysol,color='black',lw=3,ls='--')

                x,y = plot_constraints('All',fdm)
                current_ax.plot(x,y,color='black',lw=3)
                if fdm == 0:
                    x,y = plot_constraints('Migdal',fdm)
                    current_ax.plot(x,y,color='black',lw=3,ls=':')


                from scipy.interpolate import interp1d
                constraint_interp = interp1d(x,y,bounds_error=False,fill_value=np.nan)
                solar_constraint_interp = interp1d(xsol,ysol,bounds_error=False,fill_value=np.nan)
                print(xlow,xhigh)
                grid = np.geomspace(xlow,xhigh,50)
                ylower = []
                for m in grid:
                    ylower.append(np.nanmin(np.array([constraint_interp(m),solar_constraint_interp(m)])))
                
                ylower = np.array(ylower)

                upper_boundary=np.ones_like(grid)*yhigh
                current_ax.fill_between(grid,ylower,upper_boundary,alpha=0.3, color='grey')


            if fdm == 0:
                fdm_str = '$F_{\mathrm{DM}}= 1$'
            else:
                fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'

            import matplotlib.patches as patches
            # xw = 0.4 if fdm == 2 else 0.25
            rect = patches.Rectangle((0.01, 0.02), 0.25, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
            current_ax.add_patch(rect)
            current_ax.text(0.95, 0.93, mat,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=medium,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))
            current_ax.text(0.14, 0.05, loc,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            e_bin_str = f'{ne}' + '$e^-$ bin'
            current_ax.text(0.14, 0.13, e_bin_str,
            horizontalalignment='center',
            verticalalignment='center', 
            transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))


            
            if shadeMFP:
                import sys
                from tqdm.autonotebook import tqdm
                sys.path.append('../DaMaSCUS/')
                from MeanFreePath import Earth_Density_Layer_NU


                mX_grid_mfp = np.geomspace(mass_low,mass_high,100)
                # print(mass_low,mass_high)
                sigmaE_grid_mpf = np.geomspace(cs_low,cs_high,100)

                #np.arange(0.1,1500,0.1)
                EDLNU = Earth_Density_Layer_NU()
                r_test = 0.8*EDLNU.EarthRadius #choose mantle
                vMax = 300 * EDLNU.km / EDLNU.sec

                MFP = []
                for s in range(len(sigmaE_grid_mpf)):
                    MFP_small = []
                    for m in range(len(mX_grid_mfp)):
                        mX = mX_grid_mfp[m]*1e-3 #GeV
                        sigmaP= sigmaE_grid_mpf[s] * (EDLNU.muXElem(mX,EDLNU.mProton) / EDLNU.muXElem(mX,EDLNU.mElectron))**2

                        mfp = EDLNU.Mean_Free_Path(r_test,mX,sigmaP,vMax,fdm,doScreen=True)
                        MFP_small.append(mfp)
                    MFP_small = np.array(MFP_small)
                    MFP.append(MFP_small)
                MFP = np.array(MFP)

                X,Y = np.meshgrid(mX_grid_mfp,sigmaE_grid_mpf)
                shade = np.ma.masked_where(MFP > 1, MFP)

                current_ax.pcolor(X, Y, shade,hatch='/',alpha=0)

                

            

            if fdm == 0:
                xy = (40,2e-41)
                xysolar = (0.65,1e-37)
            else: 
                xy = (40,1e-36)
                xysolar = (2,9e-34)
          




    cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
    if fractional:

        cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)


    else:
        # cbar = current_ax.cax.colorbar(CT1,ticks=[1e-6,1e0,1e6],extend=extend,norm=norm)
        cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)
        cbar.ax.set_yticks([vmin,1,vmaxminus1])
        cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
        cbar.ax.set_ylim(vmin,vmaxminus1)
        if kgday:
            unit = 'kg'
        else:
            unit = 'g'
        cbar.ax.set_title(f'[events/{unit}/day]',fontsize=small)

        # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
        # cbar.ax.set_ylim(1e-6,1e6)
    cbar.ax.tick_params(labelsize=small)
    if savefig:
        if fractional:
            frac_str = 'fractional_'
        else:
            frac_str = ''

        savedir = f'figures/Combined/'
        summerstr = '_summer' if summer else ''
        plt.savefig(f'{savedir}/{frac_str}Mod_Amplitude_CombinedFig_FDM{fdm}{summerstr}.jpg')

    # plt.tight_layout()
    
    plt.show()
    plt.close()
    return 




            
def plotRateComparisonSubplots(material,sigmaE_list,mX_list,fdm,plotVerne=True,savefig=False,savedir=None,verneOnly=False,damascusOnly=False,ne=1,showScatter=False,showFit=False,useQCDark=True,showErr=False,showParams=True):
    """Plot rate vs angle comparisons in subplots for different parameters.
    
    Args:
        material: Target material ('Si', 'Xe', 'Ar')
        sigmaE_list: List of cross sections to plot
        mX_list: List of masses to plot
        fdm: Form factor model (0=FDM1, 2=FDMq2)
        plotVerne: Include Verne rates (default True)
        savefig: Save figure to file (default False)
        savedir: Custom save directory (default None)
        verneOnly: Only plot Verne rates (default False)
        damascusOnly: Only plot DaMaSCUS rates (default False)
        ne: Electron bin (default 1)
        showScatter: Show data points (default False)
        showFit: Show fitted curve (default False)
        useQCDark: Use form factor from QCDark (default True)
        showErr: Show error bars (default False)
    """
    import numpy as np
    # plotting specifications
    import matplotlib.pyplot as plt



    #Options
    smaller=24
    small = 32
    large= 44
    medium = 40
    set_default_plotting_params(fontsize=medium)

    matnamedict = {
        'Si': "Silicon",
        'Xe': "Xenon",
        'Ar': 'Argon',
    }

    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue
    # golden = (1 + 5 ** 0.5) / 2
    # goldenx = 15
    # goldeny = goldenx / golden
    fig,axes = plt.subplots(1,len(mX_list),layout='constrained',figsize=(29,9))
    fig.suptitle(f'{material} {ne}$e^-$ Rate vs Isoangle',fontsize=large)
    for i in range(len(mX_list)):
        current_ax = axes[i]
        mX = mX_list[i]
        sigmaE = sigmaE_list[i]
        current_ax.set_xlabel('$\Theta$\N{degree sign}')
        kgstr = 'k'
        current_ax.set_ylabel(f'Rate [events/{kgstr}g/day]')
        current_ax.grid()
        
        current_ax.set_xlim(0,180)
        # plt.ylim(1e-8,1e3)
        #reversed('RdBu')

        southx = np.linspace(89.26129275462549,164.8486791095454,50)
        southy1 = np.ones_like(southx)*1e14
        southy2 = np.zeros_like(southx)
        northx = np.linspace(6.039066639146133,81.33821611144151,50)
        northy1 = np.ones_like(northx)*1e14
        northy2 = np.zeros_like(northx)




        current_ax.fill_between(southx,southy1,southy2,color=southcolor,alpha=0.3)

        current_ax.fill_between(northx,northy1,northy2,color=northcolor,alpha=0.3)



        maxv = 0
        minv = 9999999
        if plotVerne and not showScatter:
            ls = ['-','--']
            dummy_lines = []
            for b_idx in [0,1]:
                dummy_lines.append(current_ax.plot([],[], c="black", ls = ls[b_idx])[0])
            legend2 = current_ax.legend([dummy_lines[i] for i in [0,1]], ["DaMaSCUS", "Verne"], loc='center right',prop={'size': small})
            current_ax.add_artist(legend2)

        if not verneOnly:
            isoangles,rates = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=False,ne=ne,useQCDark=useQCDark)
            isoangles,rates_high = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=False,calcError="High",ne=ne,useQCDark=useQCDark)
            isoangles,rates_low = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=False,calcError="Low",ne=ne,useQCDark=useQCDark)
            isoangles,rates_flat = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=False,calcError="Low",ne=ne,useQCDark=useQCDark,flat=True)
            rates = rates.flatten().numpy()
            rates_low = rates_low.flatten().numpy()
            rates_high = rates_high.flatten().numpy()
            rates_flat = rates_flat.flatten().numpy()
            isoangles = isoangles.flatten().numpy()



            rates = rates * nu.kg * nu.day
            rates_high = rates_high * nu.kg * nu.day
            rates_low = rates_low * nu.kg * nu.day
            rates_flat = rates_flat * nu.kg * nu.day

            

            # current_ax.plot(isoangles,rates_flat,color='green',label="Flat",lw=3)
            maxv = np.max(rates_high)
            minv = np.min(rates_low)

            amp = (maxv-minv) / 2
        

            rate_err = rates_high - rates
            if showScatter:
                if showErr:
                    current_ax.errorbar(isoangles,rates,yerr=rate_err,linestyle='')
                current_ax.scatter(isoangles,rates,label='Data',s=100,color='black')

                if showFit:
                    if fdm == 2 and (material == 'Xe' or material == "Ar"):
                        angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates,linear=True)
                    else:
                        try:
                            angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates)
                        except ValueError:
                            # try:
                            #     angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates)
                            # except ValueError:
                            angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates,linear=True)
                            

                    fit = fit_vector[0]
                    amp = (np.max(fit)-np.min(fit)) / 2
   

                    current_ax.plot(angle_grid,fit,color='red',label="Fit",lw=4)

            else:
                
                current_ax.fill_between(isoangles,rates_low,rates_high)
            x = isoangles[25]
            y = rates_high[18]*1.2
            # current_ax.text(x,y,f'{mX} MeV',fontsize=30,color=colorlist[i],horizontalalignment='center',verticalalignment='center')

        if not damascusOnly:
            isoangles_v,rates_v = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=True,ne=ne,useQCDark=useQCDark)
            rates_v = rates_v.flatten().numpy()
            rates_v = rates_v * nu.kg  * nu.day
            
         
            current_ax.plot(isoangles_v,rates_v,ls='--',label="Verne",color='slategrey',lw=3)
            if np.max(rates_v) > maxv:
                maxv = np.max(rates_v)
            if np.min(rates_v) < minv:
                minv = np.min(rates_v)

        if showScatter:
            if (material == 'Ar' or material == 'Xe') and fdm == 2 or (fdm == 0 and (i ==0 or i ==1) and (material == 'Ar' or material == 'Xe')) or (material == 'Si' and (i == 2) and fdm == 0) or (material == 'Si' and (i ==1 or i == 2) and fdm == 2):# or (material == 'Si' and i ==0 and fdm == 2): 
                current_ax.legend(loc='upper left',prop={'size': small})
            elif material == 'Si' and i ==0:
                current_ax.legend(loc='center left',prop={'size': small})
            else:
                current_ax.legend(loc='center right',prop={'size': small})


        if not verneOnly and showParams:
            try:
                rbar = np.mean(rates)
                amp = np.abs(parameters[0])*rbar
                amplitude = round_to_sig_figs(amp,3)
                inflection =  round_to_sig_figs(parameters[1],3)
                slope_angle =  round_to_sig_figs(parameters[2],3)
                shift =  round_to_sig_figs(parameters[3]*rbar,3)

       
                
            # plt.setp(current_ax.get_yticklabels()[::2], visible=False)
                current_ax.text(0.05,0.33,f'$A={amplitude}$',fontsize=smaller,color='black',horizontalalignment='left',verticalalignment='center',transform = current_ax.transAxes)
                current_ax.text(0.05,0.27,f'$\\theta_0={inflection}$',fontsize=smaller,color='black',horizontalalignment='left',verticalalignment='center',transform = current_ax.transAxes)
                current_ax.text(0.05,0.19,f'$\\theta_s={slope_angle}$',fontsize=smaller,color='black',horizontalalignment='left',verticalalignment='center',transform = current_ax.transAxes)
                current_ax.text(0.05,0.12,f'$C={shift}$',fontsize=smaller,color='black',horizontalalignment='left',verticalalignment='center',transform = current_ax.transAxes)
            except IndexError:
                current_ax.text(0.05,0.12,f'Linear Fit',fontsize=smaller,color='black',horizontalalignment='left',verticalalignment='center',transform = current_ax.transAxes)

        current_ax.text(0.25,0.05,'SNOLAB ($46$\N{degree sign}N)',fontsize=small,color='grey',horizontalalignment='center',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.72,0.05,'SUPL ($37$\N{degree sign}S)',fontsize=small,color='grey',horizontalalignment='center',verticalalignment='center',transform = current_ax.transAxes)
        if fdm == 2:
            fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'
        elif fdm == 0:
            fdm_str = '$F_{\mathrm{DM}} = 1$'

        mX_str = '$m_\chi = $' + f'{mX} MeV'

        
        
       
        sE_str = str(sigmaE)
        sigmaE_str = '$\overline{\sigma}_e =$ ' + r'${} \times 10^{{{}}}$'.format(*sE_str.split('e')) + 'cm$^2$'
        current_ax.text(0.99,0.86,sigmaE_str,fontsize=small,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.99,0.95,fdm_str,fontsize=small,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.99,0.77,mX_str,fontsize=small,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        

        minv*=0.9
        maxv*=1.1

        current_ax.set_xticks(np.linspace(0,180,19)[::2])
        current_ax.set_ylim(minv,maxv)
    if savefig:
        if savedir is None:
            savedir = f'figures/{matnamedict[material]}/'
            file = f'{material}_Rates_Comparison_FDM{fdm}_subfigs.pdf'
            savefile = savedir+file
        plt.savefig(savefile)

    plt.show()

    plt.close()
    return



def plotMeanFreePath(FDMn,plotConstraints=True,cmap_name='viridis', mX_range=(0.1, 1000), sigmaE_range=None):
    """Plot DM mean free path through Earth for given form factor.
    
    Args:
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        plotConstraints: Plot experimental constraints (default True)
        cmap_name: matplotlib Colormap name (default 'viridis')
        mX_range: Tuple of (min, max) for DM mass in MeV (default (0.1, 1000))
        sigmaE_range: Tuple of (min, max) for cross section in cm^2 (default depends on FDMn)
    """
    import numpy as np
    from tqdm.autonotebook import tqdm
    import sys
    from MeanFreePath import Earth_Density_Layer_NU
    import matplotlib
    import matplotlib.ticker as ticker
    # sigmaEs = np.arange(-40,-28,1)
    # sigmaEs = np.arange(-40,-26,(-26 + 40)/1000)
    # sigmaEs = 10**(sigmaEs)
    # # mX_array = np.concatenate((np.arange(0.2,0.8,0.025),np.array([0.9]),np.arange(1,5,0.05),np.arange(5,11,1),np.array([20,50,100,200,500,1000])))
    # mX_array_heavy = np.concatenate((np.arange(0.2,10,0.025),np.arange(10,1500,0.1)))

    if sigmaE_range is None:
        if FDMn == 0:
            sigmaE_range = (1e-42, 1e-28)
        elif FDMn == 2:
            sigmaE_range = (1e-38, 1e-29)
        else:
            sigmaE_range = (1e-42, 1e-28)

    mX_array = np.geomspace(mX_range[0], mX_range[1], 100)
    sigmaEs = np.geomspace(sigmaE_range[0], sigmaE_range[1], 100)

    #np.arange(0.1,1500,0.1)
    EDLNU = Earth_Density_Layer_NU()
    r_test = 0.8*EDLNU.EarthRadius
    vMax = 300 * EDLNU.km / EDLNU.sec

    MFP = []
    for s in range(len(sigmaEs)):
        MFP_small = []
        for m in range(len(mX_array)):
            mX = mX_array[m]*1e-3 #GeV
            sigmaP= sigmaEs[s] * (EDLNU.muXElem(mX,EDLNU.mProton) / EDLNU.muXElem(mX,EDLNU.mElectron))**2

            mfp = EDLNU.Mean_Free_Path(r_test,mX,sigmaP,vMax,FDMn,doScreen=True)
            MFP_small.append(mfp)
        MFP_small = np.array(MFP_small)
        MFP.append(MFP_small)
    MFP = np.array(MFP)


    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib

    # plotting specifications

    #Options

    smallest = 16
    smaller = 24
    medium = 32
    large = 40

    set_default_plotting_params(fontsize=medium)
   



    cmap = plt.get_cmap(cmap_name) # default color cycle, call by using color=cmap(i) i=0 is blue


    color_re = 'black'




    golden = (1 + 5 ** 0.5) / 2
    goldenx = 16
    goldeny = goldenx / golden
    plt.figure(figsize=(goldenx,goldeny))


    plt.xlim(mX_range[0], mX_range[1])
    
    plt.xscale('log')
    plt.yscale('log')
    if FDMn == 0:
        medtitstr= 'Heavy'
        xy = (4,5e-35)
        plt.ylim(sigmaE_range[0], sigmaE_range[1])
    elif FDMn == 2:
        xy = (7,1e-34)
        plt.ylim(sigmaE_range[0], sigmaE_range[1])
        medtitstr = 'Light'
    plt.title(f'{medtitstr} Mediator Mean Free Path' +  ' [$R_{\\oplus}$]',fontsize=large,y=1.01)
    plt.xlabel('$m_\chi$ [MeV]')
    plt.ylabel('$\overline{\sigma}_e$ [cm$^2$]')
    plt.tight_layout()

    
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))

    if FDMn == 0:
        plt.text(0.85,0.23,'MFP $= R_{\\bigoplus}$',fontsize=medium,color=color_re,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    elif FDMn == 2:
        plt.text(0.40,0.46,'MFP $= R_{\\bigoplus}$',fontsize=medium,color=color_re,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)



    #plotting


    lower = np.floor(np.log10(MFP.min())-1)
    upper = np.ceil(np.log10(MFP.max())+1)
    num_steps = 100
    lev_exp = np.arange(lower,upper)
    levs = np.power(10,lev_exp)

    CT_MFP= plt.contourf(mX_array,sigmaEs,MFP,levs,norm=colors.LogNorm(),locator=plt.LogLocator(),cmap=cmap)

    CTL_MFP = plt.contour(mX_array,sigmaEs,MFP,levs,cmap=plt.get_cmap('binary'),linewidths=0)

    CTI_MFP = plt.contour(mX_array,sigmaEs,MFP,levs,cmap=plt.get_cmap('binary'),alpha = 0)
    if FDMn == 0:
        index = 14
    elif FDMn == 2:
        index = 8
    zeropos = CTI_MFP.allsegs[index][0]
    plt.plot(zeropos[:, 0], zeropos[:, 1], color=color_re,lw=3,ls='dashdot')


    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()

    clevs =  CTL_MFP.levels

    clevs = np.array(clevs)
    oddlevs = clevs[::2]
    evenlevs = clevs[1::2]
    if 1e0 in oddlevs:

        newlevs = oddlevs[oddlevs> 1e-6]
    else:
        newlevs = evenlevs[evenlevs> 1e-6]
        
    plt.clabel(CTL_MFP,newlevs, fmt=fmt)

   
    if plotConstraints:
        import sys
        sys.path.append('../limits/')
        from Constraints import plot_constraints

        x,y = plot_constraints('All',FDMn)
        plt.plot(x,y,color='black',lw=3)
        x,y = plot_constraints('Solar',FDMn)
        plt.plot(x,y,color='black',lw=3,ls='--')
        if FDMn == 0:
            x,y = plot_constraints('Migdal',FDMn)
            plt.plot(x,y,color='black',lw=3,ls=':')


    if FDMn == 0:
        xy = (40,3e-42)
        xysolar = (0.65,7e-39)
        title_str = "Heavy"

    elif FDMn == 2:
        xy = (8,3e-38)
        title_str = "Light"
        xysolar = (0.65,3e-35)

    plt.annotate('Halo DM',xy,fontsize=medium)

    plt.annotate('SRDM',xysolar,fontsize=smallest)



    plt.savefig(f'figures/Misc/{title_str}Mediator_MFP.pdf')
    plt.show()
    plt.close()



def plotLocationExposure(address1,address2,savefig=True):
    """Plot isoangle over a day/ effective exposure over a day for two locations.
    
    Args:
        address1: First location address
        address2: Second location address
        savefig: Save figure to file (default True)
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    from astropy.coordinates import SkyCoord,AltAz
    from astropy.time import Time
    import astropy.units as u
    from astropy.utils import iers
    from isoangle import get_site_location
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    # plotting specifications
    import matplotlib.pyplot as plt

    #Options
    small = 16
    large= 24
    medium = 20
    set_default_plotting_params(fontsize=small)
    
    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    sidereal_day = 23.9344696
    th=np.arange(0,sidereal_day,0.001) #change to sidereal day, finer bin
    t=[]
    for i in th:
        t.append(Time('2025-3-8 00:00:00')+i*u.hour)

    wind=SkyCoord(l=90*u.deg,b=0*u.deg,frame="galactic")

    loc_aliases = {
        "SNOLAB": "SNO",
        "SNO": "SNO",
        "SNOlab, Canada".upper(): "SNO",
        "SUPL": "SUPL",
        "STAWELL": "SUPL",
        "STAWELL, AUSTRALIA": "SUPL",
        "BARILOCHE": "BRC",
        "SAN CARLOS DE BARILOCHE, ARGENTINA": "BRC",
        "FERMILAB": "FNAL",
        "FERMILAB, USA": "FNAL",
        "GRAN SASSO": "GSSI",
        "MODANE": "MODANE",
        "SOUDAN": "SOUDAN",
        "CAPE TOWN": "CAPETOWN",
    }

    def _resolve_site_key(name):
        key = name.strip().upper()
        resolved = loc_aliases.get(key, key)
        try:
            get_site_location(resolved)
        except KeyError as exc:
            raise ValueError(
                f"Unknown location '{name}'. "
                f"Supported names include SNOLAB/SNO, SUPL/Stawell, "
                "Bariloche, Fermilab, Gran Sasso, Modane, Soudan, Cape Town."
            ) from exc
        return resolved

    site_key1 = _resolve_site_key(address1)
    site_key2 = _resolve_site_key(address2)
    loc1 = get_site_location(site_key1)
    loc2 = get_site_location(site_key2)

    wimploc1 = wind.transform_to(AltAz(obstime=t,location=loc1))
    wimploc2 = wind.transform_to(AltAz(obstime=t,location=loc2))



    isoloc1=90-wimploc1.alt.deg
    isoloc2=90-wimploc2.alt.deg

    # fig, ax = plt.subplots(2)
    colorlist = [northcolor,southcolor]

    plt.figure(figsize=(15, 5),dpi=80)
    plt.subplot(1,2,1)
    plt.plot(th,isoloc1,label=address1,color=colorlist[0],lw=2)

    plt.plot(th,isoloc2,label=address2,color=colorlist[1],lw=2)

    plt.xlim(0,24)
    plt.ylim(-5,185)
    plt.legend()
    plt.ylabel("Isoangle [degrees]")
    plt.xlabel("UTC time of day on March 8th 2025 [hours]")
    plt.grid()
    ax=plt.gca()
    ax.xaxis.set_ticks(np.arange(0, 24.01, 4))
    ax.yaxis.set_ticks(np.arange(0, 180.01, 15))
    plt.subplot(1,2,2)
    hb=np.linspace(0,180,int(180*4))
    plt.hist(isoloc1,label=address1,bins=hb,histtype=u'step',color=colorlist[0])
    plt.hist(isoloc2,label=address2,bins=hb,histtype=u'step',color=colorlist[1])
    plt.xlabel("Isoangle")
    plt.ylabel("Exposure (arb. units)")
    plt.xticks(np.linspace(0,180,19)[::2])
    plt.legend()
   
    if savefig:
        plt.savefig(f'figures/Misc/IsoLoc.pdf')
    if "agg" not in plt.get_backend().lower():
        plt.show(block=False)
        plt.pause(0.001)
    plt.close()



    return


def significance(average,fracamp,exposure,background):
    """Calculate statistical significance for modulation amplitude.
    
    Args:
        average: Average event rate
        fracamp: Fractional modulation amplitude
        exposure: Experiment exposure
        background: Background rate
        
    Returns:
        Significance value
    """
    import numpy as np
    numerator = average * fracamp * exposure
    denominator = average + background
    denominator*= exposure
    denominator = np.sqrt(denominator)
    sig = numerator / denominator
    return sig


def find_sigma_cross_section(material,FDMn,test_mX,test_exposure,background,ne=1,loc='SNO',useVerne=True,fromFile=True,useQCDark = True,sigma=5,plot=False):
    """Find cross section needed for given significance.
    
    Args:
        material: Target material ('Si', 'Xe', 'Ar')
        FDMn: Form factor model (0=FDM1, 2=FDMq2)
        test_mX: DM mass in MeV
        test_exposure: Experiment exposure
        background: Background rate
        ne: Electron bin (default 1)
        loc: Location name (default 'SNO')
        useVerne: Use Verne distribution (default True)
        fromFile: Load rates from file (default True)
        useQCDark: Use form factor from QCDark (default True)
        sigma: Desired significance (default 5)
        plot: Make diagnostic plot (default False)
        
    Returns:
        Cross section in cm^2 required for given significance
    """
    from scipy.interpolate import Akima1DInterpolator 
    from tqdm.autonotebook import tqdm
    import os
    import re
    import numericalunits as nu


    import numpy as np
    calc_method_dict = {True: "verne", False: "damascus"}   

    qedict = {True: "_qcdark",False: "_qedark"}

    qestr = qedict[useQCDark] if material == 'Si' else ""

    halo_type = calc_method_dict[useVerne]
    screenstr = '_screened' if material == 'Si' else ""
    halo_dir = f'./{halo_type}_modulated_rates{screenstr}{qestr}_{material}/'
    sigmaEs = []
    file_list = os.listdir(halo_dir)
    for f in tqdm(range(len(file_list)),desc="Fetching Modulation Data"):
        file = file_list[f]
        if 'mX' not in file:
            continue
        mass_str = re.findall('mX_.+MeV',file)[0][3:-4]
        mX = float(mass_str.replace('_','.'))

        sigmaE = re.findall('sigmaE_.+_FD',file)[0][7:-3]

        sigmaE = float(sigmaE)
        Fdm= int(re.findall('FDM.+.csv',file)[0][3:-4])
        if Fdm != FDMn:
            continue
        sigmaEs.append(sigmaE)
    sigmaEs = np.array(sigmaEs)

    possible_cs = np.unique(np.sort(sigmaEs))
    true_cs = []
    significances = []
    for cs in possible_cs:
        try:
            minangle,maxangle = get_angle_limits(loc)
            test_amp = get_amplitude(test_mX,cs,FDMn,material,minangle,maxangle,ne=ne,fractional=False,useVerne = useVerne,fromFile=fromFile)
            test_amp_average = get_amplitude(test_mX,cs,FDMn,material,minangle,maxangle,ne=ne,fractional=False,useVerne = useVerne,fromFile=fromFile,returnaverage=True)
            test_amp_frac = get_amplitude(test_mX,cs,FDMn,material,minangle,maxangle,ne=ne,fractional=True,useVerne = useVerne,fromFile=fromFile)

            sig = significance(test_amp_average,test_amp_frac,test_exposure,background)
            significances.append(sig)
            true_cs.append(cs)
        except:
            continue
    true_cs = np.array(true_cs)
    significances = np.array(significances)

    significances = significances[np.argsort(true_cs)]
    true_cs = np.sort(true_cs)



    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f'mX {test_mX}, FDMn{FDMn}')
        plt.xlabel('cross-section')
        plt.ylabel("Significance")
        plt.xscale('log')
        plt.plot(true_cs,significances)


        plt.show()
        plt.close()

    crop_indices = significances < 1000
    significances_x = significances[crop_indices]
    cs_y = true_cs[crop_indices]

    sig_interp = Akima1DInterpolator(significances_x,cs_y)
    # print(significances_x,cs_y)
    sig5_cs = sig_interp(sigma)
    return sig5_cs




def plot_silicon_1e_limit_comparison(plotsig=False):
    """Plot comparison of direct vs modulation limits for silicon 1e bin.
    
    Args:
        plotsig: Plot significance curves (default False)
    """
     
    background_rates = [40,30,20,10,5,1,0.1] #events /g/day
    import numpy as np
    import matplotlib.lines as mlines

    exp = 1 * nu.kg * nu.day
    kgyear = 1 * nu.kg * nu.year
    mod_5sigma_limits = []
    mod_2sigma_limits = []
    mod_2sigma_limits_kgyear = []

    for background in background_rates:
        background_rate = background / nu.g / nu.day
        lim = find_sigma_cross_section('Si',2,1.,exp,background_rate,ne=1,loc='SNO',useVerne=True,fromFile=True,useQCDark = True,plot=plotsig)
        lim1 = find_sigma_cross_section('Si',2,1.,exp,background_rate,ne=1,loc='SNO',useVerne=True,fromFile=True,useQCDark = True,sigma=2)
        mod_2sigma_limits.append(lim1)
        lim1kgyear = find_sigma_cross_section('Si',2,1.,kgyear,background_rate,ne=1,loc='SNO',useVerne=True,fromFile=True,useQCDark = True,sigma=2)
        mod_2sigma_limits_kgyear.append(lim1kgyear)
    
        mod_5sigma_limits.append(lim)
    mod_5sigma_limits = np.array(mod_5sigma_limits)
    mod_2sigma_limits = np.array(mod_2sigma_limits)
    mod_2sigma_limits_kgyear = np.array(mod_2sigma_limits_kgyear)

    #1 MeV Limits as function of background (g-day)
    import numpy as np
    direct_cs_limits = np.array([3.8170770113468556e-33,
    2.8678251847968514e-33,
    1.9157631409445324e-33,
    9.629290521658613e-34,
    4.846533143572661e-34,
    9.988858693071315e-35,
    1.1162567592194836e-35]) #for 1 kg day
    import matplotlib.pyplot as plt
    set_default_plotting_params(fontsize=24)
    plt.figure(figsize=(8,6))
    small = 12
    medium = 16
    large = 24

    ax = plt.gca()
    plt.plot([b*1000 for b in background_rates],direct_cs_limits,label='Direct 95\% Confidence',color='red')
    # plt.plot([b*1000 for b in background_rates],mod_5sigma_limits,label='Modulation 5$\sigma$ Discovery',color='steelblue')
    plt.plot([b*1000 for b in background_rates],mod_2sigma_limits,label='Modulation 2$\sigma$ Discovery  1 kg-day',color=northcolor)
    plt.plot([b*1000 for b in background_rates],mod_2sigma_limits_kgyear,label='Modulation 2$\sigma$ Discovery 1 kg-year',color=northcolor,ls='--')

    ax.invert_xaxis()
    plt.yscale('log')
    plt.xscale('log')
    plt.title("$F_{\mathrm{DM}} = (\\alpha m_e/q)^2$ Limit",fontsize=large)
    plt.xlabel("Background Rate [events/kg/day] ")
    plt.ylabel('$\overline{\sigma}_e$ [cm$^2$]')
    plt.text(0.99,0.95,'Si',fontsize=large,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.88,'SNOLAB',fontsize=medium,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.81,'$m_\chi$ = 1 MeV',fontsize=medium,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    # plt.text(0.99,0.77,'1 kg-day',fontsize=medium,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.ylim(1e-36,1e-32)

    red = mlines.Line2D([], [], color='red', label='Direct 95\%')
    blue = mlines.Line2D([], [], color=northcolor, label='Modulation 2$\sigma$')

    legend1 = ax.legend(handles=[red,blue],loc='lower left',fontsize=medium,frameon=False)
    ax.add_artist(legend1)

    dashed = mlines.Line2D([], [], color='black', linestyle='--', label='1 kg-year')
    solid = mlines.Line2D([], [], color='black', label='1 kg-day')
    legend2 = ax.legend(handles=[solid, dashed], fontsize=medium, loc='lower right',frameon=False)


    plt.savefig('figures/Silicon/direct_mod_limit_comparison_fdmq2.pdf')
    plt.show()
    plt.close()
