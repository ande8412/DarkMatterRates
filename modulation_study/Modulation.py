import numericalunits as nu

def get_modulated_rates(material,mX,sigmaE,fdm,ne,useVerne=True,calcError=None,useQCDark=True,DoScreen = True,verbose = False,flat=False,dmRateObject = None):
    import os
    import torch
    import sys
    sys.path.append('..')

    if dmRateObject is not None:
        dmrates = dmRateObject
    else:
        import DMeRates
        import DMeRates.DMeRate as DMeRate
        dmrates = DMeRate.DMeRate(material,QEDark= not useQCDark)

    if useQCDark:
        integrate = True
    else:
        integrate = False

    fdm_dict = {0: "Scr", 2: "LM"}    
    calc_method_dict = {True: "Verne", False: "Parameter_Scan"}    

    dmrates.update_crosssection(sigmaE)

    halo_model = 'modulated'

    fdm_str = fdm_dict[fdm]
    calc_str = calc_method_dict[useVerne]
    mass_str = str(mX).replace('.','_')
    loc_dir = f'../halo_data/modulated/{calc_str}_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'

    if type(ne) == int:
        ne = [ne]
    # else:
    #     loc_dir = f'./halo_data/modulated/Parameter_Scan_{fdm_str}/mDM_{mass_str}_MeV_sigmaE_{sigmaE}_cm2/'


    if os.path.isdir(loc_dir) and len(os.listdir(loc_dir)) > 0:
        dir_contents = os.listdir(loc_dir)
        dir_contents = [i for i in dir_contents if i != '.DS_Store']
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
                    result = dmrates.calculate_rates(mX,halo_model,fdm,ne,integrate=integrate,DoScreen=DoScreen,isoangle=isoangle,useVerne=useVerne,calcErrors=calcError).flatten()
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

    
    
def generate_modulated_rates(material,FDMn,useQCDark = True,useVerne=True,calcError=None,doScreen=True,overwrite=False,verbose=False,save=True):
    import csv
    import re
    import numpy as np
    import os
    import sys
    sys.path.append('..')
    from tqdm.autonotebook import tqdm
    import numericalunits as nu
    import DMeRates
    import DMeRates.DMeRate as DMeRate
    dmrates = DMeRate.DMeRate(material,QEDark= not useQCDark)




    fdm_dict = {0: "Scr", 2: "LM"}    
    calc_method_dict = {True: "Verne", False: "Parameter_Scan"}  

    scr_dict = {True: "_screened", False: "_unscreened"}  

    qedict = {True: "_qcdark",False: "_qedark"}
    
    

    nes = [1,2,3,4,5,6,7,8,9,10]

    halo_model = 'modulated'
    scr_str = scr_dict[doScreen] if material == 'Si' else ""
    qestr = qedict[useQCDark]
    if material != 'Si':
        qestr = ''

    fdm_str = fdm_dict[FDMn]
    calc_str = calc_method_dict[useVerne]



    write_dir= f'damascus_modulated_rates{scr_str}{qestr}_{material}'
    

    if useVerne:
        write_dir= f'verne_modulated_rates{scr_str}{qestr}_{material}'
        calcError=None



    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    module_dir = os.path.dirname(__file__)
    halodir = os.path.join(module_dir,f'../halo_data/modulated/{calc_str}_{fdm_str}/')

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

        
        isoangles,rate_per_angle = get_modulated_rates(material,mX,sigmaE,FDMn,nes,useVerne=useVerne,calcError=calcError,useQCDark=useQCDark,DoScreen = doScreen,verbose = verbose,flat=False, dmRateObject = dmrates)
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

def generate_damascus_rates_with_error(ne,material,FDMn,useQCDark = True,DoScreen=True,overwrite=False,verbose=False,save=True,fit=False):
    import csv
    import re
    import numpy as np
    import os
    import sys
    sys.path.append('..')
    from tqdm.autonotebook import tqdm
    import numericalunits as nu
    import DMeRates
    import DMeRates.DMeRate as DMeRate
    dmrates = DMeRate.DMeRate(material,QEDark= not useQCDark)




    fdm_dict = {0: "Scr", 2: "LM"}    

    scr_dict = {True: "_screened", False: "_unscreened"}  

    qedict = {True: "_qcdark",False: "_qedark"}
    
    

    halo_model = 'modulated'
    scr_str = scr_dict[DoScreen] if material == 'Si' else ""
    qestr = qedict[useQCDark]
    if material != 'Si':
        qestr = ''

    fdm_str = fdm_dict[FDMn]



    write_dir= f'damascus_modulated_rates_{ne}e{scr_str}{qestr}_{material}'
    
    write_dir_fit= f'fitted_damascus_modulated_rates_{ne}e{scr_str}{qestr}_{material}'



    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    if not os.path.isdir(write_dir_fit):
        os.mkdir(write_dir_fit)

    module_dir = os.path.dirname(__file__)
    halodir = os.path.join(module_dir,f'../halo_data/modulated/Parameter_Scan_{fdm_str}/')


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




def hyp_tan_ff(theta,a,theta_0,theta_s,ff):
        import numpy as np
        #rbar is mean
        #a = amplitude
        #theta is angle
        #theta_0 is transition angle
        #theta_s is slope fit
        return (a/2)*np.tanh((theta-theta_0)/theta_s) + ff

def fitted_rates(angles,rates,rates_err=None,linear=False):
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







def plot_damascus_output(test_mX,FDMn,cross_section,long=True,savefig=False):
    import os
    from scipy.interpolate import CubicSpline
    import torch
    import numpy as np
    
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    params = {'text.usetex' : True,
            'font.size' : 40,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=40
    


    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden

    plt.rcParams['figure.figsize']=(16,12)
    plt.figure()

    # mediator = "LM"
    fdm_dict = {0: "Scr", 2: "LM"}    
    mediator = fdm_dict[FDMn]

    mX_str = float(test_mX)
    mX_str = np.round(mX_str,3)

    # if mass_string.is_integer():
    #     mass_string = int(mass_string)
    # else:
    mX_str = str(mX_str)
    mX_str = mX_str.replace('.',"_")
    # QE_Modulated_Halo = QEDark()
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate



    dmrates = DMeRate.DMeRate('Si')

   
    dmrates.update_params(220,232,544,0.3e9,1e-36)
    vhigh = 3*(dmrates.vEarth + dmrates.vEscape)
    vMins = np.linspace(0,vhigh,1000)

    
    # fig,ax = plt.subplots(figsize=(15,10))
    shm_etas = []
    for v in vMins:
        shm_eta = dmrates.DM_Halo.etaSHM(v)
        shm_etas.append(shm_eta)
    shm_etas = np.array(shm_etas)
    shm_etas /= (nu.s / nu.km) #in s/km
    vMins/=(nu.km/nu.s)

    cmap = plt.get_cmap('viridis', 180) 
    if long:
        long_str = '_long'
        dirend = '_cluster'
    else:
        long_str = ''
        dirend = ''

    steps = len(os.listdir(f'../halo_data/modulated/Parameter_Scan_{mediator}{dirend}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2{long_str}/'))
    actual_angle = np.linspace(0,180,steps)
    for isoangle in range(steps):
        ai = actual_angle[isoangle]
        ai = round(ai)
        # print(isoangle,ai,cmap(ai))

        fname = f'../halo_data/modulated/Parameter_Scan_{mediator}{dirend}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2{long_str}/DM_Eta_theta_{isoangle}.txt'
        # fname_DAMASCUS = f'./DaMaSCUS/results/5MeV_test_histograms/eta.{isoangle}'
        fdata = np.loadtxt(fname,delimiter='\t')
        vmin = fdata[:,0]
        eta = fdata[:,1]

        plt.plot(vmin,eta,color=cmap(ai))


    
    plt.xlim([0, 700])

    ax = plt.gca()

    # EE_Index = 0
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180) 

    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm) 
    # for i in range(35):
    #     ax.plot(vMins[EE_Index,:],all_etas[i][EE_Index,:],color=cmap(i))
    # ax.set_xlim([0, 800])
    plt.plot(vMins,shm_etas,linewidth=4,ls=':',color='black',label='SHM')
    plt.legend(prop={'size': 32},loc=3)
    plt.xlabel('$v_{\mathrm{min}}$ [km/s]')
    plt.ylabel('$\eta$ [s/km]')
    ticks = np.linspace(0,180,19)[::2]
    clb = plt.colorbar(sm,ax=ax,ticks=ticks)
    clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)
    # fig.suptitle('Ansh Recreation $m_{dm}$' + f' = {test_mX} σ$_E$ = {cross_section} cm$^2$',fontsize=32)
    # title = '$m_{\chi} =$ ' + f'{test_mX} MeV' + ' $\overline{\sigma}_e = $ ' + cs_str
    # plt.title(title)

    if FDMn == 2:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    else:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    cs_str = r'${} \times 10^{{{}}}$'.format(*str(cross_section).split('e')) + 'cm$^2$'
    plt.text(0.99,0.87,'$\overline{\sigma}_{e} =$ ' + cs_str,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.80,'$m_\chi=$ ' + f'{test_mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)





    # fname_max = f'./halo_data/modulated/Parameter_Scan_{mediator}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2/DM_Eta_theta_{steps-1}.txt'
    # fname_min = f'./halo_data/modulated/Parameter_Scan_{mediator}/mDM_{mX_str}_MeV_sigmaE_{cross_section}_cm2/DM_Eta_theta_{0}.txt'

    # fdata_min = np.loadtxt(fname_min,delimiter='\t')
    # x1 = vmin = fdata_min[:,0]
    # y1 = fdata_min[:,1]

    # fdata_max = np.loadtxt(fname_max,delimiter='\t')
    # x2 = fdata_max[:,0]
    # y2  = fdata_max[:,1]
    # spl_high = CubicSpline(x2,y2)
    # spl_low= CubicSpline(x1,y1)

    # x = np.linspace(0,800,200)
    # y1 = spl_low(x)
    # y2 = spl_high(x)


    # polygon = plt.fill_between(x, y1, y2, lw=0, color='none')
    # xlim = plt.xlim()
    # ylim = plt.ylim()
    # verts = np.vstack([p.vertices for p in polygon.get_paths()])
    # gradient = plt.imshow(np.linspace(0, 1, 180).reshape(-1, 1), cmap='viridis', aspect='auto',
    #                     extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    # gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    # plt.xlim(xlim)
    # plt.ylim(ylim)



    # plt.tight_layout()
    # plt.title('$\eta$ Distribution vs Isoangle $\Theta$')
    if savefig:
        plt.savefig(f'figures/Misc/Eta_{test_mX}MeV_{cross_section}sigmaE_FDM{FDMn}.pdf')
    plt.show()
    plt.close



def mu_Xe(mX):
    me_eV = 5.1099894e5
    """
    DM-electron reduced mass
    """
    return mX*me_eV/(mX+me_eV)

def mu_XP(mX):

    """
    DM-proton reduced mass
    """
    mP_eV = 938.27208816 *1e6
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

def get_damascus_output(mX,sigmaE,FDMn):
    import numpy as np
    import os
    if FDMn == 0:
        dir_stir = 'Scr'
    else:
        dir_stir = 'LM'
    # mX= np.round(float(mX),2)
    mX_str = str(mX).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    ddir = f'halo_data/modulated/Parameter_Scan_{dir_stir}/mDM_{mX_str}_MeV_sigmaE_{sigmaE}_cm2/'
    data = []
    num_angles = len(os.listdir(ddir))
    for i in range(num_angles):
        file = ddir + f'DM_Eta_theta_{i}.txt'
        filedata = np.loadtxt(file,delimiter='\t')
        file_vmin = filedata[:,0]
        file_eta = filedata[:,1]
        data.append([file_vmin,file_eta])
    return data


def get_raw_damascus_output(dirname,mX,sigmaE,FDMn,rhoX=0.3):
    import os
    from scipy.interpolate import CubicSpline,Akima1DInterpolator,BarycentricInterpolator,PchipInterpolator
    import torch
    import sys
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate



    dmrates = DMeRate.DMeRate('Si')

    import numpy as np
    mX_str = float(mX)
    mX_str = np.round(mX_str,3)

    # if mass_string.is_integer():
    #     mass_string = int(mass_string)
    # else:
    mX_str = str(mX_str)
    mX_str = mX_str.replace('.',"_")
    dmrates.update_params(220,232,544,0.3e9,1e-36)
    vhigh = 3*(dmrates.vEarth + dmrates.vEscape)
    vMins = np.linspace(0,vhigh,1000)

    
    # fig,ax = plt.subplots(figsize=(15,10))
    shm_etas = []
    for v in vMins:
        shm_eta = dmrates.DM_Halo.etaSHM(v)
        shm_etas.append(shm_eta)
    shm_etas = np.array(shm_etas)
    shm_etas /= (nu.s / nu.km) #in s/km


    
   
    dir = f'/Users/ansh/Local/SENSEI/DaMaSCUS/results/{dirname}_histograms/'
    rhofile =  f'/Users/ansh/Local/SENSEI/DaMaSCUS/results/{dirname}.rho'

    km = 5.067*1e18
    s = 1.51905*1e24
    # print(rhofile)
    rhofiledata = np.loadtxt(rhofile,delimiter='\t')
    

    rho = rhofiledata[:,1]
    rho_i = rhofiledata[:,0]
    rho_func = PchipInterpolator(rho_i,rho)
    steps = len(os.listdir(dir)) //2
    actual_angle = np.linspace(0,180,steps)

    data = []

    for isoangle in range(steps):
        ai = actual_angle[isoangle]
        ai = round(ai)
        # print(isoangle,ai,cmap(ai))

        fname = f'{dir}eta.{isoangle}'
        # fname_DAMASCUS = f'./DaMaSCUS/results/5MeV_test_histograms/eta.{isoangle}'
        # print(fname)
        fdata = np.loadtxt(fname,delimiter='\t')
        vmin = fdata[:,0]* s/km
        eta = fdata[:,1]*km/s
        eta*=(rho_func(ai)/rhoX)
        filter_indices= np.where(vmin > 0)
        vmin = vmin[filter_indices]
        eta = eta[filter_indices]
        #interp

        eta_func = PchipInterpolator(vmin,eta)
        vmin_test = np.linspace(1,np.max(vmin),50)
        eta_interpolated = eta_func(vmin_test)
        data.append([vmin,eta])
    
    data.append([vMins,shm_etas])
    return data

def getVerneData(mX,sigmaE,FDMn):
    import numpy as np
    import os
    if FDMn == 0:
        dir_stir = 'Scr'
    else:
        dir_stir = 'LM'
    mX= np.round(float(mX),2)
    mX_str = str(mX).replace('.','_')
    sigmaE = float(format(sigmaE, '.3g'))
    ddir = f'../halo_data/modulated/Verne_{dir_stir}/mDM_{mX_str}_MeV_sigmaE_{sigmaE}_cm2/'
    data = []
    num_angles = len(os.listdir(ddir))
    for i in range(num_angles):
        file = ddir + f'DM_Eta_theta_{i}.txt'
        filedata = np.loadtxt(file,delimiter='\t')
        file_vmin = filedata[:,0]
        file_eta = filedata[:,1]
        data.append([file_vmin,file_eta])
    return data




def plot_raw_damascus_output(dirname,mX,sigmaE,FDMn,rhoX=0.3,logy=False,save=False):
    import os
    from scipy.interpolate import CubicSpline,Akima1DInterpolator,BarycentricInterpolator,PchipInterpolator
    import torch
    import numpy as np


    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    params = {'text.usetex' : True,
            'font.size' : 12,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=12
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate



    dmrates = DMeRate.DMeRate('Si')




    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden

    plt.rcParams['figure.figsize']=(6,4)
    plt.figure()

    # mediator = "LM"
    if FDMn == 0:
        fdm = 0
        mediator = "Scr"
    elif FDMn == 2:
        fdm = 2
        mediator = "LM"
    else:
        fdm = 0
    
    mX_str = float(mX)
    mX_str = np.round(mX_str,3)

    # if mass_string.is_integer():
    #     mass_string = int(mass_string)
    # else:
    mX_str = str(mX_str)
    mX_str = mX_str.replace('.',"_")

    dmrates.update_params(220,232,544,0.3e9,1e-36)
    vhigh = 3*(dmrates.vEarth + dmrates.vEscape)
    vMins = np.linspace(0,vhigh,1000)

    shm_etas = []
    for v in vMins:
        shm_eta = dmrates.DM_Halo.etaSHM(v)
        shm_etas.append(shm_eta)
    shm_etas = np.array(shm_etas)
    shm_etas /= (nu.s / nu.km) #in s/km


    cmap = plt.get_cmap('viridis', 180) 
   
    dir = f'/Users/ansh/Local/SENSEI/DaMaSCUS/results/{dirname}_histograms/'
    rhofile =  f'/Users/ansh/Local/SENSEI/DaMaSCUS/results/{dirname}.rho'

    km = 5.067*1e18
    s = 1.51905*1e24
    rhofiledata = np.loadtxt(rhofile,delimiter='\t')
    rho = rhofiledata[:,1]
    rho_i = rhofiledata[:,0]
    rho_func = PchipInterpolator(rho_i,rho)
    steps = len(os.listdir(dir)) //2
    actual_angle = np.linspace(0,180,steps)
    for isoangle in range(steps):
        ai = actual_angle[isoangle]
        ai = round(ai)
        # print(isoangle,ai,cmap(ai))

        fname = f'{dir}eta.{isoangle}'
        # fname_DAMASCUS = f'./DaMaSCUS/results/5MeV_test_histograms/eta.{isoangle}'
        fdata = np.loadtxt(fname,delimiter='\t')
        vmin = fdata[:,0]* s/km
        eta = fdata[:,1]*km/s
        eta*=(rho_func(ai)/rhoX)
        filter_indices= np.where(vmin > 0)
        vmin = vmin[filter_indices]
        eta = eta[filter_indices]
        #interp

        eta_func = PchipInterpolator(vmin,eta)


        plt.scatter(vmin,eta,color=cmap(ai))

        plt.plot(vmin,eta,color=cmap(ai))
        vmin_test = np.linspace(1,np.max(vmin),50)
        plt.plot(vmin_test,eta_func(vmin_test),color=cmap(ai),ls='dotted')



    
    plt.xlim([0, 700])

    ax = plt.gca()

    # EE_Index = 0
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180) 

    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm) 
    # for i in range(35):
    #     ax.plot(vMins[EE_Index,:],all_etas[i][EE_Index,:],color=cmap(i))
    # ax.set_xlim([0, 800])
    plt.plot(vMins,shm_etas,linewidth=4,ls=':',color='black',label='SHM')
    plt.legend(loc='lower right')
    plt.xlabel('$v_{\mathrm{min}}$ [km/s]')
    plt.ylabel('$\eta$ [s/km]')
    ticks = np.linspace(0,180,19)[::2]
    clb = plt.colorbar(sm,ax=ax,ticks=ticks)
    clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)
    # fig.suptitle('Ansh Recreation $m_{dm}$' + f' = {test_mX} σ$_E$ = {cross_section} cm$^2$',fontsize=32)
    # title = '$m_{\chi} =$ ' + f'{test_mX} MeV' + ' $\overline{\sigma}_e = $ ' + cs_str
    # plt.title(title)

    if FDMn == 2:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    elif FDMn == 0:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    else:
        plt.text(0.99,0.95,'$F_{\mathrm{DM}} = None$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.87,'$\overline{\sigma}_{e} =$ ' + str(sigmaE),color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.80,'$m_\chi=$ ' + f'{mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)

    if logy:
        plt.yscale('log')
    if not save:
        plt.show()
    else:
        plt.savefig(f'plotting/mX{mX}_sigmaE{sigmaE}_FDMn{FDMn}_raw_damascus_eta.jpg')
    plt.close()
    return



def point_checking(mX,sigmaE,FDMn,save=False,skipVerne=False,ne=1,useQCDark=True):
    
    
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    import numpy as np
    import sys
    sys.path.append('..')
    import DMeRates
    import DMeRates.DMeRate as DMeRate

    

    #font sizes
    small = 30
    smaller = 24
    medium = 36
    large = 40

    params = {'text.usetex' : True,
            'font.size' : medium,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=medium
    


    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden

    plt.rcParams['figure.figsize']=(16,12)


    sigmaP = sigmaE_to_sigmaP(sigmaE,mX)
    sigmaP = float(format(sigmaP, '.3g')) 
    mX = np.round(float(mX),3)
    mX_str = str(mX).replace('.','_')


    fig,ax = plt.subplots(2,3,figsize=(36,24))
    fig.delaxes(ax[1,2]) # The indexing is zero-based here
    name = f'mX{mX_str}_sigma{sigmaP}_fdm{FDMn}.cfg'
    damascus_found  = True
    try:
        damascus_output_data = get_raw_damascus_output(name,mX,sigmaE,FDMn,rhoX=0.3)
    except:
        try:
            damascus_output_data = get_damascus_output(mX,sigmaE,FDMn)
        except:
            damascus_found = False
            print('No Verne Data found')
    # shm = damascus_output_data[-1]
    if not skipVerne:
        verne_data = getVerneData(mX,sigmaE,FDMn)


    cmap = plt.get_cmap('viridis', 180) 
    norm = matplotlib.colors.Normalize(vmin=0, vmax=180) 
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm) 
    # from QEDark.QEDarkConstants import lightSpeed
    # vmins = np.linspace(0,800,100)
    # imb_etas  =eta_SHM(vmins*1000/lightSpeed)
    if damascus_found:

        num_angles = len(damascus_output_data) - 1
        isoangles = np.linspace(0,180,num_angles)
        for i in range(num_angles):
            ai =int(np.round(isoangles[i]))
            vmin = np.array(damascus_output_data[i][0])
            eta = np.array(damascus_output_data[i][1])
            ax[1,0].plot(vmin,eta,color=cmap(ai))


        

        


        # ax[1,0].plot(vmins,imb_etas,linewidth=4,ls=':',color='black',label='SHM')
        ax[1,0].legend(loc='lower right',fontsize=small)
    ax[1,0].set_xlabel('$v_{\mathrm{min}}$ [km/s]')
    ax[1,0].set_ylabel('$\eta$ [s/km]')
    ax[1,0].set_title("DaMaSCUS Eta Distribution")
    ticks = np.linspace(0,180,19)[::2]
    clb = plt.colorbar(sm,ax=ax[1,0],ticks=ticks)
    clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)



    if FDMn == 2:
        ax[1,0].text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,0].transAxes)
    elif FDMn == 0:
        ax[1,0].text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,0].transAxes)
    else:
        ax[1,0].text(0.99,0.95,'$F_{\mathrm{DM}} = None$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,0].transAxes)
    ax[1,0].text(0.99,0.87,'$\overline{\sigma}_{e} =$ ' + str(sigmaE),color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,0].transAxes)
    ax[1,0].text(0.99,0.80,'$m_\chi=$ ' + f'{mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,0].transAxes)



    if not skipVerne:
        num_angles = len(verne_data)
        isoangles = np.linspace(0,180,num_angles)
        for i in range(num_angles):
            ai =int(np.round(isoangles[i]))
            vmin = np.array(verne_data[i][0])
            eta = np.array(verne_data[i][1])
            ax[1,1].plot(vmin,eta,color=cmap(ai))

        # ax[1,1].plot(vmins,imb_etas,linewidth=4,ls=':',color='black',label='SHM')
        ax[1,1].legend(loc='lower right',fontsize=small)
        ax[1,1].set_xlabel('$v_{\mathrm{min}}$ [km/s]')
        ax[1,1].set_ylabel('$\eta$ [s/km]')
        ax[1,1].set_title("Verne Eta Distribution")
        ticks = np.linspace(0,180,19)[::2]
        clb = plt.colorbar(sm,ax=ax[1,1],ticks=ticks)
        clb.ax.set_title('$\Theta$\N{degree sign}',horizontalalignment='center',x=0.8)


        if FDMn == 2:
            ax[1,1].text(0.99,0.95,'$F_{\mathrm{DM}} = \\alpha m_e / q^2$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,1].transAxes)
        elif FDMn == 0:
            ax[1,1].text(0.99,0.95,'$F_{\mathrm{DM}} = 1$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,1].transAxes)
        else:
            ax[1,1].text(0.99,0.95,'$F_{\mathrm{DM}} = None$',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,1].transAxes)
        ax[1,1].text(0.99,0.87,'$\overline{\sigma}_{e} =$ ' + str(sigmaE),color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,1].transAxes)
        ax[1,1].text(0.99,0.80,'$m_\chi=$ ' + f'{mX} MeV',color='black',horizontalalignment='right',verticalalignment='center',transform = ax[1,1].transAxes)






    materials = ['Si', 'Xe', 'Ar']
    for j,mat in enumerate(materials):
        plot_ax = ax[0,j]
        angles,rate_high = get_modulated_rates(mat,mX,sigmaE,FDMn,useVerne=False,calcError="High",ne=ne,useQCDark=useQCDark) 
        angles,rate_low = get_modulated_rates(mat,mX,sigmaE,FDMn,useVerne=False,calcError="Low",ne=ne,useQCDark=useQCDark) 
        
        angles,rate = get_modulated_rates(mat,mX,sigmaE,FDMn,useVerne=False,calcError=None,ne=ne,useQCDark=useQCDark) * nu.kg * nu.day
        if not skipVerne:
            angles_v,rate_verne = get_modulated_rates(mat,np.round(mX,2),sigmaE,FDMn,useVerne=True,calcError=None,ne=ne,useQCDark=useQCDark) * nu.kg * nu.day

        rate_high*= nu.kg * nu.day
        rate_low *= nu.kg * nu.day
        rate_verne*= nu.kg * nu.day
        rate*= nu.kg * nu.day


        rate_err = rate_high - rate
            
        if useQCDark:
            dmrates = DMeRate.DMeRate(mat)
            integrate = True
        else:
            dmrates = DMeRate.DMeRate(mat,QEDark=True)
            integrate = False
        dmrates.update_crosssection(sigmaE)

        dmrates.update_params(238,250,544,0.3e9,sigmaE)

        result_flat = dmrates.calculate_rates(mX,'shm',FDMn,ne,integrate=integrate,DoScreen=True,isoangle=None,useVerne=False) * nu.kg *nu.day

    
        # kg_year = float(base_result)
        # g_year = kg_year / 1000
        # g_day = g_year * (1/365)
        base_result = np.ones_like(angles) * result_flat
        


        try:
            angle_grid,fit_vector,parameters,error = fitted_rates(angles,rate,rate_err)
        except ValueError:
            angle_grid,fit_vector,parameters,error = fitted_rates(angles,rate,rate_err,linear=True)
        fit = fit_vector[0]
        fit_upper = fit_vector[1]
        fit_lower = fit_vector[2]

        plot_ax.plot(angle_grid,fit,color='red',label="Fit")
        plot_ax.fill_between(angle_grid,fit_lower,fit_upper,color='red',label="Fit Uncertainty",alpha=0.3)
        
        if len(parameters) > 3:
            inflection = parameters[1]
            inflection_err = error[1]
            plot_ax.axvspan(inflection-inflection_err,inflection+inflection_err,alpha=0.3,label="Inflection band")
            fit_type = fit_vector[-1]

            amp = (np.max(fit) - np.min(fit)) / 2
            frac_amp = amp / np.mean(fit)
            frac_amp = np.round(frac_amp,2)
            amp = np.round(amp,2)
            plot_ax.text(0.03, 0.25, f'frac amp = {frac_amp}, amp = {amp}',
        horizontalalignment='left',
        verticalalignment='center',
        transform =plot_ax.transAxes,c='blue',fontsize=smaller)


        else: #fit failed, returned a linear regression
            slope = parameters[0]
            intercept = parameters[1]
            r = parameters[2]
            r_squared = r**2
            p = error[0]
            std_err = error[1]
            intercept_stderr = error[2]
            
            fit_type = 'Linear'

        round_values = []
        mse = fit_vector[3]
        rmse = fit_vector[4]
        ssr = fit_vector[5]
        # [linear_fit,linear_fit_upper,linear_fit_lower,mse_linear,rmse_linear,ssr_linear]
        # mse_fit = np.mean((rate - fit)**2)
        # rmse_fit = np.sqrt(mse_fit)
        
        # ssr_fit =  ((rate - fit) ** 2).sum()
        
        for err in [ssr,mse,rmse]:
            if err > 1:
                round_values.append(1)
            else:
                round_values.append(5)


        plot_ax.text(0.03, 0.03, f'{fit_type} SSR = {np.round(ssr,int(round_values[0]))} MSE = {np.round(mse,int(round_values[1]))} RMSE = {np.round(rmse,int(round_values[2]))}',
        horizontalalignment='left',
        verticalalignment='center',
        transform =plot_ax.transAxes,c='red',fontsize=smaller)





        # print("Si")
        # print(len(angles),len(rate))

        plot_ax.errorbar(angles,rate,yerr=rate_err,label="Uncertainty",linestyle='')
        plot_ax.scatter(angles,rate,label='Data')
        plot_ax.plot(angles,base_result,color='green',label="No Modulation")  
        if not skipVerne:   
            plot_ax.plot(angles_v,rate_verne,label='Verne',ls='--')

        if FDMn == 0:
            plot_ax.set_title(f"Heavy Mediator {materials[j]} {ne} e$^-$")
        elif FDMn == 2:
            plot_ax.set_title(f"Light Mediator {materials[j]} {ne} e$^-$")
        plot_ax.set_ylabel('Rate [events/g/day]')
        plot_ax.set_xlabel('Isoangle')
        plot_ax.set_xlim(0,180)
        plot_ax.set_xticks(np.linspace(0,180,19)[::2])
        plot_ax.legend(fontsize=smaller,loc='upper right')

    if not save:
        plt.show()
    else:
        plt.savefig(f'plotting/{mX_str}mX_{sigmaE}cm2_FDMn{FDMn}_ratechecking.jpg')
    plt.close()

    return 
    
    


def eta_SHM(vmins):
    import numpy as np
    from scipy.special import erf
    from QEDarkConstants import lightSpeed
    eta = np.zeros_like(vmins)
    vEarth = 250.2e3 #m/s
    vEscape = 544e3 #m/s
    v0 = 238e3 #m/s
    vEarth/=lightSpeed
    vEscape/=lightSpeed
    v0/=lightSpeed
    

    val_below = -4.0*vEarth*np.exp(-(vEscape/v0)**2) + np.sqrt(np.pi)*v0*(erf((vmins+vEarth)/v0) - erf((vmins - vEarth)/v0))

    val_above = -2.0*(vEarth+vEscape-vmins)*np.exp(-(vEscape/v0)**2) + np.sqrt(np.pi)*v0*(erf(vEscape/v0) - erf((vmins - vEarth)/v0))

    above_mask = vmins < vEscape + vEarth
    eta = np.where(above_mask,val_below,eta)
    below_mask = vmins < vEscape - vEarth
    eta = np.where(below_mask,val_below,eta)
    

    K = (v0**3)*(-2.0*np.pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (np.pi**1.5)*erf(vEscape/v0))

    etas = (v0**2)*np.pi/(2.0*vEarth*K)*eta #units of c^-1
    etas/=lightSpeed #convert to s/m
    etas*=1e3#convert to s/km
    #not sure if etas is allowed to be zero.
    etas[np.where(etas < 0)] = 0
    return etas



def get_angle_limits(loc,date=[8,8,2024]):
    
    import numpy as np
    from scipy.interpolate import CubicSpline
    # try:
    from isoangle import ThetaIso,sites,FracDays
    if loc == 'SNOLAB':
        loc_key = 'SNO'
    elif loc == 'Bariloche':
        loc_key = 'BRC'
    elif loc == 'Fermilab':
        loc_key = 'FNAL'
    else:
        loc_key = loc
    nlist1 = [FracDays(np.array(date),np.array([h,0,0])) for h in range(24)]
    y = [np.rad2deg(ThetaIso(sites[loc_key]['loc'],n)) for n in nlist1]

    x = [h for h in range(24)]

    xnew = np.linspace(0,24,num=1000)
    spl = CubicSpline(x,y)
    ynew = spl(xnew)
    min_angle = np.min(ynew)
    max_angle = np.max(ynew)
    # except:
    #     #no internet
    #     print("no internet access or this site is not defined, just returning snolabish")
    #     min_angle = 6
    #     max_angle = 89
    return min_angle,max_angle


def get_amplitude(mX,sigmaE,FDMn,material,min_angle,max_angle,ne=1,fractional=False,useVerne=False,verbose=False,fromFile=False,returnaverage=False,useQCDark=True,fit=None):

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

        file = f'./{type_str}_modulated_rates{screenstr}{qestr}_{material}/mX_{mass_str}_MeV_sigmaE_{sigmaE}_FDM{FDMn}.csv'
        

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
        isoangles,rate = get_modulated_rates(material,mX,sigmaE,FDMn,ne=ne,useVerne=useVerne,calcError=None,useQCDark=useQCDark)
        if not useVerne:
            isoangles_h,rate_high = get_modulated_rates(material,mX,sigmaE,FDMn,ne=ne,useVerne=useVerne,calcError='High')
                
                

        if not useVerne:
            rate_err = rate_high - rate


    # except:
    #     #something went wrong
    #     return -1
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
            try:
                angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=rate_err)
                
            except:
                try:
                    angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=None)
                    fitFailed_w_errors = True
                except:
                    fitFailed = True
                    

        if fitFailed:
            print(f'Warning, fit failed for this point mX = {mX} sigmaE = {sigmaE}')
            return np.nan

        # plt.scatter(isoangles,rate)

        # plt.plot(angle_grid,fit_vector[0])

        if fitFailed_w_errors:
            print(f'Warning, fit failed with errors for this point mX = {mX} sigmaE = {sigmaE}')

        lab_angles = np.linspace(min_angle,max_angle,100)
        try:
            lab_rate = hyp_tan_ff(lab_angles,*parameters)*np.mean(rate)
        except:
            print('fit returned a linear fit i think')
            return np.nan
        # plt.plot(lab_angles,lab_rate)
        # plt.show()
        # plt.close()
    else: # interpolate
        import numpy as np
        lab_angles = np.linspace(min_angle,max_angle,100)

        lab_rate = np.interp(lab_angles,isoangles,rate)



    amplitude = (np.max(lab_rate) - np.min(lab_rate))/ 2
    average = np.mean(lab_rate)
    fractional_amplitude = amplitude / average
    

    # if np.max(lab_rate) < 1e-10:
    #     return 0

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


    qedict = {True: "_qcdark",False: "_qedark"}

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
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    large = 24
    small = 16
    medium = 20
    params = {'text.usetex' : True,
            'font.size' : small,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=small
    plt.rcParams['figure.figsize']=(8,8)

    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue

    plot_nes = np.arange(1,len(nes) +2)


    fig = plt.figure(layout='constrained')
    ax = plt.gca()
    plt.xlabel("Q")
    
    colorlist = ['steelblue','black']
    plt.stairs(frac_amps_pt1_loc1,plot_nes,color=colorlist[0],lw=3)
    plt.stairs(frac_amps_pt1_loc2,plot_nes,color=colorlist[1],lw=3)
    plt.stairs(frac_amps_pt2_loc1,plot_nes,ls='--',color=colorlist[0],lw=3)
    plt.stairs(frac_amps_pt2_loc2,plot_nes,ls='--',color=colorlist[1],lw=3)

    if FDMn == 0:
        fdm_str = 'FDM $\propto$ 1'
    else:
        fdm_str = 'FDM $\propto$ $1/q^2$'

    if (FDMn ==0) and (material == 'Xe' or material == 'Ar'):
        x = 0.93
        y = 0.86
        
    else:
        x = 0.98
        y = 0.96
        
    plt.text(x, y, material,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c='Black',fontsize=large)

    plt.text(x, y-0.06, f'{fdm_str}',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c='Black',fontsize=medium)

    firstPointStr = '$m_\chi = $' + f'{mX1}  MeV' + ' $\overline{\sigma}_e =$ ' + f'{sigmaE1}' + ' cm$^2$'
    secondPointStr = '$m_\chi = $' + f'{mX2} MeV' + ' $\overline{\sigma}_e =$ ' + f'{sigmaE2}' + ' cm$^2$'
    plt.text(x, y-0.06-0.06, location1,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c=colorlist[0],fontsize=medium)
    plt.text(x, y-0.06-0.06-0.06, location2,horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c=colorlist[1],fontsize=medium)
    # ax.annotate("Dashed Line Annotation", xy=(0.99,.72), xytext=(0.8, .72), 
    #         arrowprops=dict(arrowstyle="-", linestyle="--"),transform=ax.transAxes)
    
    # plt.text(0.99, 0.72, 'DaMaSCUS',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes,c='Black',fontsize=medium,withdash=True)


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='-',lw=3),
                    Line2D([0], [0], color='black', linestyle='--',lw=3)]

    # Create the legendplt.legend(frameon=False, framealpha=0)
    if material == 'Si':
        xshift = 0.62
    else:
        if FDMn == 0:
            xshift = 0.64
        else:
            xshift = 0.66
    legend = plt.legend(custom_lines, [firstPointStr,secondPointStr],loc=(x-xshift,y-0.33),fontsize=medium,frameon=False,framealpha=0)
    legend.get_texts()[0].set_horizontalalignment('right')
    legend.get_texts()[1].set_horizontalalignment('right')
    plt.xticks(plot_nes)
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
        savedir = f'figures/{material}/'
        if fractional:
            frac_str = 'fractional_'
        else:
            frac_str = ''
        if useVerne:
            verne_str=  'verne'
        else:
            verne_str= 'damascus'
        name = f'{frac_str}modulation_amp_ne_bins_{location1}_vs_{location2}_FDM{FDMn}_{verne_str}_{qestr}.jpg'
        plt.savefig(savedir+name)
    plt.show()

    plt.close()

    


def getModulationAmplitudes(material,FDMn,location,fractional=False,useVerne=True,fromFile=True,verbose=False,ne=1,returnaverage=False,useQCDark=True):
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

    # if FDMn ==0:
    #     dir_str = 'Scr'
    # else:
    #     dir_str = "LM"
    # halo_dir = f'halo_data/modulated/{halo_type}_{dir_str}'
    screenstr = '_screened' if material == 'Si' else ""
    halo_dir = f'./{halo_type}_modulated_rates{screenstr}{qestr}_{material}/'


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

        amp = get_amplitude(mX,sigmaE,FDMn,material,min_angle,max_angle,fractional=fractional,useVerne=useVerne,verbose=verbose,fromFile=fromFile,ne=ne,returnaverage=returnaverage,useQCDark=useQCDark)


        amplitudes.append(amp)
        sigmaEs.append(sigmaE)
        masses.append(mX)


    sigmaEs = np.array(sigmaEs)
    masses = np.array(masses)

    amplitudes = np.array(amplitudes)

    return masses,sigmaEs,amplitudes



def getContourData(material,FDMn,location,fractional=False,useVerne=True,fromFile=True,verbose=False,getAll=True,masses=None,sigmaEs=None,ne=1,returnaverage=False,useQCDark=True,unitize=False):
    import numpy as np
    from scipy.interpolate import griddata
    masses,cross_sections,amplitudes = getModulationAmplitudes(material,FDMn,location,fractional=fractional,useVerne=useVerne,fromFile=fromFile,verbose=verbose,ne=ne,returnaverage=returnaverage,useQCDark=useQCDark)
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
        from math import log10, floor
        base10 = log10(number)
        return floor(base10)




                          

# def plotModulationContour(material,FDMn,location,fractional=False,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs =None,plotConstraints=True,savefig=False,ne=1,shadeMFP=False,massBounds=None,csBounds=None):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib
#     from matplotlib import colors
#     from matplotlib import cm, ticker

#     # plotting specifications
#     from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                 AutoMinorLocator)
#     from matplotlib.offsetbox import AnchoredText
#     #Options

#     large = 48
#     small = 36
#     medium = 40
#     smaller = 30
#     smallest=16
#     params = {'text.usetex' : True,
#         'font.size' : medium,
#             'font.family' : 'cmr10',
#             'figure.autolayout': True
#         }
#     plt.rcParams.update(params)
#     plt.rcParams['axes.unicode_minus']=False
#     plt.rcParams['axes.labelsize']=32
#     plt.rcParams['figure.figsize']=(16,12)
#     plt.rcParams['axes.formatter.use_mathtext']=True
#     if masses is None and sigmaEs is None:
#         getAll = True
#     else:
#         getAll = False
#     Masses,CrossSections,Amplitudes = getContourData(material,FDMn,location,fractional=fractional,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne)
#     if fractional:
#         Amplitudes[np.isnan(Amplitudes)] = 0
#         Amplitudes[Amplitudes == np.inf] = 0
#         Amplitudes[Amplitudes == -np.inf] = 0
        
#     else:
#         # Amplitudes[np.isnan(Amplitudes)] =0
#         Amplitudes[Amplitudes==0] = 1e-8

#     if masses is not None and sigmaEs is not None:
#         Amplitudes = Amplitudes.T

#     fig = plt.figure()
#     ax = plt.gca()
#     plt.xscale('log')
#     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#     plt.yscale('log')
#     if masses is not None:
#         mass_low = np.min(masses)
#         mass_high = np.max(masses)
#     elif massBounds is not None:
#         mass_high = massBounds[1]
#         mass_low = massBounds[0]
#     else:
#         mass_low = 0.5
#         mass_high = 1000
#     if sigmaEs is not None:
#         cs_low = np.min(sigmaEs)
#         cs_high = np.max(sigmaEs)
#     elif csBounds is not None:
#         cs_high = csBounds[1]
#         cs_low = csBounds[0]
#     else:
#         cs_low = np.min(CrossSections)
#         cs_high = np.max(CrossSections)
#     if FDMn == 2 and csBounds is None:
#         cs_low = 1e-40
#         cs_high = 1e-30
#     plt.xlim(mass_low,mass_high)
#     plt.ylim(cs_low,cs_high)
#     ax.tick_params('x', top=True, labeltop=False)
#     ax.tick_params('y', right=True, labelright=False)
#     plt.xlabel('$m_\chi$ [MeV]',fontsize=small)
#     plt.ylabel('$\overline{\sigma}_e$ [cm$^2$]',fontsize=small)

#     if fractional:
#         vmin = 0
        
#         max_amp = np.nanmax(Amplitudes) + np.mean(Amplitudes)
#         min_amp = np.nanmin(Amplitudes)
        
#         # levs = np.round(levs,3)
#         # if max_amp < 2:
#         #     vmax = 2
#         # else:
#         #     vmax = 10
#         vmax = 2
#         levs = np.arange(0,10,0.1)
#         norm =colors.Normalize(vmin=vmin,vmax=vmax)
#         extend = 'max'
        
        

#     else:
#         Amplitudes[Amplitudes == np.inf] = np.nan
#         Amplitudes[Amplitudes == -np.inf] = np.nan
#         max_amp = np.nanmax(Amplitudes)
        
#         min_amp = np.nanmin(Amplitudes)
#         low = -10 #find_exp(min_amp) 
#         high = find_exp(max_amp) + 2
#         if high <= 1:
#             high = 3

        
#         levs = np.arange(low,high) 
#         levs = np.power(10.,levs)
#         vminexp = -8
#         vmaxexp = 6
#         vmin = np.power(10.,vminexp)
#         vmax = np.power(10.,vmaxexp)
#         vmaxminus1 = np.power(10.,vmaxexp-1)

#         # levs = np.arange(vminexp,vmaxexp) 
#         # levs = np.power(10.,levs)
        

#         # norm =colors.LogNorm(vmin=vmin,vmax=vmax)
#         norm =colors.LogNorm(vmin=vmin,vmax=vmax)
#         extend = 'both'
#     cmap = 'Spectral_r'#diverging

#     #actual plotting
#     background_filling = np.ones_like(Amplitudes)*1e-10
#     BF = plt.contourf(Masses,CrossSections,background_filling,norm=norm,cmap=cmap,extend=extend)
#     CT1 =  plt.contourf(Masses,CrossSections,Amplitudes,levs,norm=norm,cmap=cmap,extend=extend)
#     # CT1.cmap.set_under(camp(norm(vmin)))


#     # if fractional:
#     #     if max_amp < 2:
#     #         CT1.cmap.set_over(camp(norm(max_amp)))
#     #     else:
#     #         CT1.cmap.set_over(camp(norm(vmax)))
#     # else:
#     #     CT1.cmap.set_over(camp(norm(vmax)))


#     #plotting for contour labels
#     if fractional:
#         levs = np.arange(0,2.1,0.1)
#         AmpContours = plt.contour(Masses,CrossSections,Amplitudes,levs,cmap=plt.get_cmap('binary'),alpha = 0)
#         # if FDMn == 0:
#         #     index = 14
#         # elif FDMn == 2:
#         #     index = 8
#         # zeropos = AmpContours.allsegs[index][0]
#         # plt.plot(zeropos[:, 0], zeropos[:, 1], color='white',lw=3,ls=':')


#         fmt = ticker.LogFormatterMathtext()
#         fmt.create_dummy_axis()
#         deflevs = AmpContours.levels
#         maxlev =  np.max(deflevs)
#         clevsd = [0,0.5,1,1.5]
#         final_levs = []
#         for c in clevsd:
#             if c < maxlev:
#                 final_levs.append(c)


#         # clevs = np.array(clevs)
#         # oddlevs = clevs[::2]
#         # evenlevs = clevs[1::2]
#         # if 1e0 in oddlevs:

#         # # clevs = [1.e-6,1.e-4,1.e-02, 1.e+00, 1.e+02, 1.e+04,1.e+06]
#         #     newlevs = oddlevs[oddlevs> 1e-6]
#         # else:
#         #     newlevs = evenlevs[evenlevs> 1e-6]
            
#         plt.clabel(AmpContours,deflevs, fmt=fmt)

    

#     if plotConstraints:
#         import sys
#         # sys.path.append('../../../limits/other_experiments/')
#         sys.path.append('../limits/')
#         from Constraints import plot_constraints
        
#         from modulation_contour import plot_constraints
#         x,y = plot_constraints('All',FDMn)
#         plt.plot(x,y,color='black',lw=3)
#         x,y = plot_constraints('Solar',FDMn)
#         plt.plot(x,y,color='black',lw=3,ls='--')

#     if FDMn == 0:
#         fdm_str = '$F_{\mathrm{DM}}= 1$'
#     else:
#         fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'

#     plt.text(0.95, 0.93, material,
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes,c='Black',fontsize=medium)
#     plt.text(0.02, 0.19, location,
#     horizontalalignment='left',
#     verticalalignment='center',
#     transform = ax.transAxes,c='Black',fontsize=small)
#     plt.text(0.02, 0.12, fdm_str,
#     horizontalalignment='left',
#     verticalalignment='center', 
#     transform = ax.transAxes,c='Black',fontsize=small)

#     e_bin_str = '$e^-$ bin = '+f'{ne}'
#     plt.text(0.02, 0.05, e_bin_str,
#     horizontalalignment='left',
#     verticalalignment='center', 
#     transform = ax.transAxes,c='Black',fontsize=small)



#     if fractional:
#         plt.title(f'Fractional Modulation Amplitude',fontsize=medium,y=1.03)

#     else:
#         plt.title(f'Modulation Amplitude [Events/g/day]',fontsize=medium,y=1.03)



    
#     if shadeMFP:
#         import sys
#         from tqdm.autonotebook import tqdm
#         sys.path.append('../DaMaSCUS/')
#         from MeanFreePath import Earth_Density_Layer_NU


#         mX_grid_mfp = np.geomspace(mass_low,mass_high,100)
#         print(mass_low,mass_high)
#         sigmaE_grid_mpf = np.geomspace(cs_low,cs_high,100)

#         #np.arange(0.1,1500,0.1)
#         EDLNU = Earth_Density_Layer_NU()
#         r_test = 0.8*EDLNU.EarthRadius #choose mantle
#         vMax = 300 * EDLNU.km / EDLNU.sec

#         MFP = []
#         for s in tqdm(range(len(sigmaE_grid_mpf))):
#             MFP_small = []
#             for m in range(len(mX_grid_mfp)):
#                 mX = mX_grid_mfp[m]*1e-3 #GeV
#                 sigmaP= sigmaE_grid_mpf[s] * (EDLNU.muXElem(mX,EDLNU.mProton) / EDLNU.muXElem(mX,EDLNU.mElectron))**2

#                 mfp = EDLNU.Mean_Free_Path(r_test,mX,sigmaP,vMax,FDMn,doScreen=True)
#                 MFP_small.append(mfp)
#             MFP_small = np.array(MFP_small)
#             MFP.append(MFP_small)
#         MFP = np.array(MFP)

#         # shade = np.zeros_like(MFP,dtype=bool)
#         # shade[MFP < 1] = True
#         # shade[MFP>=1] = False
#         X,Y = np.meshgrid(mX_grid_mfp,sigmaE_grid_mpf)
#         shade = np.ma.masked_where(MFP > 1, MFP)

#         plt.pcolor(X, Y, shade,hatch='/',alpha=0)

        

#     if fractional:
#         max_vis_amp = np.round(np.nanmax(Amplitudes),2)
#         if np.nanmax(Amplitudes) > 1:
#             cbar = fig.colorbar(CT1,extend=extend,norm=norm)
#             # cbar.ax.set_yticklabels(['0', '$1$', f'$>2$'])
#             cbar.ax.set_ylim(0,10)
#         else:
#             cbar = fig.colorbar(CT1,extend=extend,ticks=[0,max_vis_amp],norm=norm)
#             cbar.ax.set_yticklabels(['0',f'$>${max_vis_amp}'])
#             cbar.ax.set_ylim(0,max_vis_amp)
#         # cbar = fig.colorbar(CT1,extend=extend,norm=norm)


#     else:
#         cbar = fig.colorbar(CT1,extend=extend,norm=norm)
#         cbar.ax.set_yticks([vmin,1,vmaxminus1])
#         cbar.ax.set_yticklabels(['$< 10^{-8}$', '$1$', '$ > 10^6$'])
#         cbar.ax.set_ylim(vmin,vmaxminus1)
        

#         # cbar = fig.colorbar(CT1,extend='both')
#         # # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])

#     if FDMn == 0:
#         xy = (50,2e-41)
#         xysolar = (0.65,1e-37)
#     else: 
#         xy = (50,2e-36)
#         xysolar = (0.65,3e-34)
    
#     if plotConstraints:
#         plt.annotate('Current Constraints',xy,fontsize=small)
#         plt.annotate('Solar Bounds',xysolar,fontsize=smallest)
#     if savefig:
#         if fractional:
#             frac_str = 'fractional_'
#         else:
#             frac_str = ''
#         savedir = f'/Users/ansh/Local/SENSEI/paper_writing/ModulationTheory/figures/{material}/'

#         plt.savefig(f'{savedir}/{frac_str}Mod_Amplitude_{location}_fdm{FDMn}_mat{material}_{ne}ebin.png')
#     plt.show()
#     plt.close()
#     return

            
    

    
def modify_colormap(cmap_name,divisor=2, white_at_bottom=True):
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



                          

# def plotSignificanceContour(material,FDMn,location,exposure,background_rate,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs =None,plotConstraints=True,savefig=False,ne=1,xlow=None,ylow=None,xhigh=None,yhigh=None):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#     from matplotlib import colors
#     from matplotlib import cm, ticker

#     # plotting specifications
#     from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                 AutoMinorLocator)
#     from matplotlib.offsetbox import AnchoredText
#     #Options

#     large = 48
#     small = 36
#     medium = 40
#     smallest=16
#     smaller = 30
#     params = {'text.usetex' : True,
#         'font.size' : medium,
#             'font.family' : 'cmr10',
#             'figure.autolayout': True
#         }
#     plt.rcParams.update(params)
#     plt.rcParams['axes.unicode_minus']=False
#     plt.rcParams['axes.labelsize']=32
#     plt.rcParams['figure.figsize']=(16,12)
#     plt.rcParams['axes.formatter.use_mathtext']=True
#     if masses is None and sigmaEs is None:
#         getAll = True
#     else:
#         getAll = False
#     Masses,CrossSections,FractionalAmplitudes = getContourData(material,FDMn,location,fractional=True,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne)
#     Masses,CrossSections,Amplitudes = getContourData(material,FDMn,location,fractional=False,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,returnaverage=True)
#     FractionalAmplitudes[np.isnan(FractionalAmplitudes)] = 0

#     Significance = (FractionalAmplitudes*Amplitudes)*exposure / np.sqrt((Amplitudes + background_rate)*exposure)

#     Significance[np.isnan(Significance)] = 0

#     # Significance[Significance < 0.3] = 0

#     # Significance[Significance > 5] = 5.5

#     if masses is not None and sigmaEs is not None:
#         Significance = Significance.T

#     fig = plt.figure()
#     ax = plt.gca()
#     plt.xscale('log')
#     ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
#     plt.yscale('log')

#     if xlow is not None:
#         mass_low = xlow
#     else:
#         mass_low = np.min(Masses)
#     if xhigh is not None:
#         mass_high = xhigh
#     else:
#         mass_high = np.max(Masses)

        
#     if ylow is not None:
#         cs_low = ylow
#     else:
#         cs_low = np.min(CrossSections)
#         if FDMn == 2:
#             cs_low = 1e-40

#     if yhigh is not None:
#         cs_high = yhigh
#     else:
#         cs_high = np.max(CrossSections)
#         if FDMn == 2:
#             cs_high =  1e-30



    
#     plt.xlim(mass_low,mass_high)
#     plt.ylim(cs_low,cs_high)
#     ax.tick_params('x', top=True, labeltop=False)
#     ax.tick_params('y', right=True, labelright=False)
#     plt.xlabel('$m_\chi$ [MeV]',fontsize=small)
#     plt.ylabel('$\overline{\sigma}_e$ [cm$^2$]',fontsize=small)

 
        

#     vmin = 0
#     vmax = 5
#     max_sig = np.max(Significance)
#     min_sig = np.min(Significance)
#     # if max_sig > 2:
#     #     levs = np.arange(0,int(max_sig),1)
#     # else:
#     levs = np.arange(0,vmax,0.5)
#     # levs = np.linspace(0,vmax,50)

#     # norm =colors.LogNorm(vmin=vmin,vmax=vmax)
#     norm =colors.Normalize(vmin=vmin,vmax=vmax)
#     extend = 'max'
#     cmap = 'Spectral_r'#diverging

#     # Example usage:
#     original_cmap = cmap
#     cmap = modify_colormap(original_cmap,divisor=10)
#     # cmap = 'Reds'#single color
#     # cmap = 'afmhot_r'#sequential
#     # cmap = 'RdYlBu_r'#diverging


#     # upper = mpl.cm.Reds(np.arange(256))
#     # lower = np.ones((int(256/4),4))
#     # for i in range(3):
#     #     lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])

#     # cmap = np.vstack(( lower, upper ))
#     # cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


#     #actual plotting

#     CT1 =  plt.contourf(Masses,CrossSections,Significance,levs,norm=norm,cmap=cmap,extend=extend)
#     # CT1.cmap.set_under(camp(norm(vmin)))


#     # if fractional:
#     #     if max_amp < 2:
#     #         CT1.cmap.set_over(camp(norm(max_amp)))
#     #     else:
#     #         CT1.cmap.set_over(camp(norm(vmax)))
#     # else:
#     #     CT1.cmap.set_over(camp(norm(vmax)))


#     #plotting for contour labels
#     # CTFF = plt.contour(Masses,CrossSections,Amplitudes,levs,cmap=plt.get_cmap('binary'),linewidths=0)

#     if plotConstraints:
#         import sys
#         # sys.path.append('../../../limits/other_experiments/')
#         sys.path.append('../limits/')
#         from Constraints import plot_constraints
#         x,y = plot_constraints('All',FDMn)
#         plt.plot(x,y,color='black',lw=3)
#         x,y = plot_constraints('Solar',FDMn)
#         plt.plot(x,y,color='black',lw=3,ls='--')

#     if FDMn == 0:
#         fdm_str = '$F_{\mathrm{DM}}= 1$'
#     else:
#         fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'

#     e_bin_str = f'{ne}' + '$e^-$ bin'
#     plt.text(0.95, 0.93, material,
#      horizontalalignment='center',
#      verticalalignment='center',
#      transform = ax.transAxes,c='Black',fontsize=medium)
#     plt.text(0.02, 0.19, location,
#     horizontalalignment='left',
#     verticalalignment='center',
#     transform = ax.transAxes,c='Black',fontsize=small)
#     plt.text(0.02, 0.12, fdm_str,
#     horizontalalignment='left',
#     verticalalignment='center', 
#     transform = ax.transAxes,c='Black',fontsize=small)
#     plt.text(0.02, 0.05, e_bin_str,
#     horizontalalignment='left',
#     verticalalignment='center', 
#     transform = ax.transAxes,c='Black',fontsize=small)

#     #exposure is in gram days but can change the title if it is larger
#     if exposure >= 1e3 and exposure < 1e6:
#         mass_unit_str = 'kg'
#         exposure /= 1e3
#     elif exposure < 1e3:
#         mass_unit_str = 'g'

#     elif exposure >= 1e6 and material != 'Si':
#         mass_unit_str = 'tonne'
#         exposure /= 1e6
    


    


#     plt.title(f'Exposure = {exposure} {mass_unit_str}-days',fontsize=medium,y=1.03)


   


  
#     cbar = fig.colorbar(CT1,extend=extend,norm=norm)
#     cbar.ax.set_ylim(0,vmax)
#     # cbar.ax.set_yticks([0,1,2,3,4,5])

#     # cbar.ax.set_yticklabels(['0', '1','2','3','4','$\ge 5$'])
#         # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
#         # cbar.ax.set_ylim(1e-6,1e6)
        

#         # cbar = fig.colorbar(CT1,extend='both')
#         # # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])




#     if FDMn == 0:
#         xy = (50,2e-41)
#         xysolar = (0.65,1e-37)
#     else: 
#         xy = (50,2e-36)
#         xysolar = (0.65,3e-34)

    
#     if plotConstraints:
#         plt.annotate('Current Constraints',xy,fontsize=small)
#         plt.annotate('Solar Bounds',xysolar,fontsize=smallest)
#     if savefig:
#         savedir = f'/Users/ansh/Local/SENSEI/paper_writing/ModulationTheory/figures/{material}/'
#         plt.savefig(f'{savedir}/Mod_Sensitivity_{location}_fdm{FDMn}_mat{material}_exp{exposure}{mass_unit_str}days_{ne}ebin.jpg')
#     else:
#         plt.show()
#     plt.close()
#     return


def plotMaterialSignifianceFigure(fdm,material='Si',plotConstraints=True,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs=None,ne=1,shadeMFP=True,savefig=False,standardizeGrid = False,useQCDark=True,showProjection=False):
    from tqdm.autonotebook import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors
    from matplotlib import cm, ticker

    # plotting specifications
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options

    large = 48
    small = 36
    medium = 40
    smaller = 30
    smallest=16
    params = {'text.usetex' : True,
        'font.size' : medium,
            'font.family' : 'cmr10',
            'figure.autolayout': True
        }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=32
    plt.rcParams['figure.figsize']=(26,26)
    plt.rcParams['axes.formatter.use_mathtext']=True
    nrows = 3
    ncols = 2

    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,layout='constrained')
    ebinstr = f'{ne}' + '$e^-$ bin'
    fdmstr = '$F_{\mathrm{DM}} \propto q^{-2}$' if fdm == 2 else '$F_{\mathrm{DM}} = 1$'
    matdit = {
        'Si': 'Silicon',
        "Xe": 'Xenon',
        'Ar': "Argon"
    }
    fig.suptitle(f"{matdit[material]} Sensitivity {fdmstr}",fontsize=large)


    temp_amps = []

    masses_list = []
    cs_list = []
    famp_list = []
    amp_list = []
    if masses is None and sigmaEs is None:
        getAll = True
    else:
        getAll = False


       

    for loc in ["SNOLAB","SUPL"]:
        mini_mass_list = []
        mini_famp_list = []
        mini_cs_list = []
        mini_amp_list = []
        # for fdm in [0,2]:
        Masses,CrossSections,FractionalAmplitudes = getContourData(material,fdm,loc,fractional=True,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,useQCDark=useQCDark)
        Masses,CrossSections,Amplitudes = getContourData(material,fdm,loc,fractional=False,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,returnaverage=True,useQCDark=useQCDark)
        FractionalAmplitudes[np.isnan(FractionalAmplitudes)] = 0
        # mini_mass_list.append(Masses)
        # mini_cs_list.append(CrossSections)
        # mini_famp_list.append(FractionalAmplitudes)
        # mini_amp_list.append(Amplitudes)
        masses_list.append(Masses)
        cs_list.append(CrossSections)
        amp_list.append(Amplitudes)
        famp_list.append(FractionalAmplitudes)

        


    exposure_dict = {
        'Si': np.array([1 * nu.kg * nu.day,30 * nu.kg * nu.day,30 * nu.kg * nu.year]), #kg day, kg month, 30 kg year
        'Xe': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day ,1 * nu.tonne * nu.year]),#tonne day, tonne month, 1 tonne year
        'Ar': np.array([1 * nu.tonne * nu.day,30 * nu.tonne * nu.day, 17.4 * nu.tonne * nu.year])#tonne day, tonne month, ~17.4 tonne year
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
               first_index = 0 #SNOLAB
               loc = "SNOLAB"
            else:
                first_index = 1 #SUPL
                loc = "SUPL"
            # if j == 0 or j == 1:
            #     second_index = 0 #FDM 1
            #     fdm = 0
            # else:
            #     second_index = 1 #FDM q2
            #     fdm = 2
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
            current_ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
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
                        ylow=1e-37
                    elif ne == 2:
                        xlow= 10
                        xhigh = mass_high
                        yhigh=1e-32
                        ylow=1e-36

                    elif ne == 3 or ne == 4:
                        xlow = 10
                        xhigh = mass_high
                        yhigh = 1e-31
                        ylow = 1e-36
                   
                    

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
                        yhigh=1e-31
                        ylow=1e-36
                    elif ne == 4:
                        xlow= 10
                        xhigh = mass_high
                        yhigh = 1e-29
                        ylow = 1e-36
                

         



                # if i== 0:
                #     yhigh = 1e-36
                #     ylow = 1e-40


                # elif i==1:
                #     yhigh = 1e-36
                #     ylow = 1e-41

                # elif i==2:
                #     yhigh=1e-36
                #     ylow=1e-42
                # if standardizeGrid:
                    


            # elif material == 'Xe' and fdm == 2 and ne == 1:
                
                
                # if i ==0 :
                #     yhigh = 1e-33
                #     ylow = 1e-36


                # elif i == 1:
                #     yhigh = 1e-33
                #     ylow = 1e-36

                # elif i == 2:
                #     yhigh=1e-34
                #     ylow=1e-37

                # if standardizeGrid:
                #     yhigh=1e-34
                #     ylow=1e-37

            
            # elif material == 'Xe' and fdm == 2 and ne == 2:
                
                
            #     # if i== 0:
            #     #     yhigh = 1e-36
            #     #     ylow = 1e-40


            #     # elif i==1:
            #     #     yhigh = 1e-36
            #     #     ylow = 1e-41

            #     # elif i==2:
            #     #     yhigh=1e-36
            #     #     ylow=1e-42
            #     # if standardizeGrid:
                




            
                
            

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


            #actual plotting
            # if fractional:
            #     background_filling = np.zeros_like(Amplitudes)
            #     BF = current_ax.contourf(Masses,CrossSections,background_filling,norm=norm,cmap=cmap,extend=extend)
            # elif not plotSignificance:
            # background_filling = np.ones_like(Amplitudes)*1e-10
            # BF = current_ax.contourf(Masses,CrossSections,background_filling,norm=norm,cmap=cmap,extend=extend)

            CT1 =  current_ax.contourf(Masses,CrossSections,Amplitudes,levs,norm=norm,cmap=cmap,extend=extend)
            
            # CT1.cmap.set_under(camp(norm(vmin)))


            # if fractional:
            #     if max_amp < 2:
            #         CT1.cmap.set_over(camp(norm(max_amp)))
            #     else:
            #         CT1.cmap.set_over(camp(norm(vmax)))
            # else:
            #     CT1.cmap.set_over(camp(norm(vmax)))


            #plotting for contour labels
            # CTFF = plt.contour(Masses,CrossSections,Amplitudes,levs,cmap=plt.get_cmap('binary'),linewidths=0)

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


                





            if fdm == 0:
                fdm_str = '$F_{\mathrm{DM}}= 1$'
            else:
                fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'

    #         bbox=dict(
    #     boxstyle="round",  # Shape of the box
    #     facecolor="wheat",  # Background color
    #     edgecolor="black",  # Border color
    #     linewidth=1,  # Border width
    #     alpha=0.5,  # Transparency
    #     pad=0.5,  # Padding between text and border
    # )
            import matplotlib.patches as patches
            # xw = 0.4 if fdm == 2 else 0.25
            rect = patches.Rectangle((0.01, 0.02), 0.25, 0.155, linewidth=1, edgecolor='black', facecolor='white',transform = current_ax.transAxes,zorder = 2)
            current_ax.add_patch(rect)
            current_ax.text(0.95, 0.93, material,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=medium,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))
            current_ax.text(0.14, 0.05, loc,
            horizontalalignment='center',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))
            # current_ax.text(0.02, 0.16, fdm_str,
            # horizontalalignment='left',
            # verticalalignment='center', 
            # transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            e_bin_str = f'{ne}' + '$e^-$ bin'
            current_ax.text(0.14, 0.13, e_bin_str,
            horizontalalignment='center',
            verticalalignment='center', 
            transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            # 

            # print(exp_str,exposure)
            # print(exp_str >=1e3)
            time_unit = time_unit_strs[i]
            exp_str = exposures[i] / time_units[material][i]

            if material=='Si':
                mass_unit_str = 'kg'
            else:
                mass_unit_str = 'tonne'


            

        
            exp_str = int(exp_str)
            exposure_str = f'{exp_str} {mass_unit_str}-{time_unit}'
            # if material == 'Ar' and i == 2:
            #     exposure_str = f'17.4 tonne-years'

            current_ax.text(0.98, 0.05, exposure_str,
            horizontalalignment='right',
            verticalalignment='center',
            transform = current_ax.transAxes,c='Black',fontsize=small,bbox=dict(boxstyle='square',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))



           

            # current_ax.set_title(f'Modulation Significance Curves',fontsize=medium,y=1.03)


            
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

                

            
                # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
                # cbar.ax.set_ylim(1e-6,1e6)
                

                # cbar = fig.colorbar(CT1,extend='both')
                # # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])

            
        


    cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
    # if fractional:
    #     # max_vis_amp = np.round(np.max(temp_amps),2)
    #     # print(max_vis_amp)
    #     # if np.max(temp_amp) > 1:
    #     cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)
    #         # cbar = current_ax.cax.colorbar(CT1,)
    #         # cbar.ax.set_yticklabels(['0', '$1$', f'$>2$'])
    #         # cbar.ax.set_ylim(0,2)
    #         # cbar = current_ax.cax.colorbar(CT1,extend=extend,ticks=[0,max_vis_amp],norm=norm)
          
    #     # cbar = fig.colorbar(CT1,extend=extend,norm=norm)


    # else:
        # cbar = current_ax.cax.colorbar(CT1,ticks=[1e-6,1e0,1e6],extend=extend,norm=norm)
    cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)

        # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
        # cbar.ax.set_ylim(1e-6,1e6)
    cbar.ax.tick_params(labelsize=large)
    if savefig:
        mat_str_dict = {
            'Si': 'Silicon',
            'Xe': 'Xenon',
            'Ar': 'Argon',
        }
        matstr = mat_str_dict[material]
        savedir = f'figures/{matstr}'
        plt.savefig(f'{savedir}/Mod_Sensitivity_{material}_CombinedFig_{ne}ebin_fdm{fdm}.jpg')

    # plt.tight_layout()
    plt.show()
    plt.close()
    return 



def plotModulationFigure(fdm,fractional=False,plotConstraints=True,useVerne=True,fromFile=True,verbose=False,masses=None,sigmaEs=None,ne=1,shadeMFP=True,savefig=False,kgday=True,useQCDark=True,logfractional=True):
    from tqdm.autonotebook import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors
    from matplotlib import cm, ticker

    # plotting specifications
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options

    large = 48
    small = 36
    medium = 40
    smaller = 30
    smallest=16
    params = {'text.usetex' : True,
        'font.size' : medium,
            'font.family' : 'cmr10',
            'figure.autolayout': True
        }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=32
    plt.rcParams['figure.figsize']=(26,26)
    plt.rcParams['axes.formatter.use_mathtext']=True
    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,layout='constrained')
    fracstr = "Fractional" if fractional else ""
    fdmstr = '$F_{\mathrm{DM}} \propto q^{-2}$' if fdm == 2 else '$F_{\mathrm{DM}} = 1$'
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
            Masses,CrossSections,Amplitudes = getContourData(mat,fdm,loc,fractional=fractional,useVerne=useVerne,fromFile=fromFile,verbose=verbose,getAll=getAll,masses=masses,sigmaEs=sigmaEs,ne=ne,unitize=unitize,useQCDark=useQCDark)
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
            current_ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            current_ax.set_yscale('log')
            if masses is not None:
                mass_low = np.min(masses)
                mass_high = np.max(masses)
            else:
                mass_low = np.min(Masses)#0.6
                mass_high = np.max(Masses) #1000



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

                
                # levs = np.arange(low,high) 
                # levs = np.power(10.,levs)
                
                vminexp = -6
                vmaxexp = 6
                vmax = np.power(10.,vmaxexp)
                vmin = np.power(10.,vminexp)
                vmaxminus1 =  np.power(10,vmaxexp-1)
                levs = np.arange(vminexp,vmaxexp) 
                levs = np.power(10.,levs)
                divisor = len(levs)
    

                # norm =colors.LogNorm(vmin=vmin,vmax=vmax)
                norm =colors.LogNorm(vmin=vmin,vmax=vmax)
                extend = 'both'
            cmap = 'Spectral_r'#diverging
            cmap = 'Reds'#sequential
            original_cmap = cmap
            # if fractional:
            #     cmap = modify_colormap(original_cmap,divisor=divisor)


            #actual plotting
            # if fractional:
            #     background_filling = np.zeros_like(Amplitudes)
            #     BF = current_ax.contourf(Masses,CrossSections,background_filling,norm=norm,cmap=cmap,extend=extend)
            # elif not plotSignificance:
            background_filling = np.ones_like(Amplitudes)*1e-10
            # BF = current_ax.contourf(Masses,CrossSections,background_filling,norm=norm,cmap=cmap,extend=extend)

            CT1 =  current_ax.contourf(Masses,CrossSections,Amplitudes,levs,norm=norm,cmap=cmap,extend=extend)
            CT1.cmap.set_under(color='white')


            # if fractional:
            #     if max_amp < 2:
            #         CT1.cmap.set_over(camp(norm(max_amp)))
            #     else:
            #         CT1.cmap.set_over(camp(norm(vmax)))
            # else:
            #     CT1.cmap.set_over(camp(norm(vmax)))


            #plotting for contour labels
            # CTFF = plt.contour(Masses,CrossSections,Amplitudes,levs,cmap=plt.get_cmap('binary'),linewidths=0)

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
            # current_ax.text(0.02, 0.16, fdm_str,
            # horizontalalignment='left',
            # verticalalignment='center', 
            # transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))

            e_bin_str = f'{ne}' + '$e^-$ bin'
            current_ax.text(0.14, 0.13, e_bin_str,
            horizontalalignment='center',
            verticalalignment='center', 
            transform = current_ax.transAxes,c='Black',fontsize=small,zorder=3)#,bbox=dict(boxstyle='round',edgecolor="black",linewidth=1,alpha=1,pad=0.2,facecolor='white'))



            # if fractional:
            #     current_ax.set_title(f'Fractional Modulation Amplitude',fontsize=medium,y=1.03)

            # else:
            #     current_ax.set_title(f'Modulation Amplitude [Events/g/day]',fontsize=medium,y=1.03)
          

            
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

                # shade = np.zeros_like(MFP,dtype=bool)
                # shade[MFP < 1] = True
                # shade[MFP>=1] = False
                X,Y = np.meshgrid(mX_grid_mfp,sigmaE_grid_mpf)
                shade = np.ma.masked_where(MFP > 1, MFP)

                current_ax.pcolor(X, Y, shade,hatch='/',alpha=0)

                

            
                # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])
                # cbar.ax.set_ylim(1e-6,1e6)
                

                # cbar = fig.colorbar(CT1,extend='both')
                # # cbar.ax.set_yticklabels(['$< 10^{-6}$', '$1$', '$ > 10^6$'])

            if fdm == 0:
                xy = (40,2e-41)
                xysolar = (0.65,1e-37)
            else: 
                xy = (40,1e-36)
                xysolar = (2,9e-34)
            
            # if plotConstraints:
            #     current_ax.annotate('Current Constraints',xy,fontsize=smaller)
            #     current_ax.annotate('Solar Bounds',xysolar,fontsize=smallest)




    cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
    if fractional:
        # max_vis_amp = np.round(np.max(temp_amps),2)
        # print(max_vis_amp)
        # if np.max(temp_amp) > 1:
        cbar = fig.colorbar(CT1, cax=cax,extend=extend,norm=norm,**kw)
            # cbar = current_ax.cax.colorbar(CT1,)
            # cbar.ax.set_yticklabels(['0', '$1$', f'$>2$'])
            # cbar.ax.set_ylim(0,2)
            # cbar = current_ax.cax.colorbar(CT1,extend=extend,ticks=[0,max_vis_amp],norm=norm)
          
        # cbar = fig.colorbar(CT1,extend=extend,norm=norm)


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
        plt.savefig(f'{savedir}/{frac_str}Mod_Amplitude_CombinedFig_FDM{fdm}.png')

    # plt.tight_layout()
    
    plt.show()
    plt.close()
    return 


def find_exp(number) -> int:
    from math import log10, floor
    base10 = log10(number)
    return floor(base10)


def plotRateComparison(material,sigmaE,mX_list,fdm,plotVerne=True,savefig=False,savedir=None,verneOnly=False,damascusOnly=False,ne=1,useQCDark=True):
    import numpy as np
    # plotting specifications
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    params = {'text.usetex' : True,
            'font.size' : 40,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=40
    plt.rcParams['figure.figsize']=(16,14)

    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue
    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden
    fig = plt.figure()


    plt.xlabel('$\Theta$\N{degree sign}')
    plt.ylabel('Rate [events/g/day]')
    plt.yscale('log')
    plt.grid()
    plt.title(f'{material} {ne}$e^-$ Rate vs Isoangle',fontsize=40)
    plt.xlim(0,180)
    # plt.ylim(1e-8,1e3)
    #reversed('RdBu')

    southx = np.linspace(89.26129275462549,164.8486791095454,50)
    southy1 = np.ones_like(southx)*1e14
    southy2 = np.zeros_like(southx)
    northx = np.linspace(6.039066639146133,81.33821611144151,50)
    northy1 = np.ones_like(northx)*1e14
    northy2 = np.zeros_like(northx)



    plt.fill_between(southx,southy1,southy2,color='grey',alpha=0.3)

    plt.fill_between(northx,northy1,northy2,color='grey',alpha=0.3)




    ax = plt.gca()
    if plotVerne:
        ls = ['-','--']
        dummy_lines = []
        for b_idx in [0,1]:
            dummy_lines.append(ax.plot([],[], c="black", ls = ls[b_idx])[0])
        legend2 = plt.legend([dummy_lines[i] for i in [0,1]], ["DaMaSCUS", "Verne"], loc=2,prop={'size': 32})
        ax.add_artist(legend2)

        # plt.annotate('10 MeV',(60,1.5e0*24),color=colorslist[2],fontsize=30)
        # plt.annotate('1 MeV',(60,2.24*2e-2),color=colorslist[1],fontsize=30)

        # plt.annotate('0.6 MeV',(60,24*1e-4),color=colorslist[0],fontsize=30)


    colorlist = ['steelblue','crimson','forestgreen','rebeccapurple']

    maxv = 0
    minv = 1e20


    for i,mX in enumerate(mX_list):
        if type(sigmaE) == list:
            sE = sigmaE[i]
        else:
            sE = sigmaE


        if not verneOnly:
            isoangles,rates = get_modulated_rates(material,mX,sE,fdm,useVerne=False,ne=ne,useQCDark=useQCDark)
            isoangles,rates_high = get_modulated_rates(material,mX,sE,fdm,useVerne=False,calcError="High",ne=ne,useQCDark=useQCDark)
            isoangles,rates_low = get_modulated_rates(material,mX,sE,fdm,useVerne=False,calcError="Low",ne=ne,useQCDark=useQCDark)


            if maxv < np.max(rates_high):
                maxv = np.max(rates_high)

            if minv > np.min(rates_low[:25]):
                minv = np.min(rates_low[:25])


            rate_err = rates_high - rates

            plt.fill_between(isoangles,rates_low,rates_high,color=colorlist[i])
            x = isoangles[18]
            y = rates_high[18]*2.0
            plt.text(x,y,f'{mX} MeV',fontsize=30,color=colorlist[i],horizontalalignment='center',verticalalignment='center')

        if not damascusOnly:
            isoangles_v,rates_v = get_modulated_rates(material,mX,sE,fdm,useVerne=True,ne=ne,useQCDark=useQCDark)


            plt.plot(isoangles_v,rates_v,ls='--',color=colorlist[i])

            if maxv < np.max(rates_v):
                maxv = np.max(rates_v)
            if minv > np.min(rates_v[:25]):
                minv = np.min(rates_v[:25])







        # try:
        #     angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rates,rates_err=rate_err)
            
        # except:
        #     try:
        #         angle_grid,fit_vector,parameters,errors = fitted_rates(isoangles,rate,rates_err=None)
        #         fitFailed_w_errors = True
        #     except:
        #         fitFailed = True
        # plt.tight_layout()

    minexp = find_exp(minv) - 1
    if minexp < -6:
        minexp = -6
    maxexp = find_exp(maxv) + 3

    print(minv,maxv,minexp,maxexp)

    yticks = np.arange(minexp,maxexp+1) 
    yticks = np.power(10.,yticks)

    plt.ylim(yticks[0],yticks[-1])
    plt.yticks(yticks)
    plt.setp(plt.gca().get_yticklabels()[::2], visible=False)


    plt.text(0.28,0.05,'SNOLAB ($46$\N{degree sign}N)',fontsize=30,color='grey',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.72,0.05,'SUPL ($37$\N{degree sign}S)',fontsize=30,color='grey',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    if fdm == 2:
        fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'
    elif fdm == 0:
        fdm_str = '$F_{\mathrm{DM}} = 1$'

    
    
    if type(sigmaE) == list:
        for i,sE in enumerate(sigmaE):
            sE_str = str(sE)
            sigmaE_str = '$\overline{\sigma}_e =$ ' + r'${} \times 10^{{{}}}$'.format(*sE_str.split('e')) + 'cm$^2$'
            ycoord = 0.86 - 0.07*i
            plt.text(0.99,ycoord,sigmaE_str,fontsize=32,color=colorlist[i],horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)

    else:
        sE_str = str(sE)
        
        sigmaE_str = '$\overline{\sigma}_e =$ ' + r'${} \times 10^{{{}}}$'.format(*sE_str.split('e')) + 'cm$^2$'
        plt.text(0.99,0.86,sigmaE_str,fontsize=32,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    plt.text(0.99,0.95,fdm_str,fontsize=32,color='black',horizontalalignment='right',verticalalignment='center',transform = ax.transAxes)
    


    # yticks = np.arange(-8,3) 
    # yticks = np.power(10.,yticks)
    # plt.yticks(yticks)
    # plt.setp(plt.gca().get_yticklabels()[::2], visible=False)

    plt.xticks(np.linspace(0,180,19)[::2])
    if savefig:
        if savedir is None:
             mat_str_dict = {
            'Si': 'Silicon',
            'Xe': 'Xenon',
            'Ar': 'Argon',
        }
        matstr = mat_str_dict[material]
        savedir = f'figures/{matstr}'
        file = f'{material}_Rates_Comparison_FDM{fdm}.pdf'
        savefile = savedir+file
        plt.savefig(savefile)

    plt.show()

    plt.close()




            
def plotRateComparisonSubplots(material,sigmaE_list,mX_list,fdm,plotVerne=True,savefig=False,savedir=None,verneOnly=False,damascusOnly=False,ne=1,showScatter=False,showFit=False,useQCDark=True,kgday=True):
    import numpy as np
    # plotting specifications
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    if kgday:
        mass_factor = nu.kg
    else:
        mass_factor = nu.g
    time_factor = nu.day


    small = 32
    large= 40
    medium = 36
    params = {'text.usetex' : True,
            'font.size' : small,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=small
    plt.rcParams['figure.figsize']=(16,32)

    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue
    golden = (1 + 5 ** 0.5) / 2
    goldenx = 15
    goldeny = goldenx / golden
    fig,axes = plt.subplots(len(mX_list),layout='constrained')
    fig.suptitle(f'{material} {ne}$e^-$ Rate vs Isoangle',fontsize=large)
    # colorlist = ['steelblue','crimson','forestgreen','rebeccapurple']
    for i in range(len(mX_list)):
        current_ax = axes[i]
        mX = mX_list[i]
        sigmaE = sigmaE_list[i]
        current_ax.set_xlabel('$\Theta$\N{degree sign}')
        kgstr = 'k' if kgday else ''
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




        current_ax.fill_between(southx,southy1,southy2,color='grey',alpha=0.3)

        current_ax.fill_between(northx,northy1,northy2,color='grey',alpha=0.3)



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
            rates *= mass_factor * time_factor
            rates_high *= mass_factor * time_factor
            rates_low *= mass_factor * time_factor
            rates_flat *= mass_factor * time_factor

            current_ax.plot(isoangles,rates_flat,color='green',label="Flat",lw=3)

            maxv = np.max(rates_high)*1.2
            minv = np.min(rates_low)
        

            rate_err = rates_high - rates
            if showScatter:
                current_ax.errorbar(isoangles,rates,yerr=rate_err,linestyle='')
                current_ax.scatter(isoangles,rates,label='Data')

                if showFit:
                    try:
                        angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates,rate_err)
                    except ValueError:
                        try:
                            angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates)
                        except ValueError:
                            angle_grid,fit_vector,parameters,error = fitted_rates(isoangles,rates,linear=True)
                            

                    fit = fit_vector[0]
   

                    current_ax.plot(angle_grid,fit,color='red',label="Fit",lw=3)

            else:
                
                current_ax.fill_between(isoangles,rates_low,rates_high)
            x = isoangles[25]
            y = rates_high[18]*1.2
            # current_ax.text(x,y,f'{mX} MeV',fontsize=30,color=colorlist[i],horizontalalignment='center',verticalalignment='center')

        if not damascusOnly:
            isoangles_v,rates_v = get_modulated_rates(material,mX,sigmaE,fdm,useVerne=True,ne=ne,useQCDark=useQCDark)
            rates_v *= mass_factor * time_factor
         
            current_ax.plot(isoangles_v,rates_v,ls='--',label="Verne")
            if np.max(rates_v) > maxv:
                maxv = np.max(rates_v)
            if np.min(rates_v) < minv:
                minv = np.min(rates_v)

        if showScatter:
            if (material == 'Ar' or material == 'Xe') and fdm == 2: 
                current_ax.legend(loc='upper left',prop={'size': small})
            else:
                current_ax.legend(loc='center right',prop={'size': small})


    

        # plt.setp(current_ax.get_yticklabels()[::2], visible=False)


        current_ax.text(0.28,0.05,'SNOLAB ($46$\N{degree sign}N)',fontsize=30,color='grey',horizontalalignment='center',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.72,0.05,'SUPL ($37$\N{degree sign}S)',fontsize=30,color='grey',horizontalalignment='center',verticalalignment='center',transform = current_ax.transAxes)
        if fdm == 2:
            fdm_str = '$F_{\mathrm{DM}} = \\alpha m_e / q^2$'
        elif fdm == 0:
            fdm_str = '$F_{\mathrm{DM}} = 1$'

        mX_str = '$m_\chi = $' + f'{mX} MeV'

        
        
       
        sE_str = str(sigmaE)
        sigmaE_str = '$\overline{\sigma}_e =$ ' + r'${} \times 10^{{{}}}$'.format(*sE_str.split('e')) + 'cm$^2$'
        current_ax.text(0.99,0.86,sigmaE_str,fontsize=32,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.99,0.95,fdm_str,fontsize=32,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        current_ax.text(0.99,0.77,mX_str,fontsize=32,color='black',horizontalalignment='right',verticalalignment='center',transform = current_ax.transAxes)
        

        minv*=0.9
        maxv*=1.1
        # yticks = np.arange(-8,3) 
        # yticks = np.power(10.,yticks)
        # plt.yticks(yticks)
        # plt.setp(plt.gca().get_yticklabels()[::2], visible=False)

        current_ax.set_xticks(np.linspace(0,180,19)[::2])
        current_ax.set_ylim(minv,maxv)
    if savefig:
        if savedir is None:
            savedir = f'figures/{material}/'
            file = f'{material}_Rates_Comparison_FDM{fdm}_subfigs.pdf'
            savefile = savedir+file
        plt.savefig(savefile)

    plt.show()

    plt.close()



def plotMeanFreePath(FDMn,plotConstraints=True):
    import numpy as np
    from tqdm.autonotebook import tqdm
    import sys
    from MeanFreePath import Earth_Density_Layer_NU
    # sigmaEs = np.arange(-40,-28,1)
    # sigmaEs = np.arange(-40,-26,(-26 + 40)/1000)
    # sigmaEs = 10**(sigmaEs)
    # # mX_array = np.concatenate((np.arange(0.2,0.8,0.025),np.array([0.9]),np.arange(1,5,0.05),np.arange(5,11,1),np.array([20,50,100,200,500,1000])))
    # mX_array_heavy = np.concatenate((np.arange(0.2,10,0.025),np.arange(10,1500,0.1)))

    mX_array = np.geomspace(0.1,1000,100)
    sigmaEs = np.geomspace(1e-42,1e-28,100)

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
    import matplotlib.cm as cm
    from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator    
    import matplotlib.colors as colors
    from matplotlib import cm, ticker

    import matplotlib

    # plotting specifications
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options

    smallest = 16
    smaller = 24
    medium = 32
    large = 40
    params = {'text.usetex' : True,
            'font.size' : medium,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=medium
    plt.rcParams['figure.figsize']=(5,5)
    plt.rcParams['axes.formatter.use_mathtext']=True



    cmap = plt.get_cmap("tab10") # default color cycle, call by using color=cmap(i) i=0 is blue
    #reversed('RdBu')


    color_re = 'black'



    #formatting
    # fig, ax = plt.subplots(figsize=(15,10))

    golden = (1 + 5 ** 0.5) / 2
    goldenx = 16
    goldeny = goldenx / golden
    plt.figure(figsize=(goldenx,goldeny))

    # plt.figure(figsize=(12,12))


    # plt.ylim(np.min(sigmaEs),np.max(sigmaEs))
    # plt.xlim(np.min(mX_array),np.max(mX_array))
    plt.xlim(0.6,1000)
    
    plt.xscale('log')
    plt.yscale('log')
    if FDMn == 0:
        medtitstr= 'Heavy'
        xy = (4,5e-35)
        plt.ylim(1e-42,1e-28)
    elif FDMn == 2:
        xy = (7,1e-34)
        plt.ylim(1e-38,1e-29)
        medtitstr = 'Light'
    plt.title(f'{medtitstr} Mediator Mean Free Path' +  ' [$R_{\\oplus}$]',fontsize=large,y=1.01)
    plt.xlabel('$m_\chi$ [MeV]')
    plt.ylabel('$\overline{\sigma}_e$ [cm$^2$]')
    plt.tight_layout()

    
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if FDMn == 0:
        plt.text(0.85,0.23,'MFP $= R_{\\bigoplus}$',fontsize=medium,color=color_re,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
    elif FDMn == 2:
        plt.text(0.40,0.46,'MFP $= R_{\\bigoplus}$',fontsize=medium,color=color_re,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)



    #plotting




    # X,Y = np.meshgrid(mX_array,sigmaEs)

    lower = np.floor(np.log10(MFP.min())-1)
    upper = np.ceil(np.log10(MFP.max())+1)
    num_steps = 100
    lev_exp = np.arange(lower,upper)
    levs = np.power(10,lev_exp)

    CT_MFP= plt.contourf(mX_array,sigmaEs,MFP,levs,norm=colors.LogNorm(),locator=plt.LogLocator())

    CTL_MFP = plt.contour(mX_array,sigmaEs,MFP,levs,cmap=plt.get_cmap('binary'),linewidths=0)

    CTI_MFP = plt.contour(mX_array,sigmaEs,MFP,levs,cmap=plt.get_cmap('binary'),alpha = 0)
    if FDMn == 0:
        index = 14
    elif FDMn == 2:
        index = 8
    zeropos = CTI_MFP.allsegs[index][0]
    plt.plot(zeropos[:, 0], zeropos[:, 1], color=color_re,lw=3,ls=':')


    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()

    clevs =  CTL_MFP.levels

    clevs = np.array(clevs)
    oddlevs = clevs[::2]
    evenlevs = clevs[1::2]
    if 1e0 in oddlevs:

    # clevs = [1.e-6,1.e-4,1.e-02, 1.e+00, 1.e+02, 1.e+04,1.e+06]
        newlevs = oddlevs[oddlevs> 1e-6]
    else:
        newlevs = evenlevs[evenlevs> 1e-6]
        
    plt.clabel(CTL_MFP,newlevs, fmt=fmt)

    # ax.set_xlim(np.min(mX_array),np.max(best_c_x))


    # ax.plot(mX_array,mfp_1y,color='black',lw=1)

    # ax.plot(light_limx,sensei_func(light_limx),color='blue',lw=5)


    # solar_light_bound = np.loadtxt('../../../limits/other_experiments/DM-e-FDMq2/SolarReflection_XE1T_S2_FDMq2_210810332.csv',delimiter=',')
    # solarbx = solar_light_bound[:,0]

    # solarby= solar_light_bound[:,1]

    # solarby = solarby[np.argsort(solarbx)]
    # solarbx = solarbx[np.argsort(solarbx)]

    # ax.plot(solarbx,solarby,color='grey',lw=3)
    # ax.fill_between(solarbx,solarby,1e-29,color='gray',alpha=0.3)


    # ax.fill_between(damic_limx,damic_limy,1e-29,color='gray',alpha=0.3)

    # plt.colorbar()
    # CS = ax.contour(X,Y,MFP_light)
    # ax.clabel(CS,inline=True)

    if plotConstraints:
        import sys
        sys.path.append('../limits/')
        from Constraints import plot_constraints

        x,y = plot_constraints('All',FDMn)
        plt.plot(x,y,color='black',lw=3)
        x,y = plot_constraints('Solar',FDMn)
        plt.plot(x,y,color='black',lw=3,ls='--')
    

    if FDMn == 0:
        xy = (50,3e-42)
        xysolar = (0.65,7e-39)
        title_str = "Heavy"

    elif FDMn == 2:
        xy = (8,3e-38)
        title_str = "Light"
        xysolar = (0.65,3e-35)

    plt.annotate('Current Constraints',xy,fontsize=medium)

    plt.annotate('Solar Bounds',xysolar,fontsize=smallest)



    plt.savefig(f'figures/Misc/{title_str}Mediator_MFP.pdf')
    plt.show()
    plt.close()



def plotLocationExposure(address1,address2,savefig=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    from scipy.optimize import curve_fit
    from astropy.coordinates import EarthLocation, SkyCoord,AltAz
    from astropy.time import Time
    import astropy.units as u
    # plotting specifications
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tck
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                AutoMinorLocator)
    from matplotlib.offsetbox import AnchoredText
    #Options
    small = 16
    large= 24
    medium = 20
    params = {'text.usetex' : True,
            'font.size' : small,
            'font.family' : 'cmr10',
            'figure.autolayout': True
            }
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.labelsize']=small
    plt.rcParams['figure.figsize']=(16,16)

    import matplotlib.cm as mplcm
    import matplotlib.colors as colors

    sidereal_day = 23.9344696
    th=np.arange(0,sidereal_day,0.001) #change to sidereal day, finer bin
    t=[]
    for i in th:
        t.append(Time('2025-1-1 00:00:00')+i*u.hour)

    wind=SkyCoord(l=90*u.deg,b=0*u.deg,frame="galactic")

    loc1=EarthLocation.of_address(address1)
    loc2=EarthLocation.of_address(address2)

    wimploc1 = wind.transform_to(AltAz(obstime=t,location=loc1))
    wimploc2 = wind.transform_to(AltAz(obstime=t,location=loc2))



    isoloc1=90-wimploc1.alt.deg
    isoloc2=90-wimploc2.alt.deg

    # fig, ax = plt.subplots(2)
    colorlist = ['steelblue','black']

    plt.figure(figsize=(15, 7),dpi=80)
    plt.subplot(1,2,1)
    plt.plot(th,isoloc1,label=address1,color=colorlist[0])

    plt.plot(th,isoloc2,label=address2,color=colorlist[1])

    plt.xlim(0,24)
    plt.ylim(-5,185)
    plt.legend()
    plt.ylabel("WIMP wind angle [degrees]")
    plt.xlabel("UTC time of day on January 1st 2025 [hours]")
    plt.grid()
    ax=plt.gca()
    ax.xaxis.set_ticks(np.arange(0, 24.01, 4))
    ax.yaxis.set_ticks(np.arange(0, 180.01, 15))
    plt.subplot(1,2,2)
    hb=np.linspace(0,180,int(180*4))
    plt.hist(isoloc1,label=address1,bins=hb,histtype=u'step',color=colorlist[0])
    plt.hist(isoloc2,label=address2,bins=hb,histtype=u'step',color=colorlist[1])
    plt.xlabel("WIMP wind isoangle")
    plt.ylabel("Exposure (arb. units)")
    plt.xticks(np.linspace(0,180,19)[::2])
    plt.legend()
   
    if savefig:
        plt.savefig(f'figures/Misc/IsoLoc.pdf')
    plt.show()
    plt.close()



    return