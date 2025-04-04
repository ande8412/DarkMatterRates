#You may need to change the path location to run this
import os

module_dir = os.path.dirname(__file__)
local_path = os.path.join(module_dir,'./')


# local_path = '/Users/ansh/Local/SENSEI/sensei_toy_limit/limits/'

def plot_constraints(material,fdm):
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator    

    if material=='Solar':
        if fdm == 0:
            constraints = np.loadtxt(local_path + 'solar_current_constraints_fdm1.csv',delimiter = ',')

        if fdm == 2:
            constraints = np.loadtxt(local_path +'solar_current_constraints_fdmq2.csv',delimiter = ',')
        return constraints[0],constraints[1]


    if material == 'Si':
        if fdm == 0:
            heavy_data = np.loadtxt(local_path + './DM-e-FDM1/snolab_heavy_limit_qcdark.csv',delimiter=',')
            heavy_limx = heavy_data[:,0]
            heavy_limy = heavy_data[:,1]



            #not official, so ignoring
            # damic_heavy_data = np.loadtxt(local_path + 'DM-e-FDM1/damic_2025_direct_qcdark.csv',delimiter=',')
            damic_heavy_data = np.loadtxt(local_path + './DM-e-FDM1/damic_2023_1.csv',delimiter=',')

            damic_limx = damic_heavy_data[:,0]
            damic_limy = damic_heavy_data[:,1]


            heavy_constraints_data = np.loadtxt(local_path + 'current_constraints_fdm1.csv',delimiter=',')
            cons_limx = heavy_constraints_data[:,0]
            cons_limy = heavy_constraints_data[:,1]

            damic_mod_data = np.loadtxt(local_path + 'DM-e-FDM1/damicM_modulation_heavy.csv',delimiter=',')
            damic_mod_limx = damic_mod_data[:,0]
            damic_mod_limy = damic_mod_data[:,1]

            indices_sort_heavy = np.argsort(heavy_limx)

            heavy_limx = heavy_limx[indices_sort_heavy]
            heavy_limy = heavy_limy[indices_sort_heavy]


            damic_func = Akima1DInterpolator(damic_limx,damic_limy)
            sensei_func = Akima1DInterpolator(heavy_limx,heavy_limy)
            damic_mod_func = Akima1DInterpolator(damic_mod_limx,damic_mod_limy)

            cons_func = Akima1DInterpolator(cons_limx,cons_limy)

            best_constraints = []
            best_c_x = []
            for mX in heavy_limx:
                minp = np.nanmin(np.array([damic_func(mX),sensei_func(mX),damic_mod_func(mX)]))
                best_constraints.append(minp)
                best_c_x.append(mX)



            best_c_x = np.array(best_c_x)
            best_constraints = np.array(best_constraints)
            
            return best_c_x,best_constraints



        elif fdm == 2:
            light_data = np.loadtxt(local_path + './DM-e-FDMq2/snolab_light_limit_qcdark.csv',delimiter=',')
            light_limx = light_data[:,0]
            light_limy = light_data[:,1]

            #not official, so ignoring
            # damic_light_data = np.loadtxt(local_path + 'DM-e-FDMq2/damic_2025_direct_qcdark.csv',delimiter=',')
            damic_light_data = np.loadtxt(local_path + 'DM-e-FDMq2/damic_2023_q2.csv',delimiter=',')

            damic_limx = damic_light_data[:,0]
            damic_limy = damic_light_data[:,1]


            damic_modulation_data = np.loadtxt(local_path + 'DM-e-FDMq2/damicM_modulation_light.csv',delimiter=',')

            damic_mod_limx = damic_modulation_data[:,0]
            damic_mod_limy = damic_modulation_data[:,1]

            damic_func = Akima1DInterpolator(damic_limx,damic_limy)
            damic_mod_func = Akima1DInterpolator(damic_mod_limx,damic_mod_limy)
            sensei_func = Akima1DInterpolator(light_limx,light_limy)

            best_constraints = []
            best_c_x = []
            for mX in light_limx:

                p1 = damic_func(mX)
                p2 = sensei_func(mX)
                p3 = damic_mod_func(mX)
                minp = np.nanmin(np.array([p1,p2,p3]))
                best_constraints.append(minp)
                best_c_x.append(mX)


            best_constraints = np.array(best_constraints)
            best_c_x  = np.array(best_c_x)

            return best_c_x,best_constraints
        
    if material == "Xe":
        if fdm == 0:
            xenon_heavy_constraints = np.loadtxt(local_path + 'DM-e-FDM1/xenon_heavy_constraints.csv',delimiter = ',')

            return xenon_heavy_constraints[:,0],xenon_heavy_constraints[:,1]
        
        if fdm == 2:
            xenon_light_constraints = np.loadtxt(local_path + 'DM-e-FDMq2/XENON10_2017FDMq2Data.csv',delimiter = ',')

            return xenon_light_constraints[:,0],xenon_light_constraints[:,1]
        
    if material == "Ar":
        if fdm == 0:
            argon_heavy_constraints = np.loadtxt(local_path + 'DM-e-FDM1/DarkSide2022FDM1.csv',delimiter = ',')

            return argon_heavy_constraints[:,0],argon_heavy_constraints[:,1]
        
        if fdm == 2:
            argon_light_constraints = np.loadtxt(local_path + 'DM-e-FDMq2/Darkside2022fdmq2.csv',delimiter = ',')

            return argon_light_constraints[:,0],argon_light_constraints[:,1]
        
    if material == 'All':
        if fdm == 0:
            constraints = np.loadtxt(local_path + 'current_constraints_fdm1.csv',delimiter = ',')

        if fdm == 2:
            constraints = np.loadtxt(local_path + 'current_constraints_fdmq2.csv',delimiter = ',')
        return constraints[0],constraints[1]




