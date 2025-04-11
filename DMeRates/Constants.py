import numericalunits as nu
import random
#-2 to 2 are default values.
nu.m =10 ** random.uniform(1,2) # meter --this scale should be fine
nu.s =10 ** random.uniform(5,7) # s -- relevant scale is days or years, so this is fine
nu.kg =10 ** random.uniform(10,12) # kg -- working with tiny masses, so setting the scale up
nu.C = 10 ** random.uniform(-2,2) # coulomb (not relevant)
nu.K = 10 ** random.uniform(-2,2) # kelvin (not relevant)

#comment this out if you want to debug and make sure units are correct, otherwise leave on to avoid numerical instability from picking random units
# nu.reset_units('SI')

nu.set_derived_units_and_constants()
#ATOMIC WEIGHTS



"""Useful Constant Definitions"""
mP_eV = (nu.mp * nu.c0**2) #mass of proton in energy units
me_eV = (nu.me * nu.c0**2) #mass of electron in energy units
ry = nu.me * nu.e ** 4 / (8 * nu.eps0 ** 2 * nu.hPlanck ** 2) #rydberg energy


"""Thomas-Fermi Screening Parameters"""
tf_screening = {
    'Si' : 
    {
    'eps0': 11.3  ,
    'qTF' : 4.13e3*nu.eV,
    'omegaP': 16.6*nu.eV,
    'alphaS': 1.563
    },

    'Ge':
    {
    'eps0': 14.0,
    'qTF' : 3.99e3*nu.eV,
    'omegaP': 15.2*nu.eV,
    'alphaS': 1.563
    }

}


"""Halo Model Parameters"""
q_Tsallis = 0.773
# v0_Tsallis = 267.2 #km/s
# vEsc_Tsallis = 560.8 #km/s
k_DPL = 2.0 #1.5 <= k <= 3.5 found to give best fit to N-body simulations. 
# p_MSW =  ?


"""Dark Matter Parameters"""
v0 = 238.0 * nu.km / nu.s                                    # In units of km/s
vEarth = 250.2 * nu.km / nu.s                                # In units of km/s
vEscape = 544.0 * nu.km / nu.s                               # In units of km/s
rhoX = 0.3 * nu.GeV / nu.c0**2 / nu.cm**3                               # In GeV/c^2/cm^3
crosssection = 1e-36 * nu.cm**2                              # In cm^2

"""Material Parameters"""

Sigapsize= 3.8 *nu.eV
Gegapsize = 3. * nu.eV

ATOMIC_WEIGHT = {
    'Xe': 131.293 *nu.amu,
    "Ar": 39.948*nu.amu,
    "Ge": 72.64 * nu.amu,
    "Si": 28.0855*nu.amu,
}


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
        '4s': 213.8 * nu.eV,
        '4p': 163.5 * nu.eV,
        '4d': 75.6 * nu.eV,
        '5s': 25.7 * nu.eV,
        '5p': 12.4 * nu.eV
        },
    # 'Ar': {
    #     '3s': 29.3,
    #     '3p12': 15.9,
    #     '3p32': 15.7,
    #     }
    'Ar': { #the darkside versions
        '3s': 34.76 * nu.eV,
        '3p12': 16.08 * nu.eV,
        '3p32': 16.08 * nu.eV,
        }
}
work_function = {
    'Xe': 13.8 * nu.eV,
    'Ar': 19.5 * nu.eV
}
skip_keys = {
    'Xe': ['3s','3p','3d'],
    'Ar': ['3p32'],
}