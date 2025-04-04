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

ATOMIC_WEIGHT = dict(
    Xe=131.293 *nu.amu,
    Ar=39.948*nu.amu,
    Ge=72.64 * nu.amu,
    Si=28.0855*nu.amu
)

"""Useful Constant Definitions"""
mP_eV = (nu.mp * nu.c0**2) #mass of proton in energy units
me_eV = (nu.me * nu.c0**2) #mass of electron in energy units

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

si_screening = {
    'eps0': 11.3  ,
    'qTF' : 4.13e3*nu.eV,
    'omegaP': 16.6*nu.eV,
    'alphaS': 1.563
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
rhoX = 0.3 * nu.GeV / nu.cm**3                               # In GeV/cm^3
crosssection = 1e-36 * nu.cm**2                              # In cm^2

"""Material Parameters"""

Sigapsize= 3.8 *nu.eV
Gegapsize = 3. * nu.eV


