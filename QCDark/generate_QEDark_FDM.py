import numpy as np
import scipy as sp
import scipy.integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as clb
import matplotlib.cm as cm
import h5py
import matplotlib.ticker
import pdb
import dark_matter_rates as dmr
import os

"""Global Constants & Conversion"""
amu2eV = 9.315e8                                             # eV/u
lightSpeed = 299792.458                                     # In km/s
pi = np.pi
alpha = 1.0/137.03599908                                    # EM fine-structure constant at low E
mElectron = 5.1099894e5                                     # In eV
BohrInv2eV = alpha*mElectron                                # 1 Bohr^-1 to eV
Ryd2eV = 0.5*mElectron*alpha**2                             # In eV/Ryd
cm2sec = 1/lightSpeed*1e-5                                  # 1 cm in s
sec2yr = 1/(60.*60.*24*365.25)                              # 1 s in years
sec2day = 1/(60.*60.*24)                                    # 1 s in days

cmInv2eV = 50677.3093773                                    # 1 eV in cm^-1

"""Dark Matter Parameters"""
v0 = 238.0/lightSpeed                                       # In units of c
vEarth = 250.2/lightSpeed                                   # In units of c
vEscape = 544.0/lightSpeed                                  # In units of c
rhoX = 0.3e9                                                # In eV/cm^3
crosssection = 1e-36                                        # In cm^2

"""Thomas-Fermi Screening Parameters"""
eps0 = 11.3                                                 # Unitless
qTF = 4.13e3                                                # In eV
omegaP = 16.6                                               # In eV
alphaS = 1.563                                              # Unitless


ionization_model = 'R'
ionization_parameter = 'p100K.dat'
SI = dmr.form_factor('Si_final.hdf5')

mX_array = np.concatenate((np.arange(0.2,0.8,0.025),np.array([0.9]),np.arange(1,5,0.05),np.arange(5,11,1),np.array([20,50,100,200,500,1000,10000])))*1e6 # in eV

Si_QCDark = dmr.read_output('Si_final.hdf5') # for QCDark (not use here)
Si_QEDark = dmr.read_output('Si_final.hdf5') # for QEDark

Si_QEDark.ff = np.load('QEDark_f2_Si.npy')

# Heavy
resultsHeavyQEDarkScreen = dmr.calculate_rates(mX_array, Si_QEDark, ionization_model, ionization_parameter, FDM_exp = 0, max_num_electrons = 10, sigmae = crosssection, DoScreen = True, saveData = True, fileName = '../../../FDM/QEDark/FDM1_vesc544pt0-v0238pt0-vE250pt2-rhoX0pt3_nevents_funcp100_QEDarkScreen.dat')

resultsHeavyQEDarkScreen_print = resultsHeavyQEDarkScreen.T
resultsHeavyQEDarkScreen_print.shape

print("resultsHeavyQEDarkScreen")
print(resultsHeavyQEDarkScreen_print)

# Light
resultsLightQEDarkScreen = dmr.calculate_rates(mX_array, Si_QEDark, ionization_model, ionization_parameter, FDM_exp = 2, max_num_electrons = 10, sigmae = crosssection, DoScreen = True, saveData = True, fileName = '../../../FDM/QEDark/FDMq2_vesc544pt0-v0238pt0-vE250pt2-rhoX0pt3_nevents_funcp100_QEDarkScreen.dat')

resultsLightQEDarkScreen_print = resultsLightQEDarkScreen.T
resultsLightQEDarkScreen_print.shape

print("resultsLightQEDarkScreen")
print(resultsLightQEDarkScreen_print)
