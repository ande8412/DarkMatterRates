import numericalunits as nu
from .Constants import *
"""Form factor object, reads in data if available."""
class form_factor(object):
     '''
     Class containing our form factor object, including data 
     regarding input.
     
     To access crystal form factor, please use
          form_factor.ff
     For other information, please view listed data.

     Early versions of results did not store if the energies
     had been scissor corrected to band gap, and so backwards
     compatibility requires allowing this to be skipped.
     '''

     def __init__(self, filename,useQEDark=False):
          import h5py
          print(f"Using form factor calculated from file: {filename}")

          data = h5py.File(filename, 'r')
          self.lattice = data['run_settings/a'][...].copy()
          self.atom = data['run_settings'].attrs['atom']
          self.basis = data['run_settings'].attrs['basis']
          self.ecp = data['run_settings'].attrs['ecp']
          self.dft_rcut = float(data['run_settings/rcut'][...])
          self.dft_precision = float(data['run_settings/precision'][...])
          try:
               self.dark_Rvec = data['run_settings/Rvec'][...].copy()
          except:
               pass
          self.num_con = data['run_settings'].attrs['numcon']
          self.num_val = data['run_settings'].attrs['numval']
          self.dft_xc = data['run_settings'].attrs['xc']
          self.dft_density_fitting = data['run_settings'].attrs['df']
          self.kpts = data['run_settings/kpts'][...].copy()
          self.VCell = data['results'].attrs['VCell'] * nu.eV
          self.mCell = data['results'].attrs['mCell'] * nu.eV
          self.mCell/= nu.c0**2 #in mass units
          self.dq = data['results'].attrs['dq']*nu.alphaFS * me_eV
          self.dE = data['results'].attrs['dE'] * nu.eV
          
          
          self.ff = data['results/f2'][...].copy()
          
          if useQEDark:
               import os
               import numpy as np
               module_dir = os.path.dirname(__file__)
               fpath = '../QCDark/QEDark_f2_Si.npy'
               form_factor_file_filepath = os.path.join(module_dir,fpath)

               self.ff = np.load(form_factor_file_filepath)


               
          self.band_gap = data['results'].attrs['bandgap'] * nu.eV
          try:
               self.scissor_corrected = data['results'].attrs['scissor']
          except:
               pass
          self.convert_atom()
          if self.ecp == 'None':
               self.ecp = None
          data.close()

     def convert_atom(self):
          import numpy as np
          string = self.atom.replace(';', '\n')
          atoms = string.split('\n')
          self.atom = np.asarray([atom.split() for atom in atoms])

class form_factorQEDark(object):
    '''
     Class containing our form factor object, including data 
     regarding input.
     
     To access crystal form factor, please use
          form_factor.ff
     For other information, please view listed data.

     Early versions of results did not store if the energies
     had been scissor corrected to band gap, and so backwards
     compatibility requires allowing this to be skipped.
     '''
    def __init__(self,filename):
        import numpy as np
        nq = 900
        wk = 2/137
        nE = 500
        fcrys = np.transpose(np.resize(np.loadtxt(filename,skiprows=1),(nE,nq)))
        if 'Si' in filename:
            self.material = 'Si'
        elif 'Ge' in filename:
            self.material = 'Ge'

        # fcrys = {'Si': ,'Ge': np.transpose(np.resize(np.loadtxt('./QEDark/Ge_f2.txt',skiprows=1),(nE,nq)))}


        """
            materials = {name: [Mcell #eV, Eprefactor, Egap #eV, epsilon #eV, fcrys]}
            N.B. If you generate your own fcrys from QEdark, please remove the factor of "wk/4" below. 
        """
        materials = {'Si': [(2*28.0855) , 2.0, 1.2, 3.8,wk/4*fcrys],'Ge': [2*72.64, 1.8, 0.7, 3,wk/4*fcrys]}
        self.ff = fcrys*wk/4
        self.dq = .02*nu.alphaFS*me_eV
        self.dE = 0.1 * nu.eV
        self.mCell = ATOMIC_WEIGHT[self.material]#materials[self.material][0] *nu.amu #mass units
        
        self.band_gap = materials[self.material][2]*nu.eV
     #    self.Eprefactor = materials[self.material][1]
        