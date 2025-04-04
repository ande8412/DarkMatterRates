

class Earth_Density_Layer_NU:
    def __init__(self):
        self.Elements = None
        self.GeV = 1.0
        self.MeV	 = 1.0E-3 * self.GeV
        self.eV	 = 1.0E-9 * self.GeV
        self.gram = 5.617977528089887E23 * self.GeV
        self.cm			 = 5.067E13 / self.GeV
        self.meter		 = 100 * self.cm
        self.km			 = 1000 * self.meter
        self.EarthRadius = 6371 *self.km
        self.mNucleon  = 0.932 * self.GeV
        self.mProton  = 0.938 * self.GeV
        self.m  = 0.932 * self.GeV
        self.sec = 299792458 * self.meter
        self.Bohr_Radius = 5.291772083e-11 * self.meter
        self.mElectron = 0.511 * self.MeV
        self.alpha= 1.0 / 137.035999139


    def get_layer(self,r): #inner core
        #r in natural units
        x = r / self.EarthRadius
        # print('converted radius',r/self.km)
        if r < 1221.5*self.km: #km
            # print('Inner Core')
            self.Core()
            self.density = 13.0885 - 8.8381*x**2
        elif r >= 1221.5*self.km and r < 3480*self.km: #outer core
            # print('Outer Core')
            self.Core()
            self.density = 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3
        elif r >= 3480*self.km and r < 3630*self.km: #Lower Mantle 1 
            # print('Lower Mantle 1')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 3630*self.km and r < 5600*self.km: #Lower Mantle 2
            # print('Lower Mantle 2')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3
        elif r >= 5600*self.km and r < 5701*self.km: #Lower Mantle 3
            # print('Lower Mantle 3')

            self.Mantle()
            self.density = 7.9565 - 6.47618*x + 5.5283*x**2 - 3.0807*x**3

        elif r >= 5701*self.km and r < 5771*self.km:#Transition Zone 1
            # print('Transition Zone 1')

            self.Mantle()
            self.density = 5.3197 - 1.4836*x
        elif r >= 5771*self.km and r < 5971*self.km:#Transition Zone 2
            # print('Transition Zone 2')
            self.Mantle()
            self.density = 11.2494 - 8.0298*x
        elif r >= 5971*self.km and r < 6151*self.km: #Transition Zone 3
            # print('Transition Zone 3')

            self.Mantle()
            self.density = 7.1089-3.8405*x
        elif r >= 6151*self.km and r < 6291*self.km: #LVZ
            # print('LVZ')

            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6291*self.km and r < 6346.6*self.km: #LID
            # print('LID')
            self.Mantle()
            self.density = 2.6910 + 0.6924*x
        elif r >= 6346.6*self.km and r < 6356*self.km: #crust 1 
            # print('Inner Crust')
            self.Mantle()
            self.density = 2.9
        elif r >= 6356*self.km and r < 6368*self.km: #crust 2
            # print('Outer Crust')
            self.Mantle()
            self.density = 2.6
        # elif r >= 6368 and r < 6371: #ocean
        #     self.Mantle()
        self.density*= self.gram * (self.cm)**(-3) #[GeV^4]
        return

        
        

    def Core(self):
        self.Elements = [
            [26, 56, 0.855],  # # Iron			Fe
            [14, 28, 0.06],	   ## Silicon		Si
            [28, 58, 0.052],   ## Nickel		Ni
            [16, 32, 0.019],   ## Sulfur		S
            [24, 52, 0.009],   ## Chromium		Cr
            [25, 55, 0.003],   ## Manganese    Mn
            [15, 31, 0.002],   ## Phosphorus	P
            [6, 12, 0.002],	   ## Carbon		C
            [1, 1, 0.0006]	   ## Hydrogen		H
        ]
        return

    def Mantle(self):
        self.Elements = [
            [8, 16, 0.440],		# Oxygen		O
		[12, 24, 0.228],	# Magnesium		Mg
		[14, 28, 0.21],		# Silicon		Si
		[26, 56, 0.0626],	# Iron			Fe
		[20, 40, 0.0253],	# Calcium		Ca
		[13, 27, 0.0235],	# Aluminium		Al
		[11, 23, 0.0027],	# Natrium		Na
		[24, 52, 0.0026],	# Chromium		Cr
		[28, 58, 0.002],	# Nickel		Ni
		[25, 55, 0.001],	# Manganese		Mn
		[16, 32, 0.0003],	# Sulfur		S
		[6, 12, 0.0001],	# Carbon		C
		[1, 1, 0.0001],		# Hydrogen		H
		[15, 31, 0.00009]	# Phosphorus	P
        ]
        return 
    
    def NucleusMass(self,N):
        return N*self.mNucleon


 
    


    def muXElem(self,mX,mElem):     
      return mX*mElem/(mX+mElem)



    def sigma_i(self,v,isotope_mass,z,sigmaP,mX,FDMn,doScreen=True):
        import numpy as np
        qmax = 2 *  self.muXElem(mX,isotope_mass) * v
        q2max = qmax*qmax
        qref = self.alpha * self.mElectron
        # qref = 1e-3
        # qref = self.mElectron
        # sigmaP_bar = 18 * pi * alpha*alphaD * epsilon^2 * muXP ^2 / (qref + ma_prime^2)^2
        # a = 1 /4 (9 pi^2 / 2*Z) ^ 1/3
        # a0 = 0.89 *a0 / z^1/3
        a = (1/4)*((9*np.pi**2)/2/z)**(1/3)*self.Bohr_Radius
        
        x = a*a*q2max
        y = a*a * qref*qref
        
        if FDMn == 0 and not doScreen:
            fdm_factor = 1
        elif FDMn == 2:
            doScreen=False
            fdm_factor = y*y / (1+x)
        if doScreen and FDMn == 0:
            fdm_factor = (1+ (1/(1+x)) - (2/x)*np.log(1+x))
        si= sigmaP*((self.muXElem(mX,isotope_mass)/self.muXElem(mX,self.mNucleon))**2) *(z**2)* fdm_factor 

        return si #same units as sigmaP



    def Mean_Free_Path(self,r,mX,sigmaP,v,FDMn,doScreen=True):
        #r in natural units
        #mX in GeV
        #v in c
        #sigmaP in cm^2
        #convert sigmaP into energy units
        

        lambda_inv = 0
        sigmaP *= self.cm**2# [1/ev^2]
        self.get_layer(r) 
        density = self.density #natural units
        num_isotopes = len(self.Elements)
        for i in range(num_isotopes):
            Element= self.Elements[i]
            fractional_density = Element[2]
            Z = Element[0]
            N = Element[1]
            isotope_mass = self.NucleusMass(N) #GeV
            si = self.sigma_i(v,isotope_mass,Z,sigmaP,mX,FDMn,doScreen)
            # print('fractional_density,density/isotope_mass,si')
            # print(fractional_density,density/isotope_mass,si)
# 
            lambda_inv+= fractional_density * (density/isotope_mass) *si #in units of GeV
        
        mfp = 1/lambda_inv #[1/GeV]
        mfp /= self.EarthRadius
        return mfp
    

