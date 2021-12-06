import pandas as pd
import pickle
from astropy.io import fits
from astropy.table import Table
import re
from spec_tools import Gen_PPF, boot_to_posterior, convert_sfh, Derive_SFH_weights, Highest_density_region
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import astropy.units as u
### set home for files
hpath = os.environ['HOME'] + '/'

#sfh_path = '/fdata/scratch/vestrada78840/SFH/'
#sfh_path = '../data/SFH/'

#zdb = pd.read_pickle(sfh_path + 'zfit_catalog.pkl')
    
class Gen_SFH(object):
    def __init__(self, field, galaxy, trials = 1000):
        ppf_dict = {}
        flist = glob(pos_path + '{}_{}_*_Pm*.npy'.format(field,galaxy))

        for f in flist:
            ext = re.split('{}_{}_'.format(field,galaxy),re.split('_Pm[0-9].npy', os.path.basename(f))[0])[1]
            if ext in ['tabfit', 'SFfit_p1']:
                fext = ext
                break

        if fext == 'tabfit':
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
            x,px = np.load(pos_path + '{}_{}_{}_Pz.npy'.format(field, galaxy,fext))
            rshift = x[px == max(px)][0]
        else:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
            rshift = zdb.query('id == {}'.format(galaxy)).zfit.values[0]
                     
        for i in params:
            x,px = np.load(pos_path + '{}_{}_{}_P{}.npy'.format(field, galaxy, fext, i))
            ppf_dict[i] = Gen_PPF(x,px)

        idx = 0

        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        mwa_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                masses = draw[1:len(params) - 1]
                lmass = draw[-1]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum=len(params) - 2), masses, maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**lmass / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))
                t_80_grid.append(interp1d(M/ M[-1], T)(0.2))
                t_90_grid.append(interp1d(M/ M[-1], T)(0.1))

                sfrmax = np.argmax(lbsfr) 

                sfr_grid.append(interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes))

                ssfr_grid.append(lbsfr[0] / 10**lmass)
                
                mwa_grid.append(np.trapz(sfr_grid[idx]*self.fulltimes,self.fulltimes)/np.trapz(sfr_grid[idx],self.fulltimes))
                idx +=1
            except:
                pass

        SFH = []
        SFH_16 = []
        SFH_84 = []
        ftimes = []
        for i in range(len(np.array(sfr_grid).T)):
            adat = np.array(np.array(sfr_grid).T[i])
            gdat = adat[adat>0]
            if len(gdat) < trials * 0.1:
                break
            else:
                SFH.append(np.percentile(gdat,50))
                SFH_16.append(np.percentile(gdat,16))
                SFH_84.append(np.percentile(gdat,84))

                ftimes.append(self.fulltimes[i])
                
        self.SFH = np.array(SFH)
        self.SFH_16 = np.array(SFH_16)
        self.SFH_84 = np.array(SFH_84)
        self.LBT = np.array(ftimes)
        
        self.sfr_grid = np.ma.masked_less_equal(sfr_grid,1E-10)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### mwa values
        x,y = boot_to_posterior(mwa_grid[0:trials], weights)
        self.mwa, self.mwa_hci, self.mwa_offreg = Highest_density_region(y,x)
    
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_80_grid[0:trials], weights)
        self.t_80, self.t_80_hci, self.t_80_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        
        self.t_50 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.5)
        self.t_80 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.8)
        self.t_90 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.9)

        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)

        self.z_80 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_80)*u.Gyr)
        hci=[]
        for lims in self.t_80_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_80_hci = np.array(hci)
        self.z_80_offreg = np.array(self.t_80_offreg)
                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)
                              
        x,y = boot_to_posterior(np.log10(ssfr_grid[0:trials]), weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
        
class Gen_sim_SFH(object):
    def __init__(self, fname, trials = 1000, rshift = None):
        ppf_dict = {}
        fname = glob(pos_path + '{}'.format(fname))[0]
        fit_db = np.load(fname, allow_pickle = True).item()

        fext = os.path.basename(fname).split('_')[2]
        try:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm']
            x = fit_db['z']
            px = fit_db['Pz']
            rshift = x[px == max(px)][0]
            
        except:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Plm']
            
        for i in range(len(params)):
            x = fit_db[params[i]]
            px = fit_db[P_params[i]]
            ppf_dict[params[i]] = Gen_PPF(x,px)

        idx = 0

        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        SFR_grid = []
        ssfr_grid = []
        ssfr10_grid = []
        ssfr1000_grid = []
        
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        mwa_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                masses = draw[1:len(params) - 1]
                lmass = draw[-1]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum=len(params) - 2), masses, maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**lmass / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))
                t_80_grid.append(interp1d(M/ M[-1], T)(0.2))
                t_90_grid.append(interp1d(M/ M[-1], T)(0.1))

                sfrmax = np.argmax(lbsfr) 
                sfrmini = interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes)
                sfr_grid.append(sfrmini)

                
                SFR_grid.append((np.trapz(sfrmini[:11], self.fulltimes[:11])/0.1))
                ssfr_grid.append((np.trapz(sfrmini[:11], self.fulltimes[:11])/0.1) / (10**lmass))
                ssfr10_grid.append((np.trapz(sfrmini[:2], self.fulltimes[:2])/0.01) / (10**lmass))
                ssfr1000_grid.append((np.trapz(sfrmini[:101], self.fulltimes[:101])) / (10**lmass))
                
                
                mwa_grid.append(np.trapz(sfr_grid[idx]*self.fulltimes,self.fulltimes)/np.trapz(sfr_grid[idx],self.fulltimes))
                idx +=1
            except:
                pass

        SFH = []
        SFH_16 = []
        SFH_84 = []
        ftimes = []
        for i in range(len(np.array(sfr_grid).T)):
            adat = np.array(np.array(sfr_grid).T[i])
            gdat = adat[adat>0]
            if len(gdat) < trials * 0.1:
                break
            else:
                SFH.append(np.percentile(gdat,50))
                SFH_16.append(np.percentile(gdat,16))
                SFH_84.append(np.percentile(gdat,84))

                ftimes.append(self.fulltimes[i])
                
        self.SFH = np.array(SFH)
        self.SFH_16 = np.array(SFH_16)
        self.SFH_84 = np.array(SFH_84)
        self.LBT = np.array(ftimes)
        
        self.sfr_grid = np.ma.masked_less_equal(sfr_grid,1E-10)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### mwa values
        x,y = boot_to_posterior(mwa_grid[0:trials], weights)
        self.mwa, self.mwa_hci, self.mwa_offreg = Highest_density_region(y,x)
    
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        self.t50 = x
        self.Pt50 = y
        
        x,y = boot_to_posterior(t_80_grid[0:trials], weights)
        self.t_80, self.t_80_hci, self.t_80_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        self.t90 = x
        self.Pt90 = y
        
        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)

        self.z_80 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_80)*u.Gyr)
        hci=[]
        for lims in self.t_80_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_80_hci = np.array(hci)
        self.z_80_offreg = np.array(self.t_80_offreg)
                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)
        
        
        ######### SFRs
        x,y = boot_to_posterior(ssfr1000_grid[0:trials], weights)
        self.ssfr1000, self.ssfr1000_hci, self.ssfr1000_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(ssfr10_grid[0:trials], weights)
        self.ssfr10, self.ssfr10_hci, self.ssfr10_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(ssfr_grid[0:trials], weights)
        self.ssfr, self.ssfr_hci, self.ssfr_offreg = Highest_density_region(y,x)
        self.sSFR = x
        self.PsSFR = y
        
        x,y = boot_to_posterior(SFR_grid[0:trials], weights)
        self.SFR, self.SFR_hci, self.SFR_offreg = Highest_density_region(y,x)
        self.SFR_d = x
        self.PSFR = y
        
####
class Gen_sim_SFH_phot(object):
    def __init__(self, fname, trials = 1000, rshift = None):
        ppf_dict = {}
        fname = glob(pos_path + '{}'.format(fname))[0]
        fit_db = np.load(fname, allow_pickle = True).item()

        fext = os.path.basename(fname).split('_')[2]
        if len(fit_db)==69:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm']
            x = fit_db['z']
            px = fit_db['Pz']
            rshift = x[px == max(px)][0]
            
        else:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Plm']
            x = fit_db['z']
            px = fit_db['Pz']
            rshift = x[px == max(px)][0]
            
        for i in range(len(params)):
            x = fit_db[params[i]]
            px = fit_db[P_params[i]]
            ppf_dict[params[i]] = Gen_PPF(x,px)

        idx = 0

        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        mwa_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                masses = draw[1:len(params) - 1]
                lmass = draw[-1]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum=len(params) - 2), masses, maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**lmass / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))
                t_80_grid.append(interp1d(M/ M[-1], T)(0.2))
                t_90_grid.append(interp1d(M/ M[-1], T)(0.1))

                sfrmax = np.argmax(lbsfr) 
                sfrmini = interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes)
                sfr_grid.append(sfrmini)

                ssfr_grid.append(np.log10((np.trapz(sfrmini[:11], self.fulltimes[:11])/0.1) / (10**lmass)))

                mwa_grid.append(np.trapz(sfr_grid[idx]*self.fulltimes,self.fulltimes)/np.trapz(sfr_grid[idx],self.fulltimes))
                idx +=1
            except:
                pass

        SFH = []
        SFH_16 = []
        SFH_84 = []
        ftimes = []
        for i in range(len(np.array(sfr_grid).T)):
            adat = np.array(np.array(sfr_grid).T[i])
            gdat = adat[adat>0]
            if len(gdat) < trials * 0.1:
                break
            else:
                SFH.append(np.percentile(gdat,50))
                SFH_16.append(np.percentile(gdat,16))
                SFH_84.append(np.percentile(gdat,84))

                ftimes.append(self.fulltimes[i])
                
        self.SFH = np.array(SFH)
        self.SFH_16 = np.array(SFH_16)
        self.SFH_84 = np.array(SFH_84)
        self.LBT = np.array(ftimes)
        
        self.sfr_grid = np.ma.masked_less_equal(sfr_grid,1E-10)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### mwa values
        x,y = boot_to_posterior(mwa_grid[0:trials], weights)
        self.mwa, self.mwa_hci, self.mwa_offreg = Highest_density_region(y,x)
    
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        self.t50 = x
        self.Pt50 = y
        
        x,y = boot_to_posterior(t_80_grid[0:trials], weights)
        self.t_80, self.t_80_hci, self.t_80_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        self.t90 = x
        self.Pt90 = y
        
        #self.t_50 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.5)
        #self.t_80 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.8)
        #self.t_90 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.9)

        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)

        self.z_80 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_80)*u.Gyr)
        hci=[]
        for lims in self.t_80_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_80_hci = np.array(hci)
        self.z_80_offreg = np.array(self.t_80_offreg)
                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)
                              
        x,y = boot_to_posterior(ssfr_grid[0:trials], weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
        self.ssfr = x
        self.Pssfr = y
####
        
class Gen_SFH_p2(object):
    def __init__(self, field, galaxy, zgrizli, trials = 1000):
        ppf_dict = {}
        fname = glob(pos_path + '{}_{}_*p2*_fits.npy'.format(field,galaxy))[0]
        fit_db = np.load(fname, allow_pickle = True).item()

        fext = os.path.basename(fname).split('_')[2]

        if fext == 'tabfit':
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm']
            x = fit_db['z']
            px = fit_db['Pz']
            rshift = x[px == max(px)][0]
            
        else:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
            P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Plm']
            rshift = zgrizli
                     
        for i in range(len(params)):
            x = fit_db[params[i]]
            px = fit_db[P_params[i]]
            ppf_dict[params[i]] = Gen_PPF(x,px)

        idx = 0

        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        mwa_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                masses = draw[1:len(params) - 1]
                lmass = draw[-1]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum=len(params) - 2), masses, maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**lmass / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))
                t_80_grid.append(interp1d(M/ M[-1], T)(0.2))
                t_90_grid.append(interp1d(M/ M[-1], T)(0.1))

                sfrmax = np.argmax(lbsfr) 
                sfrmini = interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes)
                sfr_grid.append(sfrmini)

                ssfr_grid.append(np.log10((np.trapz(sfrmini[:11], self.fulltimes[:11])/0.1) / (10**lmass)))

                mwa_grid.append(np.trapz(sfr_grid[idx]*self.fulltimes,self.fulltimes)/np.trapz(sfr_grid[idx],self.fulltimes))
                idx +=1
            except:
                pass
        SFH = []
        SFH_16 = []
        SFH_84 = []
        ftimes = []
        for i in range(len(np.array(sfr_grid).T)):
            adat = np.array(np.array(sfr_grid).T[i])
            gdat = adat[adat>0]
            if len(gdat) < trials * 0.1:
                break
            else:
                SFH.append(np.percentile(gdat,50))
                SFH_16.append(np.percentile(gdat,16))
                SFH_84.append(np.percentile(gdat,84))

                ftimes.append(self.fulltimes[i])
                
        self.SFH = np.array(SFH)
        self.SFH_16 = np.array(SFH_16)
        self.SFH_84 = np.array(SFH_84)
        self.LBT = np.array(ftimes)
        
        self.sfr_grid = np.ma.masked_less_equal(sfr_grid,1E-10)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### mwa values
        x,y = boot_to_posterior(mwa_grid[0:trials], weights)
        self.mwa, self.mwa_hci, self.mwa_offreg = Highest_density_region(y,x)
    
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_80_grid[0:trials], weights)
        self.t_80, self.t_80_hci, self.t_80_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        
        self.t_50 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.5)
        self.t_80 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.8)
        self.t_90 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.9)

        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)

        self.z_80 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_80)*u.Gyr)
        hci=[]
        for lims in self.t_80_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_80_hci = np.array(hci)
        self.z_80_offreg = np.array(self.t_80_offreg)
                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)
        print(trials)
        x,y = boot_to_posterior(ssfr_grid[0:trials], weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
        self.log_sSFR = x
        self.Plog_sSFR = y
        
class Gen_SFH_KI(object):
    def __init__(self, file, rshift, trials = 1000):
        ppf_dict = {}
        fit_db = np.load(file, allow_pickle = True).item()

        params = ['t25', 't50', 't75', 'logssfr', 'lmass']
        P_params = ['Pt25', 'Pt50', 'Pt75', 'Plogssfr', 'Plmass']
                     
        for i in range(len(params)):
            x = fit_db[params[i]]
            px = fit_db[P_params[i]]
            ppf_dict[params[i]] = Gen_PPF(x,px)

        idx = 0

        sfr_grid = []
        t_90_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                t25, t50, t75 = draw[0:3]
                lmass = draw[-1]
                log_ssfr = draw[3]
                log_sfr = np.log10(10**log_ssfr * 10**lmass)
                
                sfh_tuple = np.hstack([lmass, log_sfr, 3, t25,t50,t75])
                sfh, timeax = tuple_to_sfh_stand_alone(sfh_tuple, rshift)
                
                sfr_grid.append(sfh[::-1])
                
                SFH = (sfh/np.trapz(sfh,timeax))[::-1]
                ICSFH = interp1d((np.cumsum(SFH[::-1]) /np.cumsum(SFH[::-1])[-1])[::-1],timeax)
                t_90_grid.append(ICSFH(0.9))

                idx +=1
            except:
                pass
        SFH = []
        SFH_16 = []
        SFH_84 = []

        self.SFH = np.percentile(sfr_grid,50, axis=0)
        self.SFH_16 = np.percentile(sfr_grid,16, axis=0)
        self.SFH_84 = np.percentile(sfr_grid,84, axis=0)
        self.LBT = np.array(timeax)
        
        self.sfr_grid = np.array(sfr_grid)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### t values
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        self.t_90 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.9)

        ####### z values                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)

class Gen_NIRISS_SFH(object):
    def __init__(self, fname, trials = 1000, rshift = None):
        ppf_dict = {}
        fit_db = np.load(fname, allow_pickle = True).item()

        params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
        P_params = ['Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm']
        x = fit_db['z']
        px = fit_db['Pz']
        rshift = x[px == max(px)][0]
            
        for i in range(len(params)):
            x = fit_db[params[i]]
            px = fit_db[P_params[i]]
            ppf_dict[params[i]] = Gen_PPF(x,px)

        idx = 0

        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        mwa_grid = []
        
        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                masses = draw[1:len(params) - 1]
                lmass = draw[-1]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum=len(params) - 2), masses, maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**lmass / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))
                t_80_grid.append(interp1d(M/ M[-1], T)(0.2))
                t_90_grid.append(interp1d(M/ M[-1], T)(0.1))

                sfrmax = np.argmax(lbsfr) 
                sfrmini = interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes)
                sfr_grid.append(sfrmini)

                ssfr_grid.append(np.log10((np.trapz(sfrmini[:11], self.fulltimes[:11])/0.1) / (10**lmass)))

                mwa_grid.append(np.trapz(sfr_grid[idx]*self.fulltimes,self.fulltimes)/np.trapz(sfr_grid[idx],self.fulltimes))
                idx +=1
            except:
                pass

        SFH = []
        SFH_16 = []
        SFH_84 = []
        ftimes = []
        for i in range(len(np.array(sfr_grid).T)):
            adat = np.array(np.array(sfr_grid).T[i])
            gdat = adat[adat>0]
            if len(gdat) < trials * 0.1:
                break
            else:
                SFH.append(np.percentile(gdat,50))
                SFH_16.append(np.percentile(gdat,16))
                SFH_84.append(np.percentile(gdat,84))

                ftimes.append(self.fulltimes[i])
                
        self.SFH = np.array(SFH)
        self.SFH_16 = np.array(SFH_16)
        self.SFH_84 = np.array(SFH_84)
        self.LBT = np.array(ftimes)
        
        self.sfr_grid = np.ma.masked_less_equal(sfr_grid,1E-10)

        weights = Derive_SFH_weights(self.SFH, sfr_grid[0:trials])
       
        ####### mwa values
        x,y = boot_to_posterior(mwa_grid[0:trials], weights)
        self.mwa, self.mwa_hci, self.mwa_offreg = Highest_density_region(y,x)
    
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50_grid = t_50_grid
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        self.t50 = x
        self.Pt50 = y
        
        x,y = boot_to_posterior(t_80_grid[0:trials], weights)
        self.t_80, self.t_80_hci, self.t_80_offreg = Highest_density_region(y,x)
        
        x,y = boot_to_posterior(t_90_grid[0:trials], weights)
        self.t_90_grid = t_90_grid

        self.t_90, self.t_90_hci, self.t_90_offreg = Highest_density_region(y,x)
        self.t90 = x
        self.Pt90 = y

        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)

        self.z_80 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_80)*u.Gyr)
        hci=[]
        for lims in self.t_80_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_80_hci = np.array(hci)
        self.z_80_offreg = np.array(self.t_80_offreg)
                       
        self.z_90 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_90)*u.Gyr)
        hci=[]
        for lims in self.t_90_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_90_hci = np.array(hci)
        self.z_90_offreg = np.array(self.t_90_offreg)
                              
        x,y = boot_to_posterior(ssfr_grid[0:trials], weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
        self.ssfr = x
        self.Pssfr = y