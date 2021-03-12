__author__ = 'vestrada'

import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from spec_stats import Highest_density_region
import fsps
import os
import re
# import img_scale
from glob import glob
# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# R = robjects.r
# pandas2ri.activate()

### set home for files
hpath = os.environ['HOME'] + '/'

h=6.6260755E-27 # planck constant erg s
c=3E10          # speed of light cm s^-1
atocm=1E-8    # unit to convert angstrom to cm
kb=1.38E-16	    # erg k-1

"""
FUNCTIONS:
-Median_w_Error_cont
-Median_w_Error
-Smooth
-Extract_BeamCutout
-Likelihood_contours
-Mag
-Oldest_galaxy
-Gauss_dist
-Scale_model
-Likelihood_contours
-Source_present
-Get_sensitivity
-Sig_int

CLASSES:
-Photometry
"""

def Median_w_Error(Pofx, x):
    #power = int(np.abs(min(np.log10(x)[np.abs(np.log10(x)) != np.inf])))+1
    
    lerr = 0
    herr = 0
    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = x[i]
        if med == 0:
            if e >= .50:
                med = x[i]
        if herr == 0:
            if e >= .84:
                herr = x[i]
                break
                
#    return np.round(med,power), med - lerr, herr - med
    return med, med - lerr, herr - med


def Median_w_Error_cont(Pofx, x):
    ix = np.linspace(x[0], x[-1], 10000)
    iP = interp1d(x, Pofx)(ix)

    C = np.trapz(iP,ix)

    iP/=C


    lerr = 0
    herr = 0
    med = 0

    for i in range(len(ix)):
        e = np.trapz(iP[0:i + 1], ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if med == 0:
            if e >= .50:
                med = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    return med, med - lerr, herr - np.abs(med)


def Smooth(f,x,bw):
    ksmooth = importr('KernSmooth')

    ### select bandwidth
    H = ksmooth.dpik(x)
    
    if bw == 'none':
        bw = H
    
    fx = ksmooth.locpoly(x,f,bandwidth = bw)
    X = np.array(fx[0])
    iFX = np.array(fx[1])
    return interp1d(X,iFX)(x)

def Extract_BeamCutout(target_id, grism_file, mosaic, seg_map, instruement, catalog):
    flt = model.GrismFLT(grism_file = grism_file ,
                          ref_file = mosaic, seg_file = seg_map,
                            pad=200, ref_ext=0, shrink_segimage=True, force_grism = instrument)
    
    # catalog / semetation image
    ref_cat = Table.read(catalog ,format='ascii')
    seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
    flt.compute_full_model(ids=seg_cat['id'])
    beam = flt.object_dispersers[target_id][2]['A']
    co = model.BeamCutout(flt, beam, conf=flt.conf)
    
    PA = np.round(fits.open(grism_file)[0].header['PA_V3'] , 1)
    
    co.write_fits(root='beams/o{0}'.format(PA), clobber=True)

    ### add EXPTIME to extension 0
    
    
    fits.setval('../beams/o{0}_{1}.{2}.A.fits'.format(PA, target_id, instrument), 'EXPTIME', ext=0,
            value=fits.open('../beams/o{0}_{1}.{2}.A.fits'.format(PA, target_id, instrument))[1].header['EXPTIME'])   


def Likelihood_contours(age, metallicty, prob):
    ####### Create fine resolution ages and metallicities
    ####### to integrate over
    m2 = np.linspace(min(metallicty), max(metallicty), 50)

    ####### Interpolate prob
    P2 = interp2d(metallicty, age, prob)(m2, age)

    ####### Create array from highest value of P2 to 0
    pbin = np.linspace(0, np.max(P2), 1000)
    pbin = pbin[::-1]

    ####### 2d integrate to find the 1 and 2 sigma values
    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(np.trapz(p, m2, axis=1), age)

    ######## Identify 1 and 2 sigma values
    onesig = np.abs(np.array(prob_int) - 0.68)
    twosig = np.abs(np.array(prob_int) - 0.95)

    return pbin[np.argmin(onesig)], pbin[np.argmin(twosig)]



def Mag(band):
    magnitude=25-2.5*np.log10(band)
    return magnitude

def Oldest_galaxy(z):
    return cosmo.age(z).value

def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return G / np.trapz(G, x)

def Scale_model(D, sig, M):
    return np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))

def Likelihood_contours(age, metallicty, prob):
    ####### Create fine resolution ages and metallicities
    ####### to integrate over
    m2 = np.linspace(min(metallicty), max(metallicty), 50)

    ####### Interpolate prob
    P2 = interp2d(metallicty, age, prob)(m2, age)

    ####### Create array from highest value of P2 to 0
    pbin = np.linspace(0, np.max(P2), 1000)
    pbin = pbin[::-1]

    ####### 2d integrate to find the 1 and 2 sigma values
    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(np.trapz(p, m2, axis=1), age)

    ######## Identify 1 and 2 sigma values
    onesig = np.abs(np.array(prob_int) - 0.68)
    twosig = np.abs(np.array(prob_int) - 0.95)

    return pbin[np.argmin(onesig)], pbin[np.argmin(twosig)]

def Source_present(fn,ra,dec):  ### finds source in flt file, returns if present and the pos in pixels
    flt=fits.open(fn)
    present = False
    
    w = wcs.WCS(flt[1].header)

    xpixlim=len(flt[1].data[0])
    ypixlim=len(flt[1].data)

    [pos]=w.wcs_world2pix([[ra,dec]],1)

    if -100 <pos[0]< (xpixlim - 35) and 0 <pos[1]<ypixlim and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
        present=True
            
    return present,pos
    
def Get_Sensitivity(filter_num):
    f=open(hpath + '/eazy-photoz/filters/FILTER.RES.latest','r')
    data=f.readlines()
    rows=[]
    for i in range(len(data)):
        rows.append(data[i].split())

    i=0
    sens_data=[]
    while i < len(data):
        sdata=[]
        amount=int(rows[i][0])
        for u in range(amount):
            r=np.array(rows[i+u+1])
            sdata.append(r.astype(np.float))
        sens_data.append(sdata)
        i=i+amount+1

    sens_wave=[]
    sens_func=[]
    s_wave=[]
    s_func=[]
    for i in range(len(sens_data[filter_num-1])):
        s_wave.append(sens_data[filter_num-1][i][1])
        s_func.append(sens_data[filter_num-1][i][2])

    for i in range(len(s_func)):
        if .001 < s_func[i]:
            sens_func.append(s_func[i])
            sens_wave.append(s_wave[i])

    
    trans = np.array(sens_func) / np.max(sens_func)
    sens_wv = np.array(sens_wave)
    
    tp = np.trapz(((trans * np.log(sens_wv)) / sens_wv), sens_wv)
    bm = np.trapz(trans / sens_wv, sens_wv)

    wave_eff = np.exp(tp / bm)
    
    return sens_wv, trans, wave_eff

def Sig_int(nu,er,trans,energy):
    sig = np.zeros(len(nu)-1)
    
    for i in range(len(nu)-1):
        sig[i] = (nu[i+1] - nu[i])/2 *np.sqrt(er[i]**2 * energy[i]**2 * trans[i]**2 + er[i+1]**2 * energy[i+1]**2 * trans[i+1]**2)
    
    return np.sum(sig) / np.trapz(trans * energy, nu)

class Photometry(object):

    def __init__(self,wv,fl,er,filter_number):
        self.wv = wv
        self.fl = fl
        self.er = er
        self.filter_number = filter_number

    def Get_Sensitivity(self, filter_num = 0):
        if filter_num != 0:
            self.filter_number = filter_num

        f = open(hpath + '/eazy-photoz/filters/FILTER.RES.latest', 'r')
        data = f.readlines()
        rows = []
        for i in range(len(data)):
            rows.append(data[i].split())
        i = 0
        sens_data = []
        while i < len(data):
            sdata = []
            amount = int(rows[i][0])
            for u in range(amount):
                r = np.array(rows[i + u + 1])
                sdata.append(r.astype(np.float))
            sens_data.append(sdata)
            i = i + amount + 1

        sens_wave = []
        sens_func = []
        s_wave = []
        s_func = []
        for i in range(len(sens_data[self.filter_number - 1])):
            s_wave.append(sens_data[self.filter_number - 1][i][1])
            s_func.append(sens_data[self.filter_number - 1][i][2])

        for i in range(len(s_func)):
            if .001 < s_func[i]:
                sens_func.append(s_func[i])
                sens_wave.append(s_wave[i])

        self.sens_wv = np.array(sens_wave)
        self.trans = np.array(sens_func)

    def Photo(self):
        wave = self.wv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans,energy) * (c / (wave_eff * atocm) ** 2)

    def Photo_clipped(self):

        IDX = [U for U in range(len(self.sens_wv)) if self.wv[0] < self.sens_wv[U] < self.wv[-1]]

        wave = self.wv * atocm
        filtnu = c /(self.sens_wv[IDX] * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans[IDX]
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans[IDX] * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans[IDX],energy) * (c / (wave_eff * atocm) ** 2)
        
    def Photo_model(self,mwv,mfl):

        wave = mwv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * mfl
        Fnu = interp1d(nu, fnu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)
        photo = photonu * (c / (wave_eff * atocm) ** 2)
        
        self.eff_mwv = wave_eff
        self.mphoto = photo

    def FWHM(self):
        top = np.trapz((self.trans * np.log(self.sens_wv/self.eff_wv)**2) / self.sens_wv, self.sens_wv)
        bot = np.trapz(self.trans / self.sens_wv, self.sens_wv)
        sigma = np.sqrt(top/bot)

        self.fwhm = np.sqrt(8*np.log(2))*sigma * self.eff_wv


def convert_sfh(agebins, mformed, epsilon=1e-4, maxage=None):
    #### create time vector
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
    bin_edges = np.unique(agebins_yrs)
    if maxage is None:
        maxage = agebins_yrs.max()  # can replace maxage with something else, e.g. tuniv
    t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
    t.sort()
    t = t[1:-1] # remove older than oldest bin, younger than youngest bin
    fsps_time = maxage - t

    #### calculate SFR at each t
    sfr = mformed / dt
    sfrout = np.zeros_like(t)
    sfrout[::2] = sfr
    sfrout[1::2] = sfr  # * (1+epsilon)

    return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

def get_agebins(maxage, binnum = 10):
    lages = [0,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9][: binnum + 1]
    
    nbins = len(lages) - 1

    tbinmax = (maxage * 0.85) * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(maxage*1e9)]
    return np.array([agelims[:-1], agelims[1:]]).T

from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde

def Gen_PPF(x,px):
    return interp1d(np.cumsum(px) / np.cumsum(px).max(),x)

def boot_to_posterior(values, weights):
    q = [0.5 - 0.5 * 0.999999426697, 0.5 + 0.5 * 0.999999426697]
    span = _quantile(values, q, weights=weights)

    s = 0.02

    bins = int(round(10. / 0.02))
    n, b = np.histogram(values, bins=bins, weights=weights,range=np.sort(span))
    n = norm_kde(n, 20.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    
    return x0, y0 / np.trapz(y0,x0)

def Derive_SFH_weights(SFH, grid):
    Y = np.array(SFH)

    weights = np.zeros(len(grid))
    for i in range(len(grid)):
        weights[i] = np.sum((grid[i][0:len(Y)] - Y) ** 2) ** -1
    return weights

rshifts = np.arange(0,14,0.01)
age_at_z = cosmo.age(rshifts).value
age_to_z = interp1d(age_at_z, rshifts)
lbt_at_z = cosmo.lookback_time(rshifts).value
lbt_to_z = interp1d(lbt_at_z, rshifts)

class Rescale_sfh(object):
    def __init__(self, field, galaxy, trials = 1000):

        ppf_dict = {}
        params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']

        for i in params:
            x,px = np.load('../data/posteriors/{0}_{1}_tabfit_P{2}.npy'.format(field, galaxy, i),allow_pickle=True)
            ppf_dict[i] = Gen_PPF(x,px)

        idx = 0
        x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pz.npy'.format(field, galaxy),allow_pickle=True)
        rshift = x[px == max(px)][0]
        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []
        t_80_grid = []
        t_90_grid = []
        t_q_grid = []

        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0]), draw[1:11], maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**draw[11] / 1E9

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

                check = False
                for i in range(len(lbsfr[0:sfrmax+1])):
                    if int(lbsfr[0:sfrmax+1][-(i+1)]) <= int(max(lbsfr[0:sfrmax+1]) * 0.1):
                        t_q_grid.append(lbt[0:sfrmax+1][-(i+1)])
                        break
                if not check:
                    t_q_grid.append(lbt[0:sfrmax+1][-(i+1)])

                sfr_grid.append(interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes))

                ssfr_grid.append(lbsfr[0] / 10**draw[11])
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
        
        x,y = boot_to_posterior(t_q_grid[0:trials], weights)
        self.t_q, self.t_q_hci, self.t_q_offreg = Highest_density_region(y,x) 

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
                       
        self.z_q = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_q)*u.Gyr)
        hci=[]
        for lims in self.t_q_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_q_hci = np.array(hci)
        self.z_q_offreg = np.array(self.t_q_offreg)
        
        x,y = boot_to_posterior(np.log10(ssfr_grid[0:trials]), weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
            
        self.weights = weights
        self.t_50_grid = t_50_grid
        self.t_80_grid = t_80_grid
        self.t_90_grid = t_90_grid
        
class Posterior_spec(object):
    def __init__(self, field, galaxy, trials = 1000):
        from sim_engine import F_lam_per_M
        sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

        ppf_dict = {}
        params = ['m','a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm','d']

        for i in params:
            x,px = np.load('../data/posteriors/{0}_{1}_tabfit_P{2}.npy'.format(field, galaxy, i),allow_pickle=True)
            ppf_dict[i] = Gen_PPF(x,px)

        x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pz.npy'.format(field, galaxy),allow_pickle=True)
        self.rshift = x[px == max(px)][0]

        idx = 0

        flam_grid=[]

        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                sp.params['dust2'] = draw[13]
                sp.params['dust1'] = draw[13]
                sp.params['logzsol'] = np.log10(draw[0])

                time, sfr, tmax = convert_sfh(get_agebins(draw[1]), draw[2:12], maxage = draw[1]*1E9)

                sp.set_tabular_sfh(time,sfr)    

                self.wave, flux = sp.get_spectrum(tage = draw[1], peraa = True)   

                flam_grid.append(F_lam_per_M(flux, self.wave * (1 + self.rshift), self.rshift, 0, sp.stellar_mass)*10**draw[12])

                idx +=1
            except:
                pass
            
        self.SPEC = np.percentile(flam_grid,50, axis = 0)
        self.SPEC_16 = np.percentile(flam_grid,16, axis = 0)
        self.SPEC_84 = np.percentile(flam_grid,84, axis = 0)
        

class Rescale_SF_sfh(object):
    def __init__(self, field, galaxy, rshift, run_name, trials = 1000):

        ppf_dict = {}
        params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
       
        for i in params:
            x,px = np.load('../Casey_data/posteriors/{0}_{1}_SF{2}_P{3}.npy'.format(field, galaxy, run_name, i),allow_pickle=True)
            ppf_dict[i] = Gen_PPF(x,px)

        idx = 0
        self.fulltimes = np.arange(0.0,Oldest_galaxy(rshift),0.01)
        sfr_grid = []
        ssfr_grid = []
        t_50_grid = []

        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                time, sfr, tmax = convert_sfh(get_agebins(draw[0], binnum = 6 ), draw[1:7], maxage = draw[0]*1E9)

                T=[0]
                M=[0]
                for i in range(len(time)//2):
                    mass = sfr[i*2+1] * (time[i*2+1] - time[i*2])
                    M.append(M[i] + mass)
                    T.append(time[i*2+1])

                sfr = sfr/ M[-1] * 10**draw[7] / 1E9

                lbt = np.abs(time - time[-1])[::-1]
                lbsfr = sfr[::-1]

                T=[0]
                M=[0]
                for i in range(len(lbt)//2):
                    mass = lbsfr[i*2+1] * (lbt[i*2+1] - lbt[i*2])
                    M.append(M[i] + mass)
                    T.append(lbt[i*2+1])

                t_50_grid.append(interp1d(M/ M[-1], T)(0.5))

                sfr_grid.append(interp1d(lbt,lbsfr,bounds_error=False,fill_value=0)(self.fulltimes))

                ssfr_grid.append(lbsfr[0] / 10**draw[7])
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
       
        ####### t values
        x,y = boot_to_posterior(t_50_grid[0:trials], weights)
        self.t_50, self.t_50_hci, self.t_50_offreg = Highest_density_region(y,x)
        self.t_50 = interp1d(np.cumsum(self.SFH[::-1]) / np.cumsum(self.SFH[::-1])[-1],self.LBT[::-1])(0.5)

        ####### z values
        self.z_50 = z_at_value(cosmo.age,(Oldest_galaxy(rshift) - self.t_50)*u.Gyr)
        hci=[]
        for lims in self.t_50_hci:
            hci.append(z_at_value(cosmo.age,(Oldest_galaxy(rshift) - lims)*u.Gyr))
        self.z_50_hci = np.array(hci)
        self.z_50_offreg = np.array(self.t_50_offreg)
        
        x,y = boot_to_posterior(np.log10(ssfr_grid[0:trials]), weights)
        self.lssfr, self.lssfr_hci, self.lssfr_offreg = Highest_density_region(y,x)
            
        self.weights = weights
        self.t_50_grid = t_50_grid
            
class Posterior_SF_spec(object):
    def __init__(self, field, galaxy, rshift, trials = 1000):
        from sim_engine import F_lam_per_M

        self.rshift = rshift
           
        sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
        sp.params['dust1'] = 0
        
        ppf_dict = {}
        params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm','d']

        for i in params:
            x,px = np.load('../data/posteriors/{0}_{1}_SFfit_p1_P{2}.npy'.format(field, galaxy, i),allow_pickle=True)
            ppf_dict[i] = Gen_PPF(x,px)

        idx = 0

        flam_grid=[]

        while idx < trials:
            try:
                draw = np.zeros(len(params))

                for i in range(len(draw)):
                    draw[i] = ppf_dict[params[i]](np.random.rand(1))[0]

                sp.params['dust2'] = draw[9]
                sp.params['logzsol'] = np.log10(draw[0])

                time, sfr, tmax = convert_sfh(get_agebins(draw[1], binnum = 6), draw[2:8], maxage = draw[1]*1E9)

                sp.set_tabular_sfh(time,sfr)    

                self.wave, flux = sp.get_spectrum(tage = draw[1], peraa = True)   

                flam_grid.append(F_lam_per_M(flux, self.wave * (1 + self.rshift), self.rshift, 0, sp.stellar_mass)*10**draw[8])

                idx +=1
            except:
                pass
            
        self.SPEC = np.percentile(flam_grid,50, axis = 0)
        self.SPEC_16 = np.percentile(flam_grid,16, axis = 0)
        self.SPEC_84 = np.percentile(flam_grid,84, axis = 0)
        
def IMG_pull(field, galaxy, smin = -0.1, smax = 0.3):
    if field[1] == 'S':
        seg = fits.open('/Volumes/Vince_CLEAR/gsd_img/goodss_3dhst.v4.0.F160W_seg.fits')[0].data
        f160 = fits.open('/Volumes/Vince_CLEAR/gsd_img/goodss_3dhst.v4.0.F160W_orig_sci.fits')[0].data
        f125 = fits.open('/Volumes/Vince_CLEAR/gsd_img/goodss_3dhst.v4.0.F125W_orig_sci.fits')[0].data
        f105 = fits.open('/Volumes/Vince_CLEAR/gsd_img/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits')[0].data

    if field[1] == 'N':
        seg = fits.open('/Volumes/Vince_CLEAR/gnd_img/goodsn_3dhst.v4.0.F160W_seg.fits')[0].data
        f160 = fits.open('/Volumes/Vince_CLEAR/gnd_img/goodsn_3dhst.v4.0.F160W_orig_sci.fits')[0].data
        f125 = fits.open('/Volumes/Vince_CLEAR/gnd_img/goodsn_3dhst.v4.0.F125W_orig_sci.fits')[0].data
        f105 = fits.open('/Volumes/Vince_CLEAR/gnd_img/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits')[0].data
        
    ###############img plot################
    idx = np.argwhere(seg == galaxy)

    ylist = np.arange(min(idx.T[0]), max(idx.T[0]) + 1, 1)
    xlist = np.arange(min(idx.T[1]), max(idx.T[1]) + 1, 1)

    segimg = seg[min(idx.T[0]): max(idx.T[0]), min(idx.T[1]): max(idx.T[1])]
    f105img = f105[min(idx.T[0]): max(idx.T[0]), min(idx.T[1]): max(idx.T[1])]
    f105img[segimg != galaxy] = 0
    maxloc = np.argwhere(f105img == np.max(f105img))[0]

    ycnt = ylist[maxloc[0]]
    xcnt = xlist[maxloc[1]]

    f105img = f105[ycnt - 40: ycnt + 41, xcnt - 40: xcnt + 41]
    f125img = f125[ycnt - 40: ycnt + 41, xcnt - 40: xcnt + 41]
    f160img = f160[ycnt - 40: ycnt + 41, xcnt - 40: xcnt + 41]

    img = np.zeros((f125img.shape[0], f125img.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.asinh(f160img, scale_min = smin, scale_max = smax)
    img[:,:,1] = img_scale.asinh(f125img, scale_min = smin, scale_max = smax)
    img[:,:,2] = img_scale.asinh(f105img, scale_min = smin, scale_max = smax)
    return img


class Gen_SFH(object):
    def __init__(self, field, galaxy, redshift, trials = 1000):

        ppf_dict = {}
        flist = glob('../data/posteriors/{}_{}_*_Pm*.npy'.format(field,galaxy))


        for f in flist:
            ext = re.split('{}_{}_'.format(field,galaxy),re.split('_Pm[0-9].npy', os.path.basename(f))[0])[1]
            if ext in ['tabfit', 'SFfit_p1']:
                fext = ext
                break

        if fext == 'tabfit':
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm']
            x,px = np.load('../data/posteriors/{}_{}_{}_Pz.npy'.format(field, galaxy,fext),allow_pickle=True)
            rshift = x[px == max(px)][0]
        else:
            params = ['a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm']
            rshift = redshift

        for i in params:
            x,px = np.load('../data/posteriors/{}_{}_{}_P{}.npy'.format(field, galaxy, fext, i),allow_pickle=True)
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
            
        self.weights = weights
        self.t_50_grid = t_50_grid
        self.t_80_grid = t_80_grid
        self.t_90_grid = t_90_grid
        self.mwa_grid = mwa_grid