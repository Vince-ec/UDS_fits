__author__ = 'vestrada'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from glob import glob
import os
from grizli import multifit
from grizli import model
from astropy.cosmology import Planck13 as cosmo
import fsps
from time import time
from sim_engine import *
from matplotlib import gridspec
from spec_exam import *
import sys
import dynesty
from scipy import stats
from spec_tools import Oldest_galaxy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from multiprocessing import Pool
from prospect.models.transforms import logsfr_ratios_to_masses
from spec_stats import Get_posterior
from scipy.special import erf, erfinv
import george
from george import kernels

hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/scratch/user/vestrada78840/data/'
    model_path ='/scratch/user/vestrada78840/fsps_spec/'
    chi_path = '/scratch/user/vestrada78840/chidat/'
    spec_path = '/scratch/user/vestrada78840/stack_specs/'
    beam_path = '/scratch/user/vestrada78840/beams/'
    template_path = '/scratch/user/vestrada78840/data/'
    out_path = '/scratch/user/vestrada78840/chidat/'
    pos_path = '/home/vestrada78840/posteriors/'
    phot_path = '/scratch/user/vestrada78840/phot/'
    sfh_path = '/scratch/user/vestrada78840/SFH/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
#     beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/out_dict/'
    pos_path = '../data/posteriors/'
    phot_path = '../phot/'

#################################
###functiongs needed for prior###
#################################

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

def get_lwa(params, agebins,sp):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_g'])
    sp.params['compute_light_ages'] = False
    
    return lwa
 
def get_lwa_SF(params, agebins,sp):
    m, a, m1, m2, m3, m4, m5, m6 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_g'])
    sp.params['compute_light_ages'] = False
    
    return lwa
    
def get_lwa_delay(params, agebins,sp):
    m, a, t = params

    sp.params['logzsol'] = np.log10(m)
    sp.params['tau'] = t 
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_g'])
    sp.params['compute_light_ages'] = False
    
    return lwa
    
####### may change to be agelim
def get_agebins(maxage, binnum = 10):
    lages = [0,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9][: binnum + 1]
    
    nbins = len(lages) - 1

    tbinmax = (maxage * 0.85) * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(maxage*1e9)]
    return np.array([agelims[:-1], agelims[1:]]).T



#############
####prior####
#############
def get_tx_vals(rand_vals, ncomp = 4):
    alpha = np.arange(ncomp-1, 0, -1)
    z_fraction = stats.beta.ppf(q=rand_vals, a = alpha*5, b=np.ones_like(alpha)*5)
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    return from_rand_to_dval(sfr_fraction)

def from_rand_to_dval(rvals):
    dvals = np.array([U/np.sum(rvals)  for U in rvals])
    x = []
    for i in range(len(dvals)):
        x.append(np.sum(dvals[0:i+1]))
    
    return x[:-1]

def log_10_prior(value, limits):
    """ Uniform prior in log_10(x) where x is the parameter. """
    value = 10**((np.log10(limits[1]/limits[0]))*value
                 + np.log10(limits[0]))
    return value

def Gaussian_prior(value, limits, mu, sigma):
    """ Gaussian prior between limits with specified mu and sigma. """
    uniform_max = erf((limits[1] - mu)/np.sqrt(2)/sigma)
    uniform_min = erf((limits[0] - mu)/np.sqrt(2)/sigma)
    value = (uniform_max-uniform_min)*value + uniform_min
    value = sigma*np.sqrt(2)*erfinv(value) + mu

    return value


def galfit_prior(u):
    m = (0.03*u[0] + 0.001) / 0.019
    
    a = (agelim - 1)* u[1] + 1
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9], u[10], u[11]])

    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)

    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a)) * 1E9
    
    z = stats.norm.ppf(u[12],loc = specz, scale = 0.005)
    
    d = u[13]
    
    bp1 = Gaussian_prior(u[14], [-0.5,0.5], 0, 0.25)
    
    rp1 = Gaussian_prior(u[15], [-0.5,0.5], 0, 0.25)
        
    ba = log_10(u[16], [0.1,10])
    bb = log_10(u[17], [0.0001,1])
    bl = log_10(u[18], [0.01,1])
    
    ra = log_10(u[19], [0.1,10])
    rb = log_10(u[20], [0.0001,1])
    rl = log_10(u[21], [0.01,1])
        
    lwa = get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a))

    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa]

def zfit_prior(u):
    m = (0.03*u[0] + 0.001) / 0.019
    
    a = (2)* u[1] + 1
    
    z = 2.5 * u[2]
        
    return [m, a, z]


##########################
#functions for likelihood#
##########################


class noise_model(object):
    """ A class for modelling the noise properties of spectroscopic
    data, including correlated noise.
    Parameters
    ----------
    spectrum : array_like
        The spectral data to which the calibration model is applied.
    spectral_model : array_like
        The physical model which is being fitted to the data.
    """

    def __init__(self, spectrum, spectral_model):
        self.max_y = np.max(spectrum[:, 1])

        # Normalise the data in y by dividing througy by max value.
        self.y = spectrum[:, 1]/self.max_y
        self.y_err = spectrum[:, 2]/self.max_y
        self.y_model = spectral_model/self.max_y

        self.diff = self.y - self.y_model

        # Normalise the data in x.
        self.x = spectrum[:, 0] - spectrum[0, 0]
        self.x /= self.x[-1]

    def GP_exp_squared(self, a, b, l):
        """ A GP noise model including an exponenetial squared kernel
        for corellated noise and white noise (jitter term). """

        scaling = a

        norm = b
        length = l

        kernel = norm**2*kernels.ExpSquaredKernel(length**2)
        self.gp = george.GP(kernel)
        self.gp.compute(self.x, self.y_err*scaling)


def lnlike_phot(Pflx, Perr, Mpflx):
    """ Calculates the log-likelihood for photometric data. """

    diff = (Pflx - Mpflx)**2
    chisq_phot = np.sum(diff*(1 / Perr**2))

    log_error_factors = np.log(2*np.pi*Perr**2)
    K_phot = -0.5*np.sum(log_error_factors)
    
    return K_phot - 0.5*chisq_phot


def forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def Full_scale(spec, Pmfl):
    return Scale_model(spec.Pflx, spec.Perr, Pmfl)

def Gather_grism_data(spec):
    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    
    if spec.g102:
        wvs.append(spec.Bwv)
        flxs.append(spec.Bfl)
        errs.append(spec.Ber)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.Rfl)
        errs.append(spec.Rer)
        beams.append(spec.Rbeam)
        trans.append(spec.Rtrans)

    return np.array([wvs, flxs, errs, beams, trans])

def Gather_simgrism_data(spec):
    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    
    if spec.g102:
        wvs.append(spec.Bwv)
        flxs.append(spec.SBfl)
        errs.append(spec.SBer)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.SRfl)
        errs.append(spec.SRer)
        beams.append(spec.Rbeam)
        trans.append(spec.Rtrans)

    return np.array([wvs, flxs, errs, beams, trans])

def Full_forward_model(spec, wave, flux, specz, wvs, flxs, errs, beams, trans):
    Gmfl = []
    
    for i in range(len(wvs)):
        Gmfl.append(forward_model_all_beams(beams[i], trans[i], wvs[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl

def Full_calibrate(Gmfl, p1, sc, wvs):
    for i in range(len(wvs)):
        rGmfl= Gmfl[i] * (p1[i] * (wvs[i] -(wvs[i][-1] + wvs[i][0])/2 ) + 1E3)
        scale = Scale_model(Gmfl[i],np.ones_like(Gmfl[i]),rGmfl)
        Gmfl[i] = scale * rGmfl * sc[i]
    return Gmfl

def Full_calibrate_2(Gmfl, p1, wvs, flxs, errs):
    for i in range(len(wvs)):
        rGmfl= Gmfl[i] * (p1[i] * (wvs[i] -(wvs[i][-1] + wvs[i][0])/2 ) + 1E3)
        scale = Scale_model(flxs[i], errs[i], rGmfl)
        Gmfl[i] = scale * rGmfl
    return Gmfl

def Calibrate_grism(spec, Gmfl, p1):
    lines = (p1 * (spec[0] -(spec[0][-1] + spec[0][0])/2 ) + 1E3)
    scale = Scale_model(spec[1]  / lines, spec[2] / lines, Gmfl)    
    return scale * lines


def Full_fit(spec, Gmfl, Pmfl, wvs, flxs, errs):
    Gchi = 0
    
    for i in range(len(wvs)):
        scale = Scale_model(flxs[i], errs[i], Gmfl[i])
        Gchi = Gchi + np.sum(((((flxs[i] / scale) - Gmfl[i]) / (errs[i] / scale))**2))
    
    Pchi = np.sum((((spec.Pflx - Pmfl) / spec.Perr)**2))
    
    return Gchi, Pchi

def Full_fit_2(spec, Gmfl, Pmfl, a, b, l, wvs, flxs, errs): 
    Gln = 0
    
    for i in range(len(wvs)):
        noise = noise_model(np.array([wvs[i],flxs[i], errs[i]]).T, Gmfl[i])
        noise.GP_exp_squared(a[i],b[i],l[i])
        Gln += noise.gp.lnlikelihood(noise.diff)

    Pln = lnlike_phot(spec.Pflx, spec.Perr, Pmfl)
    
    return Gln + Pln

def Full_fit_sim(spec, Gmfl, Pmfl, a, b, l, wvs, flxs, errs): 
    Gln = 0
    
    for i in range(len(wvs)):
        noise = noise_model(np.array([wvs[i],flxs[i], errs[i]]).T, Gmfl[i])
        noise.GP_exp_squared(a[i],b[i],l[i])
        Gln += noise.gp.lnlikelihood(noise.diff)

    Pln = lnlike_phot(spec.SPflx, spec.SPerr, Pmfl)
    
    return Gln + Pln

#####################
#####Likelihoods#####
#####################

def galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr)    
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z, wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate(Gmfl, [bp1, rp1], wvs, flxs, errs)
        
    PC= Full_scale(Gs, Pmfl)

    LOGL = Full_fit_2(Gs, Gmfl, PC*Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)
                 
#     return -0.5 * (Pchi+Gchi)

    return LOGL

def zfit_L(X):
    m, a, z = X

    sp.params['logzsol'] = np.log10(m)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z, wvs, flxs, errs, beams, trans)
              
    PC= Full_scale(Gs, Pmfl)

    Gchi, Pchi = Full_fit(Gs, Gmfl, PC*Pmfl, wvs, flxs, errs)
                  
    return -0.5 * (Gchi + Pchi)

def Gather_grism_data_from_2d(spec, sp):
    time, sfr, tmax = convert_sfh(get_agebins(5, binnum = 6), [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], maxage = 5*1E9)
    sp.set_tabular_sfh(time,sfr) 
    model_wave, model_flux = sp.get_spectrum(tage = 3.6, peraa = True)

    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    if spec.g102:
        wvs.append(np.array(spec.Bwv))
        flxs.append(np.array(spec.Bfl))
        errs.append(np.array(spec.Ber))
        BEAMS = []
        PAlist = []
        for bm in spec.mb_g102.beams:
            if bm.get_dispersion_PA() not in PAlist:
                PAlist.append(bm.get_dispersion_PA())
                BEAMS.append(bm)
        beams.append(BEAMS)
        
        TRANS = []
        for i in BEAMS:
            W, F = forward_model_grism(i, model_wave, np.ones(len(model_wave)))
            TRANS.append(interp1d(W,F)(spec.Bwv))     
        trans.append(TRANS)  

    if spec.g141:
        wvs.append(np.array(spec.Rwv))
        flxs.append(np.array(spec.Rfl))
        errs.append(np.array(spec.Rer))
        BEAMS = []
        PAlist = []
        for bm in spec.mb_g141.beams:
            if bm.get_dispersion_PA() not in PAlist:
                PAlist.append(bm.get_dispersion_PA())
                BEAMS.append(bm)
        beams.append(BEAMS)
        
        TRANS = []
        for i in BEAMS:
            W, F = forward_model_grism(i, model_wave, np.ones(len(model_wave)))
            TRANS.append(interp1d(W,F)(spec.Rwv))     
        trans.append(TRANS)  
        
    return np.array([wvs, flxs, errs, beams, trans])

import george
from george import kernels
def gp_interpolator(x,y,res = 1000, Nparam = 3):
    
    yerr = np.zeros_like(y)
    yerr[2:(2+Nparam)] = 0.001/np.sqrt(Nparam)
    if len(yerr) > 26:
        yerr[2:(2+Nparam)] = 0.1/np.sqrt(Nparam)

    #kernel = np.var(yax) * kernels.ExpSquaredKernel(np.median(yax)+np.std(yax))
    #k2 = np.var(yax) * kernels.LinearKernel(np.median(yax),order=1)
    #kernel = np.var(y) * kernels.Matern32Kernel(np.median(y)) #+ k2
    kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) + kernels.LinearKernel(np.median(y), order=2))
    gp = george.GP(kernel)

    #print(xax.shape, yerr.shape)

    gp.compute(x.ravel(), yerr.ravel())

    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred, pred_var = gp.predict(y.ravel(), x_pred, return_var=True)
    return x_pred, y_pred

def tuple_to_sfh_stand_alone(sfh_tuple, zval):
    mass_quantiles = np.array([0., 0.,  0.25, 0.5,  0.75,1.,1.,1.,1., 1.  ])
    time_quantiles = np.array([0,0.01,sfh_tuple[3],sfh_tuple[4],sfh_tuple[5],0.96,0.97,0.98,0.99,1])

    time_arr_interp, mass_arr_interp = gp_interpolator(time_quantiles, mass_quantiles, Nparam = 3)
    sfh_scale = 10**(sfh_tuple[0])/(cosmo.age(zval).value*1e9/1000)
    sfh = np.diff(mass_arr_interp)*sfh_scale
    sfh[sfh<0] = 0
    sfh = np.insert(sfh,0,[0])

    return sfh, time_arr_interp * cosmo.age(zval).value