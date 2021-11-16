#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from spec_exam import Gen_spec_2D
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    
verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
zscale = 0.0035 * (1 + specz)

def Full_forward_model_3(spec, wave, flux, specz, wvs, flxs, errs, beams, trans):
    Gmfl = []
    for i in range(len(wvs)):
        Gmfl.append(forward_model_all_beams(beams[i], trans[i], wvs[i], wave * (1 + specz), flux))
        
    spec.sim_phot(wave * (1 + specz), flux)

    return np.array(Gmfl), np.array(spec.Pmfl)


def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8], u[9], u[10],u[11]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a))
    
    lm = Gaussian_prior(u[12], [9.5, 12.5], 11, 0.75)
  
    z = stats.norm.ppf(u[13],loc = specz, scale = zscale)
    
    d = log_10_prior(u[14],[1E-3,2])

    bp1 = Gaussian_prior(u[15], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[16], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[17], [0.1,10])
    bb = log_10_prior(u[18], [0.0001,1])
    bl = log_10_prior(u[19], [0.01,1])
    
    ra = log_10_prior(u[20], [0.1,10])
    rb = log_10_prior(u[21], [0.0001,1])
    rl = log_10_prior(u[22], [0.01,1])
       
    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
Gs = Gen_spec_2D(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 
####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)
#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 23, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results

np.save('{0}_{1}_tabfit'.format(field, galaxy), dres) 
# np.save(out_path + '{0}_{1}_tabfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm',
          'z', 'd', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
#     np.save(pos_path + '{0}_{1}_tabfit_P{2}'.format(field, galaxy, params[i]),[t,pt])
    np.save('{0}_{1}_tabfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,\
    bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa= dres.samples[-1]

# np.save(pos_path + '{0}_{1}_tabfit_bfit'.format(field, galaxy),
np.save('{0}_{1}_tabfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,
         bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa, dres.logl[-1]])