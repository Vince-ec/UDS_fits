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
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import fsps
from sim_engine import *
from matplotlib import gridspec
hpath = os.environ['HOME'] + '/'


if hpath == '/home/vestrada78840/': # need to change
    data_path = '/scratch/user/vestrada78840/data/'
    chi_path = '/scratch/user/vestrada78840/chidat/'
    beam_path = '/scratch/user/vestrada78840/beams/'
    beam_2d_path = '/scratch/user/vestrada78840/beams/'
    template_path = '/scratch/user/vestrada78840/data/'
    out_path = '/scratch/user/vestrada78840/chidat/'
    phot_path = '/scratch/user/vestrada78840/phot/'

else:
    data_path = '../data/'
    chi_path = '../chidat/'
    beam_path = '' # path to the beam parameter files
    beam_2d_path = ''# path to the beams
    template_path = '../templates/'
    out_path = '../data/posteriors/'
    phot_path = '../phot/'
    
    
class Gen_spec_2D(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8000, 11300], g141_lims=[11300, 16500],
                phot_errterm = 0, irac_err = None, mask = False):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims

        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD
        galaxy_id - ID number from 3D-HST
        specz - redshift

        """
        ##load spec and phot
        self.Clean_multibeam()
        
        if self.g102:
            self.Bwv, self.Bfl, self.Ber = self.Gen_1D_spec(self.mb_g102, g102_lims, 'G102', self.specz, mask = mask)
            self.Bwv_rf = self.Bwv/(1 + self.specz)

        if self.g141:
            self.Rwv, self.Rfl, self.Rer = self.Gen_1D_spec(self.mb_g141, g141_lims, 'G141', self.specz, mask = mask)
            self.Rwv_rf = self.Rwv/(1 + self.specz)
        
#         self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
#                                 self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False, select = None)
         
#         self.Perr = apply_phot_err(self.Pflx, self.Perr, self.Pnum, base_err = phot_errterm, irac_err = irac_err)
#         # load photmetry precalculated values
#         self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
       
    def Sim_phot_premade(self, model_wave, model_flux):
        self.Pmfl = self.Sim_phot_mult(model_wave, model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)

        self.Pmfl = self.Pmfl * self.PC
     
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)
    
    def Clean_multibeam(self):
        if int(self.galaxy_id) < 10000:
            gid = '0' + str(self.galaxy_id)
        else:
            gid = self.galaxy_id
        BMX = np.load(beam_path +'{}_{}_ex.npy'.format(self.field, self.galaxy_id),allow_pickle=True)
        clip, clipspec, omitspec = np.load(beam_path +'{}_{}.npy'.format(self.field, self.galaxy_id),allow_pickle=True)
        if hpath == '/home/vestrada78840/': 
            fl = beam_2d_path + 'j021820m0510_{}.beams.fits'.format(gid)
        else:
            fl = beam_2d_path + 'j021820m0510_{}.beams.fits'.format(gid)

        mb = multifit.MultiBeam(fl,**args)
        blist = mb.beams

        #####clip or omit
        fblist = []

        idc = 0

        for bm in blist:
            if bm.grism.parent_file in BMX:            
                if clipspec[idc] == 1:
                    xspec, yspec, yerr = bm.beam.optimal_extract(bm.grism.data['SCI'] - bm.contam,ivar = bm.ivar) 
                    lms = clip[idc]
                    if len(lms) == 1:
                        lms = lms[0]
                    for i in range(len(xspec)):
                        if lms[0] < xspec[i]< lms[1]:
                            bm.grism.data['SCI'].T[i] = np.zeros_like(bm.grism.data['SCI'].T[i])
                            bm.grism.data['ERR'].T[i] = np.ones_like(bm.grism.data['ERR'].T[i])*1000  

                if omitspec[idc] == 1:
                    pass
                else:    
                    fblist.append(bm)

                idc += 1
    
            else:    
                fblist.append(bm)   


        mb = multifit.MultiBeam(fblist,**args)
        for b in mb.beams:
            if hasattr(b, 'xp'):
                delattr(b, 'xp')
        mb.initialize_masked_arrays()

        grism_beams = {}
        for g in mb.PA:
            grism_beams[g.lower()] = []
            for pa in mb.PA[g]:
                for i in mb.PA[g][pa]:
                    grism_beams[g.lower()].append(mb.beams[i])

        try:
            self.mb_g102 = multifit.MultiBeam(grism_beams['g102'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g102')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g102.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g102.initialize_masked_arrays()
            self.g102 = True
            
        except:
            self.g102 = False
            
        try:
            self.mb_g141 = multifit.MultiBeam(grism_beams['g141'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g141')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g141.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g141.initialize_masked_arrays()
            self.g141 = True
            
        except:
            self.g141 = False
            
    def Gen_1D_spec(self, MB, lims, instr, specz, mask = False):
        #if tfit != 'none':
        #    sptbl = MB.oned_spectrum(tfit = tfit)
        #else:
        #    sptbl = MB.oned_spectrum()
        temps = MB.template_at_z(specz, templates = args['t1'], fitter='lstsq')
        sptbl = MB.oned_spectrum(tfit = temps)

        w = sptbl[instr]['wave']
        f = sptbl[instr]['flux']
        e = sptbl[instr]['err']
        fl = sptbl[instr]['flat']

        clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1]]
        
        w = w[clip]
        f = f[clip]
        e = e[clip]
        fl = fl[clip]
        if mask:
            try:
                clip = np.repeat(True, len(w))

                m_fl = np.load(mask_path + '{}_{}_mask.npy'.format(self.field, self.galaxy_id),allow_pickle=True)
                for m in m_fl:
                    for i in range(len(w)):
                        if m[0] < w[i] < m[1]:
                            clip[i] = False

                w = w[clip]
                f = f[clip]
                e = e[clip]
                fl = fl[clip]
            except:
                print('no mask')

        return w[f>0], f[f>0]/fl[f>0], e[f>0]/fl[f>0]
