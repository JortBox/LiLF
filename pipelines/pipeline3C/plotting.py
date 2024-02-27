#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import aplpy
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

import os, sys, pickle
import targets
from analysis import *

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt



plt.rc('font', family='serif')
#plt.rc('text',usetex=True)







def restructure_fits(path: str, size: int = 200):
    size = int(np.round(size/2))
    
    oldhdul = fits.open(path)
    oldhdu: fits.PrimaryHDU = oldhdul[0] # type: ignore

    oldhdu.header.remove("CTYPE3")
    oldhdu.header.remove("CRPIX3")
    oldhdu.header.remove("CRVAL3")
    oldhdu.header.remove("CDELT3")
    oldhdu.header.remove("CUNIT3")

    oldhdu.header.remove("CTYPE4")
    oldhdu.header.remove("CRPIX4")
    oldhdu.header.remove("CRVAL4")
    oldhdu.header.remove("CDELT4")
    oldhdu.header.remove("CUNIT4")

    print((oldhdu.header["BMAJ"]*u.deg).to(u.arcsec))
    print((oldhdu.header["BMIN"]*u.deg).to(u.arcsec))
    
    nax = np.asarray(oldhdu.data.shape[2:])//2 # type: ignore
    hdu = fits.PrimaryHDU(oldhdu.data[0,0,nax[0]-size:nax[0]+size,nax[1]-size:nax[1]+size], header=oldhdu.header) # type: ignore
    hdu.writeto('test.fits', overwrite=True)
    
    
def plot_galaxy(source: Source3C, suffix: str = "", vmin=None, vmax=None):
    #levels = np.array([1.6**i for i in range(2,20)]) * source.rms
    levels = np.array([10]) * source.rms
    
    fontsize = 18
    if suffix != "":
        suffix = "_" + suffix
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    scale = cosmo.kpc_proper_per_arcmin(source.z)  
    
    restructure_fits(source.path, size = (2.*u.Mpc / scale).to(source.pixScale).value)
    
    gc = aplpy.FITSFigure('test.fits', subplot=[0., 0., .72, .8])

    gc.show_colorscale(cmap='gist_heat_r', vmin=vmin, vmax=vmax)
    gc.show_contour('test.fits', colors='black', levels=levels, alpha=0.9) # type: ignore

    gc.add_beam(color="none", edgecolor="black")
    gc.add_label(0.495, 0.94, source.name, relative=True, size=25)
    
    gc.add_colorbar()
    gc.colorbar.set_axis_label_text("Jy beam$^{-1}$")
    gc.colorbar.set_axis_label_font(size=fontsize)
    gc.colorbar.set_font(size=fontsize)
    
    gc.add_scalebar((500.*u.kpc/scale).to(u.deg).value , color="black", corner="bottom")
    gc.scalebar.set_label(f'{500} kpc')
    gc.scalebar.set_font(size=fontsize)

    gc.tick_labels.set_font(size=fontsize)
    gc.axis_labels.set_font(size=fontsize)
    gc.save(f"{source.name}{suffix}.png", dpi=300)
    
    os.remove('test.fits')
    
if __name__ == "__main__":
    catalog = Catalogue3C(["3c48"])
    
    DATA_DIR = "/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/"

    
    for source in catalog:
        source.set_data(DATA_DIR + source.name + "/img/img-final-MFS-image.fits")
        get_integrated_flux(source)
        #plot_galaxy(source, vmax=0.75*np.max(source.data.value), vmin=-5*source.rms.value)
