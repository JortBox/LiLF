#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import aplpy
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from pipeline_utils import source_angular_size

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
    
    if size == -1:
        hdu = fits.PrimaryHDU(oldhdu.data[0,0], header=oldhdu.header) # type: ignore
        hdu.writeto('test.fits', overwrite=True)
        return np.max(hdu.data)
    
    size = int(np.round(size/2))
    nax = np.asarray(oldhdu.data.shape[2:])//2 # type: ignore
    hdu = fits.PrimaryHDU(oldhdu.data[0,0,nax[0]-size:nax[0]+size,nax[1]-size:nax[1]+size], header=oldhdu.header) # type: ignore
    hdu.writeto('test.fits', overwrite=True)
    return np.max(hdu.data)


def plot_optical(source: Source3C):
    gc = aplpy.make_rgb_image(["/data/m82-infrared.fits","/data/m82-red.fits","/data/m82-blue.fits"], "m82-rgb.png", stretch_r='log', stretch_g='log', stretch_b='log')
    
def plot_galaxy(source: Source3C, suffix: str = "", vmin=None, vmax=None, size = None, annotation_color="white"):
    #levels = np.array([1.6**i for i in range(2,20)]) * source.rms
    levels = np.array([5, 10, 50, 75,100, 200]) * source.rms
    print(levels)
    
    fontsize = 18
    if suffix != "":
        suffix = "_" + suffix
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    scale = cosmo.kpc_proper_per_arcmin(source.z)  
    
    if size is None:
        size = (1.5*u.Mpc / scale).to(source.pixScale).value

    
    max = restructure_fits(source.path, size = size)
    
    gc = aplpy.FITSFigure('test.fits', subplot=[0., 0., .72, .8])

    gc.show_colorscale(cmap='inferno', vmin=vmin, vmax=vmax*max, stretch='log', vmid=1.5*vmin)
    gc.show_contour('test.fits', colors=annotation_color, levels=levels, alpha=0.7, linewidths=1) # type: ignore

    gc.add_beam(color="none", edgecolor=annotation_color)
    gc.add_label(0.495, 0.94, source.name, relative=True, size=25, color=annotation_color)
    
    gc.add_colorbar()
    gc.colorbar.set_axis_label_text("Jy beam$^{-1}$")
    gc.colorbar.set_axis_label_font(size=fontsize)
    gc.colorbar.set_font(size=fontsize)
    
    display_scale = 10.0
    #gc.add_scalebar((display_scale*u.kpc/scale).to(u.deg).value , color=annotation_color, corner="bottom")
    #gc.scalebar.set_label(f'{int(display_scale)} kpc')
    #gc.scalebar.set_font(size=fontsize)

    gc.tick_labels.set_font(size=fontsize)
    gc.axis_labels.set_font(size=fontsize)
    try:
        gc.save(f"{source.name}{suffix}.png", dpi=300)
    except:
        gc.save(f"{source.name}{suffix}.png", dpi=150)
    
    os.remove('test.fits')
    
if __name__ == "__main__":
    DATA_DIR = "/data/data/3Csurvey/tgts/"
    
    catalog = Catalogue3C(["3c231"])
    for source in catalog:
        source.set_data(f"{DATA_DIR}{source.name}/img/{source.name}-img-final-MFS-image.fits")
        #source.set_data(f"{DATA_DIR}{source.name}/backup-manual/img/img-all-05-MFS-image.fits")
        size = source_angular_size(source.name, source.path, threshold=5) * u.arcmin

        get_integrated_flux(source, threshold=9)
        #plot_galaxy(source, vmax=0.005, vmin=-5*source.rms.value, size=2000)
        plot_galaxy(source, vmax=1.1, vmin=-6.*source.rms.value, size=150)
        #plot_optical(source)
