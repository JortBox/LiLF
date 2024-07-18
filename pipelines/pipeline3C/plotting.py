#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, pickle, glob
import aplpy
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

import analysis as asis
from analysis import Source3C
from calibration import extended_targets, very_extended_targets

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
        return np.max(np.asarray(hdu.data))
    
    size = int(np.round(size/2))
    nax = np.asarray(oldhdu.data.shape[2:])//2 # type: ignore
    hdu = fits.PrimaryHDU(oldhdu.data[0,0,nax[0]-size:nax[0]+size,nax[1]-size:nax[1]+size], header=oldhdu.header) # type: ignore
    hdu.writeto('test.fits', overwrite=True)
    return np.max(np.asarray(hdu.data))


def plot_SED(source: Source3C):
    fontsize = 15
    fluxes = []
    #for i in range(1,5):
    #    source.set_data(f"{DATA_DIR}/{source.name}/img/img-core-0{i}-MFS-image.fits")
    #    fluxes.append(asis.measure_flux(source.path, threshold=9))
    #    source.SED.insert(fluxes[-1]*u.Jy)
    
    source.set_data(f"{DATA_DIR_HOME}/{source.name}/img/img-core-02-MFS-image.fits")
    source.SED.insert(asis.measure_flux(source.path, threshold=9)*u.Jy)
    
    asis.query_fluxes_ned(source)
    
    fig, ax = plt.subplots(figsize=(7,6))

    ax.errorbar(
        source.SED["58MHz"].freq, 
        source.SED["58MHz"].flux, 
        yerr=0.1*source.SED["58MHz"].flux, 
        color='red', 
        label="Measurement", 
        fmt="^",
    )

    spec = source.SED
    try:
        ax.errorbar(
            spec.freq[spec.refcode=="2007AJ....134.1245C"], 
            spec.flux[spec.refcode=="2007AJ....134.1245C"], 
            yerr=spec.error[spec.refcode=="2007AJ....134.1245C"], 
            color='dodgerblue', 
            label=f"VLA ({spec.freq[spec.refcode=='2007AJ....134.1245C'][0]})", 
            fmt="."
        )
    except:
        pass
    try: 
        ax.errorbar(
            spec.freq[spec.refcode=="2018AJ....155..188L"], 
            spec.flux[spec.refcode=="2018AJ....155..188L"], 
            yerr=spec.error[spec.refcode=="2018AJ....155..188L"], 
            color='black', 
            label=f"NVSS ({spec.freq[spec.refcode=='2018AJ....155..188L'][0]})", 
            fmt="."
        )
    except:
        pass
    source.SED.plot(
        show=False, fmt=".", label="NED", alpha=0.5, color='gray', zorder=0
    )

    ax.annotate(
        f"{source.display_name}", 
        (0.45, 0.93), 
        xycoords='axes fraction', 
        fontsize=fontsize
    )
    #ax.set_title(source.name)
    ax.set_xlabel("$\\nu$ [MHz]", fontsize=fontsize)
    ax.set_ylabel("Flux density [Jy]", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.legend(frameon=False, loc="lower left", fontsize=fontsize)
    plt.savefig("plots_sed/" + source.name + "_SED.png", dpi=300)
    plt.clf()
    plt.cla()
    #plt.show()


def plot_optical(source: Source3C):
    gc = aplpy.make_rgb_image(["/data/m82-infrared.fits","/data/m82-red.fits","/data/m82-blue.fits"], "m82-rgb.png", stretch_r='log', stretch_g='log', stretch_b='log')
    
def plot_galaxy(source: Source3C, suffix: str = "", vmin=None, vmax=None, size = None, annotation_color="white"):
    #levels = np.array([1.6**i for i in range(2,20)]) * source.rms
    if source.name in extended_targets:
        levels = np.array([5, 10, 50, 100, 200]) * source.rms
    elif source.name in very_extended_targets:
        levels = np.array([3, 5, 10, 50, 100, 200]) * source.rms
    else:
        levels = np.array([9, 50, 100, 200]) * source.rms
    #print(levels)
    
    fontsize = 18
    if suffix != "":
        suffix = "_" + suffix
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    scale = cosmo.kpc_proper_per_arcmin(source.z)
    
    if size is None:
        size = (1.5*u.Mpc / scale).to(source.pixScale).value

    max = restructure_fits(source.path, size = size)
    
    
    gc = aplpy.FITSFigure('test.fits', subplot=[0., 0., .72, .8])
    
    if vmin is None:
        vmin = -4.*source.rms.value
    if vmax is None:
        vmax = 1.

    gc.show_colorscale(cmap='inferno', vmin=vmin, vmax=vmax*max, stretch='log', vmid=5*vmin)
    #gc.show_contour('test.fits', colors=annotation_color, levels=levels, alpha=0.4, linewidths=1) # type: ignore

    gc.add_beam(color="none", edgecolor=annotation_color)
    gc.add_label(0.495, 0.94, source.name, relative=True, size=25, color=annotation_color)
    
    #gc.add_colorbar()
    #gc.colorbar.set_axis_label_text("Jy beam$^{-1}$")
    #gc.colorbar.set_axis_label_font(size=fontsize)
    #gc.colorbar.set_font(size=fontsize)
    axes = aplpy.AxisLabels(gc)
    axes.hide()
    
    display_scale = (size/4 * source.pixScale * scale).to(u.Mpc)
    if display_scale.value < 1:
        display_scale = display_scale.to(u.kpc).astype(np.int64)
        
    display_scale = display_scale.astype(np.int64)
        
    print(display_scale)
    gc.add_scalebar((display_scale/scale).to(u.deg).value , color=annotation_color, corner="bottom")
    gc.scalebar.set_label(f'{display_scale.value} {display_scale.unit}')
    gc.scalebar.set_font(size=fontsize)

    gc.tick_labels.set_font(size=fontsize)
    gc.axis_labels.set_font(size=fontsize)
    if not os.path.exists("plots_nc/"):
        os.mkdir("plots_nc/")
    gc.save(f"plots_nc/{source.name}{suffix}_no_contour.pdf")
    
    os.remove('test.fits')
    plt.clf()
    
if __name__ == "__main__":
    DATA_DIR_HOME = "/home/iranet/groups/lofar/j.boxelaar/products/3Csurvey"
    DATA_DIR_LOCAL = "/home/local/work/j.boxelaar/data/3Csurvey/tgts"

    
    all_targets = [target.split("/")[-1] for target in sorted(glob.glob(f"{DATA_DIR_HOME}/*"))]
    catalog = asis.Catalogue3C(all_targets, use_cache=False)
    #catalog = asis.Catalogue3C(["3c274"], use_cache=False)
    
    for source in catalog:
        paths = sorted(glob.glob(f"{DATA_DIR_LOCAL}/{source.name}/img/img-all-*-MFS-image.fits"))
        if len(paths) <= 1:
            paths = sorted(glob.glob(f"{DATA_DIR_HOME}/{source.name}/img/img-all-*-MFS-image.fits"))
            if len(paths) <= 1:
                print(f"Failed to plot {source.name}")
                continue
        
        path = paths[-2]   
        if source.name == "3c296":
            path = f"{DATA_DIR_HOME}/{source.name}/img/img-all-04-MFS-image.fits"
        
        if os.path.exists(path):
            source.set_data(path) 
        else:
            print(f"Failed to plot {source.name}. path does not exist")
            continue
        
        vmax=1
        if source.name in very_extended_targets or source.name in ["3c274", "3c236"]:
            size = int(source.data.shape[0]//2)
            vmax = 0.2
        elif source.name in extended_targets:
            size = 800
        else:
            size=300
            
        plot_galaxy(source, vmax=vmax, vmin=-4.*source.rms.value, size=size)
        
        #try: plot_SED(source)
        #except: print("Failed to plot SED")