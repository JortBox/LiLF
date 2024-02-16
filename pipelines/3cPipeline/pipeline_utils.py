#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.io import fits

from LiLF_lib.lib_ms import AllMSs as MeasurementSets

PARSET_DIR = "/net/voorrijn/data2/boxelaar/scripts/LiLF/parsets/LOFAR_3c_core/"

def maximum_station_diameter(source_angular_diameter: float, central_freq: float, amp_fraction: float = 0.95, alpha1: float = 1.3) -> float:
    """Calculate maximum station diameter after phasing up stations

    Args:
        source_angular_size (float): angular size in arcmin
        central_freq (float): in MHz
        amp_fraction (float, optional): maximum relative loss of flux at edge of the source. Defaults to 0.95.
        alpha1 (float, optional): LOFAR alpha_1. Defaults to 1.3.

    Returns:
        float: maximum station diameter in meters
    """
    central_freq *= u.MHz
    source_angular_diameter = (source_angular_diameter * u.arcmin).to(u.rad)
    factor = alpha1 * (const.c / central_freq) * (- np.log(amp_fraction) / np.log(2))**0.5 # type: ignore

    return (factor / source_angular_diameter).to(u.m / u.rad).value


def stations_to_phaseup(source_angular_diameter: float, central_freq: float) -> str:
    with open(PARSET_DIR + 'station_diameter.json') as file:
        data = json.load(file)["data"]
        
    max_diameter = maximum_station_diameter(source_angular_diameter, central_freq)

    for idx, item in enumerate(data):
        if idx == len(data) - 1:
            return item["stations"]
        
        elif item["diameter"] <= max_diameter and data[idx+1]["diameter"] > max_diameter :
            return item["stations"]
            
    return ""

def source_angular_size(source: str, img_path: str = "", threshold: float = 0.25) -> float:
    if img_path == "":
        img_path = f"/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/{source}/img/img-all-02-MFS-image.fits"
        
    with open(PARSET_DIR + "source_angular_diameter.json") as file:
        data = json.load(file)["data"]
        
    for item in data:
        # Check if diameter was already calculated
        if item["name"] == source:
            return item["diameter"] 

    with fits.open(img_path) as hdul:
        data = hdul[0].data[0,0] # type: ignore
        header = hdul[0].header # type: ignore
    

    image_center = np.asarray(data.shape)//2
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    x, y = np.meshgrid(x, y)
    r = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
    
    annuli = np.linspace(0, image_center[0], 300)
    flux = np.zeros(annuli.shape[0])
    
    mean = 0
    for i in range(1, len(annuli)):
        flux[i] = np.sum( data[(r <= annuli[i]) & (r > annuli[i-1])] )
        
        if i == 0:
            mean = flux[i]
            continue 
        
        if flux[i] < threshold * mean:
            pix_extent = annuli[i+1]
            break

        mean = np.sum(flux[:i])/(i+1)
    
    angular_arcmin_diam = (abs(header["CDELT1"]) * 2 * pix_extent) * 60
    
    with open(PARSET_DIR + "source_angular_diameter.json") as file:
        json_data = json.load(file)
    
    with open(PARSET_DIR + "source_angular_diameter.json", "w") as file:
        json_data["data"].append(dict(name=source, diameter=angular_arcmin_diam))
        json.dump(json_data, file)
        
    return angular_arcmin_diam