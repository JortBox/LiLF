#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, glob, sys, os

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.io import fits

sys.path.append("/data/scripts/LiLF")
from LiLF_lib.lib_ms import AllMSs as MeasurementSets

PARSET_DIR = "/data/scripts/LiLF/parsets/LOFAR_3c_core/"

RSISlist = [
    'RS106LBA','RS205LBA','RS208LBA','RS210LBA','RS305LBA','RS306LBA',
    'RS307LBA','RS310LBA','RS406LBA','RS407LBA','RS409LBA','RS503LBA',
    'RS508LBA','RS509LBA','DE601LBA','DE602LBA','DE603LBA','DE604LBA',
    'DE605LBA','DE609LBA','FR606LBA','SE607LBA','UK608LBA','PL610LBA',
    'PL611LBA','PL612LBA','IE613LBA','LV614LBA'
]

CS_list = [
    'CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA',
    'CS007LBA','CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA',
    'CS026LBA','CS028LBA','CS030LBA','CS031LBA','CS032LBA','CS101LBA',
    'CS103LBA','CS201LBA','CS301LBA','CS302LBA','CS401LBA','CS501LBA',
]

def get_cal_dir(timestamp: str, logger = None) -> list:
    """
    Get the proper cal directory from a timestamp
    """
    
    dirs = list()
    for cal_dir in sorted(glob.glob('../../cals/3c*')):
        calibrator = cal_dir.split("/")[-1]
        cal_timestamps = set()
        for ms in glob.glob(cal_dir+'/20*/data-bkp/*MS'):
            cal_timestamps.add("_".join(ms.split("/")[-1].split("_")[:2]))
            
        if f"{calibrator}_t{timestamp}" in cal_timestamps:
            if logger is not None:
                logger.info('Calibrator found: %s (t=%s)' % (cal_dir, timestamp))
            dirs.append(f"{cal_dir}/{timestamp[:8]}/solutions")
        else:
            pass
        
    if dirs == []:
        if logger is not None:
            logger.error('Missing calibrator.')
        sys.exit()
    
    return dirs

def rename_final_images(files: list[str], target: str = ""):
    path = '/'.join(files[0].split('/')[:-1])
    for image in reversed(files):
        try: 
            cycle = int(image.split('-')[-3])
            break
        except: 
            print(image.split('-'), "cannot be converted to float")
            continue
        
    for file in glob.glob(f'{path}/img-all-{cycle:02d}-MFS*'):
        suffix = file.split('-')[-1]
        new_path = f"{path}/{target}-img-final-MFS-{suffix}"
        os.system(f"cp {file} {new_path}")

def make_beam_region(MSs: MeasurementSets, target: str) -> tuple[str, str, str|None]:
    MSs.print_HAcov('plotHAelev.png')
    MSs.getListObj()[0].makeBeamReg('beam02.reg', freq='mid', pb_cut=0.2)
    beam02Reg = 'beam02.reg'
    MSs.getListObj()[0].makeBeamReg('beam07.reg', freq='mid', pb_cut=0.7)
    beam07reg = 'beam07.reg'

    region = f'{PARSET_DIR}/regions/{target}.reg'
    if not os.path.exists(region): 
        region = None
         
    return beam02Reg, beam07reg, region


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
    central_freq *= u.MHz # type: ignore
    source_angular_diameter = (source_angular_diameter * u.arcmin).to(u.rad) # type: ignore
    factor = alpha1 * (const.c / central_freq) * (- np.log(amp_fraction) / np.log(2))**0.5 # type: ignore

    return (factor / source_angular_diameter).to(u.m / u.rad).value


def stations_to_phaseup(source_angular_diameter: float, central_freq: float) -> str:
    with open(PARSET_DIR + 'station_diameter.json') as file:
        data = json.load(file)["data"]
        
    max_diameter = maximum_station_diameter(source_angular_diameter, central_freq)
    
    print("max_diameter", max_diameter)
    if max_diameter < data[0]["diameter"]:
        return ""

    for idx, item in enumerate(data):
        print("data", item["diameter"], item["stations"])
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
