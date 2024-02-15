#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import astropy.units as u
import astropy.constants as const
import numpy as np

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
    parset_dir = "/net/voorrijn/data2/boxelaar/scripts/LiLF/parsets/LOFAR_3c_core/"
    
    with open(parset_dir + 'station-diameter.json') as file:
        data = json.load(file)["data"]
        
    max_diameter = maximum_station_diameter(source_angular_diameter, central_freq)

    for idx, item in enumerate(data):
        if idx == len(data) - 1:
            return item["stations"]
        
        elif item["diameter"] <= max_diameter and data[idx+1]["diameter"] > max_diameter :
            return item["stations"]
            
    return ""
