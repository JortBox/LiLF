#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, sys, os, pickle
from astropy.io import fits
import numpy as np
import json
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.units.quantity import Quantity

from astroquery.vizier import Vizier 
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
from astropy.table import Table as AstroTab

sys.path.append("/data/scripts/LiLF")
from LiLF_lib import lib_util
import LiLF_lib.lib_img as lilf


Vizier.ROW_LIMIT = -1

def indices_to_slices(input):
    iteration = iter(input)
    start = next(iteration)
    slices = []
    for i, x in enumerate(iteration):
        if x - input[i] != 1:
            end = input[i]
            if start == end:
                slices.append([start])
            else:
                slices.append([start, end])
            start = x
    if input[-1] == start:
        slices.append([start])
    else:
        slices.append([start, input[-1]])
        
    return slices


class Flux(object):
    def __init__(self, flux: Quantity, freq: Quantity = 57.7*u.MHz, error: Quantity = 0.*u.Jy, spectral_index = -1.0, manual: bool = True, refcode: str = "NA"):
        self.flux = flux
        self.error = error
        self.freq = freq
        self.spectral_index = spectral_index
        self.is_manual = manual
        self.corrected = False
        self.refcode = refcode
        
        self.correct_flux(refcode)
        
        
    def __str__(self) -> str:
        return f"{self.flux} at {self.freq}"
    
    def to_frequency(self, target_freq: Quantity, spectral_index =  None):
        if spectral_index is None:
            spectral_index = self.spectral_index
            
        self.flux *= ((target_freq/self.freq).decompose())**spectral_index
        self.freq = target_freq
        #TODO: new error calc
        return self
    
    def correct_flux(self, refcode: str):
        with open("/local/work/j.boxelaar/scripts/LiLF/parsets/LOFAR_3c_core/flux_corrections.json") as file:
            corrections = json.load(file)["data"]
        
        for correction in corrections:
            if refcode == correction["refcode"] or self.freq.value == correction["freq"]:
                self.flux *= correction["correction"]
                self.corrected = True
                break
        
        del corrections
        
    
    
class SED(object):
    def __init__(self, target: str = "NA"):
        self.fluxes: list[Flux] = list()
        self.fluxes_dict = dict()
        self.target = target
    
    def __getitem__(self, item):# -> Flux|list[Flux]:
        if type(item) == str:
            index = self.fluxes_dict[item]
            if type(index) is int:
                return self.fluxes[index]
            else:
                return self.fluxes[index[-1]]
            
        elif type(item) == int or type(item) == slice:
            return self.fluxes[item]
        else:
            raise IndexError
        
    def __str__(self) -> str:
        string_to_print = f"SED ({self.target}): \n"
        for flux in self.fluxes:
            string_to_print += f"{flux.flux} at {flux.freq} \n"
        return string_to_print
        
        
    def __len__(self) -> int:
        return len(self.fluxes)
        
    def insert(self, *args, **kwargs):
        new_flux = Flux(*args, **kwargs)
        new_key = f"{int(np.round(new_flux.freq.value))}{new_flux.freq.unit}"
        if new_key in self.fluxes_dict.keys():
            if type(self.fluxes_dict[new_key]) is list:
                indices = self.fluxes_dict[new_key]
            else:
                indices = [self.fluxes_dict[new_key]]
            
            indices.append(len(self))
        else:
            indices = len(self)
            

        self.fluxes.append(new_flux)
        self.fluxes_dict.update({new_key: indices})
        
    def plot(self, show: bool = True, *args, **kwargs):
        plt.errorbar(self.freq[self.is_manual == False], self.flux[self.is_manual == False], yerr=self.error[self.is_manual == False], *args, **kwargs)
        plt.semilogy()
        plt.semilogx()
        if show:
            plt.show()
    
    @property 
    def flux(self):
        return np.asarray([flux.flux.to(u.Jy).value for flux in self.fluxes]) * u.Jy
    
    @property
    def freq(self):
        return np.asarray([flux.freq.to(u.MHz).value for flux in self.fluxes]) * u.MHz
    
    @property
    def error(self):
        return np.asarray([flux.error.to(u.Jy).value for flux in self.fluxes]) * u.Jy
    
    @property
    def is_manual(self):
        return np.asarray([flux.is_manual for flux in self.fluxes])
    
    @property
    def refcode(self):
        return np.asarray([flux.refcode for flux in self.fluxes])
    
    
    

class Source3C(object):
    def __init__(self, name : str):
        self.name = name
        self.display_name = name.replace("3c", "3C ")
        self.z = 0
        self.SED = SED(name)
        self._default_rms = False
        self.data_set = False
        self.dynamic_range = 0
        
    def set_redshift(self, z: float):
        self.z = z
        
    def set_coord(self, coord: SkyCoord):
        self.coord = coord
        
    def set_diameter(self, diameter: float):
        self.diameter = diameter
    
    #@DeprecationWarning    
    #def add_flux(self, *args):
    #    self.SED.insert(Flux(*args))
        
    def set_data(self, path: str):
        hdu: fits.PrimaryHDU = fits.open(path)[0] # type: ignore

        beam_area = get_beam_area(hdu)
        self.beam = u.def_unit("beam", beam_area)
        self.pixScale = u.def_unit("pixel", abs(hdu.header['CDELT1'] * u.deg)) # type: ignore
        self.pix = u.def_unit("pixel", abs(hdu.header['CDELT1'] * hdu.header['CDELT2']) * u.deg * u.deg) # type: ignore
        
        self.data: Quantity = np.asarray(hdu.data, dtype=np.float64)[0,0,:,:] * u.Jy/self.beam
        self.header = hdu.header
        self.path = path
        self.data_set = True
        
        self.dynamic_range = np.max(self.data.value) / np.min(self.data.value)
        
    def clear_data(self):
        self.data_set = False
        del self.data
        
    def clear_sed(self):
        self.SED = SED(self.name)
    
    @property
    def rms(self) -> Quantity:
        if (not self.data_set) and (not self._default_rms):
            return 0. * u.Jy
        
        if not self._default_rms:
            self.rms_value = get_rms(self.data.value) * u.Jy/self.beam
            self._default_rms = True
            return self.rms_value
        else:
            return self.rms_value
            


class Catalogue3C(object):
    def __init__(self, targets: list[str], suffix: str = ""):
        self._counter = 0
        self.targets = targets 
        if suffix == "":
            self.suffix = suffix
        else:
            self.suffix = "_" + suffix
            
        self.__query_objects_ned(targets)
        self.__query_objects_vizier(targets)
        #self.save()
        
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, item) -> Source3C:
        if type(item) == str:
            index = self.sources_dict[item]
            return self.sources[index]
        elif type(item) == int or type(item) == slice:
            return self.sources[item]
        else:
            raise IndexError
            
    def __iter__(self):
        return self
        
    def __next__(self) -> Source3C:
        if self._counter < len(self):
            self._counter += 1
            return self.sources[self._counter-1]
        
        else:
            self._counter = 0
            raise StopIteration
    
     
    @property
    def coord(self) -> SkyCoord:
        xcoord = np.asarray([source.coord.ra.deg for source in self.sources])*u.deg # type: ignore
        ycoord = np.asarray([source.coord.dec.deg for source in self.sources])*u.deg # type: ignore
        return SkyCoord(ra=xcoord, dec=ycoord, unit=(u.deg, u.deg), frame='fk4')
    
    @property
    def z(self) -> np.ndarray:
        return np.array([source.z for source in self.sources])
    
    @property
    def flux(self) -> np.ndarray:
        return np.array([source.flux for source in self.sources])
    
    @property
    def diameter(self) -> np.ndarray:
        return np.array([source.diameter for source in self.sources])
    
    @property
    def name(self) -> list[str]:
        return [source.name for source in self.sources]
    
    @property
    def display_name(self) -> list[str]:
        return [source.display_name for source in self.sources]
    

    def __query_objects_vizier(self, targets: list[str]) -> None:
        targets = np.asarray(targets) # type: ignore
        catalogs = Vizier.find_catalogs("3cr")
        table = Vizier.get_catalogs(catalogs.keys())[1] # type: ignore
        
        for entry in table:
            name = "3c" + entry["_3CR"]
            if name == "3c225":
                name += "b"
            if name in self.targets:
                if entry["x_Diam"] == '"':
                    self[name].set_diameter((entry["Diam"] * u.arcsec).to(u.arcmin))
                else:
                    self[name].set_diameter(entry["Diam"] * u.arcsec)
                
                self[name].SED.insert(entry["S178MHz"] * u.Jy, freq=178.*u.MHz)
                
        
    def __query_objects_ned(self, targets: list[str]) -> None:
        self.sources_dict = dict()
        self.sources = list()
        
        for i, target in enumerate(targets):
            print(f"querying target {i+1} of {len(self)}")
            table = Ned.query_object(target, verbose = False)
            source = Source3C(target)
                
            try: source.set_redshift(float(table["Redshift"])) # type: ignore
            except: source.set_redshift(0)
            
            try:
                source.set_coord(
                    SkyCoord(
                        ra = float(table["RA"]),  # type: ignore
                        dec = float(table["DEC"]),  # type: ignore
                        unit = (u.deg, u.deg), 
                        frame = 'fk4'
                    )
                )
            except:
                source.set_coord(
                    SkyCoord(ra=0.0, dec=0.0, unit=(u.deg, u.deg), frame='fk4') # type: ignore
                )
                
            self.sources_dict.update({target: i})
            self.sources.append(source)
                
    def save(self):
        with open(f"Catalogue3C{self.suffix}.pyobj", "wb") as file:
            pickle.dump(self, file)




def query_fluxes_ned(source: Source3C, freq_limit = 1.e10 * u.Hz):
    table = Ned.get_table(source.name, table="photometry", sort_by="Frequency")
    
    for item in table:
        flux = item["Flux Density"] # type: ignore
        freq = item["Frequency"] * u.Hz # type: ignore
        error = (item["Lower limit of uncertainty"] + item["Upper limit of uncertainty"])/2 # type: ignore
        
        if str(flux) == '--':
                continue
        
        if freq < freq_limit:
            if str(error) == "--":
                error = 0
                
            if item["Units"] == "milliJy": # type: ignore
                flux *= u.Jy
                error *= u.Jy
                source.SED.insert(flux, freq.to(u.MHz), error, manual=False, refcode=item["Refcode"])
                
                
            elif item["Units"] == "W m^-2^ Hz^-1^": # type: ignore
                continue
            else:
                try:
                    flux *= u.Unit(item["Units"]) # type: ignore
                    error *= u.Unit(item["Units"]) # type: ignore
                    source.SED.insert(flux.to(u.Jy), freq.to(u.MHz), error, manual=False, refcode=item["Refcode"])
                except ValueError:
                    print(f"Error: {item['Units']} not recognized")
                    
            
            
            
            
            



#DATA_DIR = "/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/"

def measure_flux(path: str, region=None, threshold: int=9, use_cache=True) -> float:
    if not use_cache:
        lib_util.check_rm("/".join(path.split("/")[:-1]) + "/*ddcal*")
        
    # load skymodel
    full_image = lilf.Image(path, userReg=region) # -MFS-image.fits
    mask_ddcal = full_image.imagename.replace(".fits", "_mask-ddcal.fits")  # this is used to find calibrators
    
    full_image.makeMask(
        mode="default",
        threshpix=int(threshold),
        atrous_do=False,
        maskname=mask_ddcal,
        write_srl=True,
        write_ds9=True,
    )
    
    cal = AstroTab.read(mask_ddcal.replace("fits", "cat.fits"), format="fits")
    cal = cal[np.where(cal["Total_flux"] > 10)]
    cal.sort(["Total_flux"], reverse=True)

    return cal["Total_flux"][0]

def get_rms(data, niter=100, maskSup=1e-7):
    m      = data[np.abs(data)>maskSup]
    rmsold = np.std(m)
    diff   = 1e-1
    cut    = 3.
    bins   = np.arange(np.min(m),np.max(m),(np.max(m)-np.min(m))/30.)
    med    = np.median(m)

    for i in range(niter):
        ind = np.where(np.abs(m-med)<rmsold*cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: 
            break
        rmsold = rms
    return rms

def get_beam_area(hdu: fits.PrimaryHDU):
    try:
        bmaj = hdu.header['BMIN']*u.deg
        bmin = hdu.header['BMAJ']*u.deg
        bpa = hdu.header['BPA']*u.deg
    except KeyError:
        print("No Beam info")
        sys.exit()

    conversion = 1. / (2.*(2.*np.log(2.))**0.5) # Convert to sigma
    beammaj = bmaj * conversion
    beammin = bmin * conversion
    #pix_area = abs(hdu.header['CDELT1'] * hdu.header['CDELT2']) * u.deg * u.deg # type: ignore
    #beam2pix = beam_area / pix_area
    return 2.0 * np.pi * beammaj * beammin

def get_flux_from_model(path: str, radius: int = 10):
    model = fits.open(path)[0].data # type: ignore
    cpix = model.shape[-1]//2
    
    model_cutout = model[0,0,cpix-radius:cpix+radius, cpix-radius:cpix+radius]
    return np.sum(model_cutout)


def get_integrated_flux(source: Source3C, size: int = 100, threshold: float = 9, plot: bool = False):
    size //= 2
    nax = np.asarray(source.data.shape)//2 # type: ignore
    data = source.data[nax[0]-size:nax[0]+size, nax[1]-size:nax[1]+size] # type: ignore
    
    
    image_center = np.asarray(data.shape)//2
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    r = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
    
    flux = np.zeros(data.shape) * u.Jy/source.beam
    flux[data > threshold * source.rms] = data[data > threshold * source.rms]
    #flux[r > 15] = 0

    total_flux = np.sum(flux.to(u.Jy/source.pix)) * 1. * source.pix 
    print(source.rms.to(u.Jy/source.pix) * flux[flux>0].shape[0] * source.pix)
    print(f"flux: {total_flux:.2f} " )
    
        
    if source.name == '3c147':
        a0 = 66.738
        a1 = -0.022
        a2 = -1.012 
        a3 = 0.549
        S = a0 * 10**(a1 * np.log10(57.7/150)) * 10**(a2 * np.log10(57.7/150)**2) * 10**(a3 * np.log10(57.7/150)**3)
        print("scaife flux",S)
        print("fraction",total_flux/S)
        
    elif source.name == '3c48':
        a0 = 64.768
        a1 = -0.387 
        a2 = -0.420
        a3 = 0.181 
        S = a0 * 10**(a1 * np.log10(57.7/150)) * 10**(a2 * np.log10(57.7/150)**2) * 10**(a3 * np.log10(57.7/150)**3) * u.Jy
        print("scaife flux",S)
        print("fraction",total_flux/S)
        
    elif source.name == '3c196':
        a0 = 83.084
        a1 = -0.699
        a2 = -0.110
        S = a0 * 10**(a1 * np.log10(57.7/150)) * 10**(a2 * np.log10(57.7/150)**2) * u.Jy
        print("scaife flux",S)
        print("fraction",total_flux/S)
        
    if plot:
        print("plot")
        plt.imshow(np.log10(flux.value))
        plt.contour(r)
        plt.savefig("flux.png", dpi=300)
        plt.clf()
        
    return total_flux.value
        
        


if __name__ == "__main__":
    DATA_DIR = "/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts"
    DATA_DIR = "/Users/jortboxelaar/Documents/PhDLocal/FITS"
    
    catalog = Catalogue3C(["3c48"])
    for source in catalog:
        #query_fluxes_ned(source)
        #print(source.SED.freq)
        
        #source.set_data(f"{DATA_DIR}{source.name}/img/img-clean-test-MFS-image.fits")
        
        #source.set_data("/Users/jortboxelaar/Documents/PhDLocal/FITS/3c196/img/img-core-01-MFS-image.fits")
        #for i in range(1,5):
        
        source.set_data(f"{DATA_DIR}/3c48/img_old/img-core-01-MFS-image.fits")
        get_integrated_flux(source, size=50, plot=False)
        
            
        plt.errorbar(source.SED.freq, source.SED.flux, yerr=source.SED.error, fmt=".", alpha=0.4, color='black')
        for i in range(1,5):
            source.SED.insert(get_flux_from_model(f"{DATA_DIR}/3c48/img_old/img-core-0{i}-MFS-model.fits", radius=14) * u.Jy)
            #print(source.SED["58MHz"])
            plt.errorbar(source.SED["58MHz"].freq, source.SED["58MHz"].flux, yerr=0.2*source.SED["58MHz"].flux, fmt=".", alpha=0.4, color='red')
        plt.semilogy()
        plt.semilogx()
        plt.show()
        
        #print(source.SED["58MHz"])
        print(source.SED)
        #source.SED.insert(measure_flux(source.path, threshold=9)*u.Jy)
        #source.SED.insert(get_integrated_flux(source, threshold=9, plot=False)*u.Jy)
        #print(source.rms)
        #source.clear_data()
        #get_integrated_flux(source, threshold=9, plot=True)
        
        #print(source.SED)
        #print(source.SED["178MHz"])
        #plot_galaxy(source, vmax=0.75*np.max(source.data.value), vmin=-5*source.rms.value)
        