#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, sys, os, pickle
from astropy.io import fits
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.units.quantity import Quantity

from astroquery.vizier import Vizier 
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
from astropy.table import Table as AstroTab

import LiLF_lib.lib_img as lilf


Vizier.ROW_LIMIT = -1

class Source3C(object):
    def __init__(self, name : str):
        self.name = name
        self.z = 0
        self._default_rms = False
        self.data_set = False
        
    def set_redshift(self, z: float):
        self.z = z
        
    def set_coord(self, coord: SkyCoord):
        self.coord = coord
        
    def set_diameter(self, diameter: float):
        self.diametr = diameter
        
    def set_flux(self, flux: float):
        self.flux = flux
        
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
    
    @property
    def rms(self) -> Quantity:
        if not self.data_set:
            return 0. * u.Jy/self.beam
        
        if not self._default_rms:
            return get_rms(self.data.value) * u.Jy/self.beam
        else:
            self._default_rms = True
            return self.rms
            


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
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, item: str | int) -> Source3C:
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
        return np.array([source.diameter for source in self.sources ])
    

    def __query_objects_vizier(self, targets: list[str]) -> None:
        targets = np.asarray(targets) # type: ignore
        catalogs = Vizier.find_catalogs("3cr")
        table = Vizier.get_catalogs(catalogs.keys())[1] # type: ignore
        
        for entry in table:
            name = "3c" + entry["_3CR"]
            if name in self.targets:
                if entry["x_Diam"] == '"':
                    self[name].set_diameter((entry["Diam"] * u.arcsec).to(u.arcmin))
                else:
                    self[name].set_diameter(entry["Diam"] * u.arcsec)
                
                self[name].set_flux(entry["S178MHz"] * u.Jy)
                
        
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








DATA_DIR = "/net/voorrijn/data2/boxelaar/data/3Csurvey/tgts/"

def measure_flux(path: str, region=None) -> float:
    # load skymodel
    full_image = lilf.Image(path, userReg=region) # -MFS-image.fits
    mask_ddcal = full_image.imagename.replace(".fits", "_mask-ddcal.fits")  # this is used to find calibrators
    
    full_image.makeMask(
        mode="default",
        threshpix=5,
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


def get_integrated_flux(source: Source3C, size: int = 100, threshold: float = 10, plot: bool = True):
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
        print(S)
        print(total_flux/S)
        
    elif source.name == '3c48':
        a0 = 64.768
        a1 = -0.387 
        a2 = -0.420
        a3 = 0.181 
        S = a0 * 10**(a1 * np.log10(57.7/150)) * 10**(a2 * np.log10(57.7/150)**2) * 10**(a3 * np.log10(57.7/150)**3) * u.Jy
        print("scaife flux",S)
        print("fraction",total_flux/S)
        
    if plot:
        plt.imshow(np.log10(flux.value))
        plt.contour(r)
        plt.savefig("flux.png", dpi=300)
        plt.clf()
    
'''
def get_total_flux(file: str, path: str = DATA_DIR, size: int = 100, threshold: float = 5, plot:bool=True):
    size //= 2
    hdu: fits.PrimaryHDU = fits.open(path + file)[0] # type: ignore
    
    nax = hdu.header["NAXIS1"]//2 # type: ignore
    beam_area = get_beam_area(hdu)
    u_beam = u.def_unit("beam", beam_area)
    u_pix = u.def_unit("pixel", abs(hdu.header['CDELT1'] * hdu.header['CDELT2']) * u.deg * u.deg) # type: ignore
    
    rms = get_rms(hdu.data[0,0,:,:]) * u.Jy/u_beam # type: ignore
    data: Quantity = hdu.data[0,0, nax-size:nax+size, nax-size:nax+size] * u.Jy/u_beam # type: ignore
    
    
    image_center = np.asarray(data.shape)//2
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    r = np.sqrt((x - image_center[0])**2 + (y - image_center[1])**2)
    
    flux = np.zeros(data.shape) * u.Jy/u_beam
    flux[data > threshold * rms] = data[data > threshold * rms]
    #flux[r > 11] = 0
    
    
    print(rms.to(u.Jy/u_pix) * flux[flux>0].shape[0] * u_pix)
    
    total_flux = np.sum(flux.to(u.Jy/u_pix)) * 1. * u_pix 
    print(f"flux: {total_flux:.2f} " )
    
    if plot:
        plt.imshow(np.log10(flux.value))
        plt.contour(r)
        plt.savefig("flux.png", dpi=300)
        plt.clf()
        
    # slux scale for 3c147 (scaife et al 2012)
    a0 = 66.738
    a1 = -0.022
    a2 = -1.012 
    a3 = 0.549
    S = a0 * 10**(a1 * np.log10(57.7/150)) * 10**(a2 * np.log10(57.7/150)**2) * 10**(a3 * np.log10(57.7/150)**3)
    print(S)
'''  
    
'''   
def rename_final_images(files: list[str]):
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
        new_path = path + "/img-final-MFS-" + suffix
        os.system(f"cp {file} {new_path}")   
'''  

if __name__ == "__main__":
    target = "3c48"
    img_path = DATA_DIR + target + "/img/"
    #rename_final_images(glob.glob(f"{DATA_DIR}{target}/img/img-all-*"))
        
    #sys.exit()
    