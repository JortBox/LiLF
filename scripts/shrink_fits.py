#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from astropy.io import fits

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('path_in', help='(str) FITS image location (containing radio halo).', type=str)
    parser.add_argument('-o','--path_out', help='(str) FITS image location (containing radio halo).', type=str, default='')
    parser.add_argument('-s', '--size', help='(int) Size of the image to be extracted.', type=int, default=200)
    parser.add_argument('-c','--centre_pix', type=int, nargs='+', help='centre', default=[0,0])
    parser.add_argument('--delete_freq', dest='delete_freq', action='store_true', default=False)
    return parser.parse_args()

def restructure_fits(path_in: str, path_out: str, size: int, centre_pix: list, delete_freq: bool) -> None:
    oldhdul = fits.open(path_in)
    oldhdu: fits.PrimaryHDU = oldhdul[0] # type: ignore
    
    size = int(np.round(size/2))
    if centre_pix == [0,0]:
        nax = np.asarray(oldhdu.data.shape[2:])//2 # type: ignore
    else:
        nax = np.asarray(centre_pix)

    if delete_freq:
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
    
    if size == -1:
        hdu = fits.PrimaryHDU(oldhdu.data[0,0], header=oldhdu.header) # type: ignore
        hdu.writeto(path_out, overwrite=True)
        return np.max(hdu.data)
    
    oldhdu.header["CRPIX1"] -= nax[0]-size 
    oldhdu.header["CRPIX2"] -= nax[1]-size
    
    if delete_freq:
        cropped_data = oldhdu.data[0, 0, nax[0]-size:nax[0]+size, nax[1]-size:nax[1]+size]
    else:
        cropped_data = oldhdu.data[:, :, nax[0]-size:nax[0]+size, nax[1]-size:nax[1]+size]
    
    hdu = fits.PrimaryHDU(cropped_data, header=oldhdu.header)
    hdu.writeto(path_out, overwrite=True)

if __name__ == "__main__":
    args = arguments()
    filename = args.path_in.split('/')[-1]

    if args.path_out == '':
        args.path_out = '/'.join(args.path_in.split('/')[:-1]) + '/cropped_' + filename
        
    restructure_fits(args.path_in, args.path_out, args.size, args.centre_pix, args.delete_freq)