import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Function to calculate beam sensitivity image
def calculate_beam_sensitivity(element_0_path, element_15_path):
    # Load element 0 and element 15 images
    element_0_data = fits.getdata(element_0_path)[0, 0, :, :]  # Extracting slices along x and y axes
    element_15_data = fits.getdata(element_15_path)[0, 0, :, :]  # Extracting slices along x and y axes

    # Calculate beam sensitivity image
    beam_sensitivity = 0.5 * element_0_data + 0.5 * element_15_data

    return beam_sensitivity

# Function to write beam sensitivity image to a new .fits file
def write_beam_sensitivity_to_fits(beam_sensitivity, output_path):
    hdu = fits.PrimaryHDU(beam_sensitivity)
    hdu.writeto(output_path, overwrite=True)

# Example usage
if __name__ == "__main__":
    # Paths to element 0 and element 15 .fits files
    element_0_path = "img/img-core-01-MFS-beam-0.fits"
    element_15_path = "img/img-core-01-MFS-beam-15.fits"

    # Calculate beam sensitivity image
    beam_sensitivity = calculate_beam_sensitivity(element_0_path, element_15_path)

    # Write beam sensitivity image to a new .fits file
    output_path = "img/beam_sensitivity.fits"
    write_beam_sensitivity_to_fits(beam_sensitivity, output_path)
    print(f"Beam sensitivity image saved to {output_path}")