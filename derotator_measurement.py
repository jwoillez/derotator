#!/usr/bin/env python

from argparse import ArgumentParser
import glob
import re
import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.table import Table


def generate_coordinates(image):
    return np.array(np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])))


def find_maximum(image, coord):
    i = np.argmax(image)
    return [coord[0].flatten()[i], coord[1].flatten()[i]]


def extract_subimage(image, coord, center, size):
    image_out = image[center[1]-size[1]//2:center[1]+size[1]//2, center[0]-size[0]//2:center[0]+size[0]//2]
    coord_out = coord[:, center[1]-size[1]//2:center[1]+size[1]//2, center[0]-size[0]//2:center[0]+size[0]//2]
    return image_out, coord_out


def measure_barycenter(image, coord):
    return np.sum(image[None,:,:]*coord, axis=(1,2))/np.sum(image)


def threshold_image(image):
    return image - np.min(image)


def measure_centroid(filename, ax=None):
    with fits.open(filename) as hdulist:
        image = hdulist[0].data
    coord = generate_coordinates(image)
    center = find_maximum(image, coord)
    subimage, subcoord = extract_subimage(image, coord, center, [40,40])
    subimage = threshold_image(subimage)
    barycenter = measure_barycenter(subimage, subcoord)
    if ax:
        ax.imshow(image)
        ax.plot(barycenter[0], barycenter[1], '.k', ms=10)
        ax.set_xlim(np.min(subcoord[0]), np.max(subcoord[0]))
        ax.set_ylim(np.min(subcoord[1]), np.max(subcoord[1]))
    return barycenter


def measure_derotator(input_root, plot=True):
    # Identify the files to process, extract te rotator angle, store in a table
    filenames = [filename for filename in glob.glob(input_root+"_*.fits") if re.match(input_root+"_[0-9]+.fits", filename)]
    angles = [float(re.search("_([0-9]+).fits", filename).group(1)) for filename in filenames]
    table = Table()
    table['angle'] = angles
    table['x'] = np.zeros(len(angles), dtype=np.float)
    table['y'] = np.zeros(len(angles), dtype=np.float)
    table['filename'] = filenames
    table.sort('angle')
    # Prepare plot if needed
    if plot:
        fig, axarr = plt.subplots(1,len(filenames), figsize=(2*len(filenames),3))
        fig.suptitle(input_root)
    # Measure centroids of identified fits files
    for i in range(len(table)):
        barycenter = measure_centroid(table['filename'][i], ax=(axarr[i] if plot else None))
        if plot:
            axarr[i].set_title("{0} deg".format(table['angle'][i]))
        angles.append(table['angle'][i])
        table['x'][i] = barycenter[0]
        table['y'][i] = barycenter[1]
    # Save result
    output_file = input_root+".txt"
    table.write(output_file, format="ascii.fixed_width_two_line")
    print("Saved {0}...".format(output_file))
    print(table)


if __name__ == '__main__':
    description = \
        """
        Measures centroids from a set of fits files with format 'root_<angle>.fits',
        where <angle> is the derotator angle of each fits file,
        and 'root' is the specified root file.

        The results are stored in an astropy table named 'root.txt'.
        """
    parser = ArgumentParser(description=description)
    parser.add_argument("-p", "--plot",
                        action="store_true", dest="plot", default=False,
                        help="plot centroiding diagnostics")
    parser.add_argument("root", help="root name of the files to process")
    args = parser.parse_args()
    measure_derotator(args.root, plot=args.plot)
    plt.show()
