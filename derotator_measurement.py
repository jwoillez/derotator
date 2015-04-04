#!/usr/bin/env python

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


def measure_derotator(input_pattern, plot=True):
    output_file = input_pattern.replace("_???.fits",".txt")
    filenames = glob.glob(input_pattern)
    if plot:
        fig, axarr = plt.subplots(1,len(filenames), figsize=(2*len(filenames),3))
        fig.suptitle(input_pattern)
    angles = []
    xs = []
    ys = []
    for i, filename in enumerate(filenames):
        angle = float(re.search("_([0-9][0-9][0-9]).fits", filename).group(1))
        barycenter = measure_centroid(filename, ax=(axarr[i] if plot else None))
        if plot:
            axarr[i].set_title("{0} deg".format(angle))
        angles.append(angle)
        xs.append(barycenter[0])
        ys.append(barycenter[1])
    table = Table()
    table['angle'] = angles
    table['x'] = xs
    table['y'] = ys
    print("Saving {0}...".format(output_file))
    print(table)
    table.write(output_file, format="ascii.fixed_width_two_line")


if __name__ == '__main__':
    input_pattern = "./DerotB_0_???.fits"
    measure_derotator(input_pattern, plot=True)
    plt.show()
