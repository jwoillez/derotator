#!/usr/bin/env python

from argparse import ArgumentParser
from collections import OrderedDict
from astropy.table import Table
import numpy as np
import pylab as plt
import dpfit
import derotator
import glob


if __name__ == '__main__':
    description = \
        """
        """
    parser = ArgumentParser(description=description)
    parser.add_argument("-p", "--plot",
                        action="store_true", dest="plot", default=False,
                        help="plot model fitting diagnostics")
    parser.add_argument("-r", "--reverse",
                        action="store_true", dest="reverse", default=False,
                        help="reverse rotation angle")
    parser.add_argument("-s", "--save",
                        action="store_true", dest="save", default=False,
                        help="save figures")
    parser.add_argument("root", help="root name of the files to process")
    args = parser.parse_args()

    params_names = ['der_x', 'der_y', 'int_x', 'int_y', 'beam_x', 'beam_y']
    params = OrderedDict()
    for name in params_names:
        params[name] = 0.0
    params['@sign'] = +1.0 if not args.reverse else -1.0

    result = Table()
    result['filename'] = glob.glob(args.root+"_*.txt")
    for name in params_names:
        result[name] = np.zeros(len(result))

    for i, filename in enumerate(result['filename']):
        data = Table.read(filename, format="ascii.fixed_width_two_line")

        # Fit a derotator model
        params_fit = dpfit.leastsq(derotator.residuals, params, [data])

        # Save fit
        for name in params_names:
            result[name][i] = params_fit[name]

        # Plot results, if requested
        if args.plot:
            fig, axarr = plt.subplots(2,1,figsize=(6,12))
            derotator.plot_errors(axarr, data, params_fit)
            derotator.plot_model(axarr, params_fit)
            derotator.plot_data(axarr, data)
            if args.save:
                fig.savefig(filename.replace('.txt','.png'))

    der_r = np.max(np.sqrt(result['der_x']**2+result['der_y']**2))
    beam_r = np.max(np.sqrt(result['beam_x']**2+result['beam_y']**2))
    int_r = np.max(np.sqrt(result['int_x']**2+result['int_y']**2))
    max_r = np.max([der_r, int_r, beam_r])

    fig, axarr = plt.subplots(1,1)
    axarr.plot(result['der_x'], result['der_y'], 'b', label="Derotator")
    axarr.plot(result['beam_x'], result['beam_y'], 'g', label="Beam")
    axarr.plot(result['int_x'], result['int_y'], 'r', label="Internal")
    axarr.plot(result['der_x'], result['der_y'], 'b.')
    axarr.plot(result['beam_x'], result['beam_y'], 'g.')
    axarr.plot(result['int_x'], result['int_y'], 'r.')
    axarr.plot(result['der_x'][-1], result['der_y'][-1], 'o', color='b')
    axarr.plot(result['beam_x'][-1], result['beam_y'][-1], 'o', color='g')
    axarr.plot(result['int_x'][-1], result['int_y'][-1], 'o', color='r')
    axarr.set_aspect('equal', adjustable='box')
    axarr.set_xlim(-max_r, +max_r)
    axarr.set_ylim(-max_r, +max_r)
    axarr.grid()
    axarr.legend()

    plt.show()
