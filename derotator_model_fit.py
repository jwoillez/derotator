from collections import OrderedDict
from astropy.table import Table
import numpy as np
import pylab as plt
import dpfit
import derotator
import glob


pattern = "./data_*.txt"
# pattern = "./data_0.txt"

der_x = []
der_y = []
int_x = []
int_y = []
beam_x = []
beam_y = []

filenames = glob.glob(pattern)

for filename in filenames:
    data = Table.read(filename, format="ascii.fixed_width_two_line")

    # Fit a derotator model
    params = OrderedDict()
    params['der_x'] = 0.0
    params['der_y'] = 0.0
    params['int_x'] = 0.0
    params['int_y'] = 0.0
    params['beam_x'] = 0.0
    params['beam_y'] = 0.0
    params['@sign'] = +1.0
    params_fit = dpfit.leastsq(derotator.residuals, params, [data])

    # Save fit
    der_x.append(params_fit['der_x'])
    der_y.append(params_fit['der_y'])
    int_x.append(params_fit['int_x'])
    int_y.append(params_fit['int_y'])
    beam_x.append(params_fit['beam_x'])
    beam_y.append(params_fit['beam_y'])

    # Plot results
    fig, axarr = plt.subplots(2,1,figsize=(6,12))
    derotator.plot_errors(axarr, data, params_fit)
    derotator.plot_model(axarr, params_fit)
    derotator.plot_data(axarr, data)

der_x = np.array(der_x)
der_y = np.array(der_y)
int_x = np.array(int_x)
int_y = np.array(int_y)
beam_x = np.array(beam_x)
beam_y = np.array(beam_y)

if len(filenames) > 1:

    der_r = np.max(np.sqrt(der_x**2+der_y**2))
    beam_r = np.max(np.sqrt(beam_x**2+beam_y**2))
    int_r = np.max(np.sqrt(int_x**2+int_y**2))
    max_r = np.max([der_r, int_r, beam_r])

    fig, axarr = plt.subplots(1,1)
    axarr.plot(der_x, der_y, 'b', label="Derotator")
    axarr.plot(beam_x, beam_y, 'g', label="Beam")
    axarr.plot(int_x, int_y, 'r', label="Internal")
    axarr.plot(der_x, der_y, 'b.')
    axarr.plot(beam_x, beam_y, 'g.')
    axarr.plot(int_x, int_y, 'r.')
    axarr.arrow(der_x[-2], der_y[-2], der_x[-1]-der_x[-2], der_y[-1]-der_y[-2], color='b', length_includes_head=True, width=0.002)
    axarr.arrow(beam_x[-2], beam_y[-2], beam_x[-1]-beam_x[-2], beam_y[-1]-beam_y[-2], color='g', length_includes_head=True, width=0.002)
    axarr.arrow(int_x[-2], int_y[-2], int_x[-1]-int_x[-2], int_y[-1]-int_y[-2], color='r', length_includes_head=True, width=0.002)
    axarr.set_aspect('equal', adjustable='box')
    axarr.set_xlim(-max_r, +max_r)
    axarr.set_ylim(-max_r, +max_r)
    axarr.grid()
    axarr.legend()

plt.show()
