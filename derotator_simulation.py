#!/usr/bin/env python

from collections import OrderedDict
from astropy.table import Table
import numpy as np
import derotator


if __name__ == '__main__':
    der_x_list = [-0.5,+0.1,+0.0,+0.0,+0.0]
    der_y_list = [-0.2,-0.2,-0.1,+0.0,+0.0]
    beam_x_list = [+0.4,+0.4,+0.4,+0.4,+0.0]
    beam_y_list = [-0.2,-0.2,-0.2,+0.1,+0.0]

    for i, der_x, der_y, beam_x, beam_y in zip(range(len(der_x_list)), der_x_list, der_y_list, beam_x_list, beam_y_list):

        # Parameters for the derotator model
        params = OrderedDict()
        params['der_x'] = der_x
        params['der_y'] = der_y
        params['int_x'] = 0.8
        params['int_y'] = 0.1
        params['beam_x'] = beam_x
        params['beam_y'] = beam_y
        params['sign'] = +1.0

        # Data generation, adding noise, and saving
        data = Table()
        data['angle'] = np.arange(0.0, 360.0, 45.0)
        x, y = derotator.model(params, data)
        x += np.random.normal(scale=0.05, size=x.shape)
        y += np.random.normal(scale=0.05, size=y.shape)
        data['x'] = x
        data['y'] = y
        data.write("data_{0}.txt".format(i), format="ascii.fixed_width_two_line")
