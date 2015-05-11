import numpy as np
from astropy.table import Table


def model(params, data):
    angle = np.deg2rad(data['angle'])
    x = params['der_x']
    y = params['der_y']
    x += + params['int_x']*np.cos(params['sign']*angle) + params['int_y']*np.sin(params['sign']*angle)
    y += - params['int_x']*np.sin(params['sign']*angle) + params['int_y']*np.cos(params['sign']*angle)
    x += + params['beam_x']*np.cos(2*params['sign']*angle) + params['beam_y']*np.sin(2*params['sign']*angle)
    y += - params['beam_x']*np.sin(2*params['sign']*angle) + params['beam_y']*np.cos(2*params['sign']*angle)
    return x, y


def residuals(params, data):
    x = data['x']
    y = data['y']
    model_x, model_y = model(params, data)
    return np.hstack([model_x-x, model_y-y])


def plot_model(axarr, params):

    # Compute plot extent, based on worst case scenario
    r1 = np.sqrt(params['der_x']**2+params['der_y']**2)
    r1 += np.sqrt(params['int_x']**2+params['int_y']**2)
    r1 += np.sqrt(params['beam_x']**2+params['beam_y']**2)
    r0 = r1*1.2

    # Compute model
    data = Table()
    data['angle'] = np.linspace(0.0, 360.0, 1000)
    x, y = model(params, data)

    # Plot model with diagnostics
    axarr[0].plot(x, y, 'k')
    line1, = axarr[0].plot(params['der_x'], params['der_y'], 'o', label="Derotator", lw=1.5, color='b')
    axarr[0].plot([params['der_x'], params['der_x']+params['beam_x']], [params['der_y'], params['der_y']+params['beam_y']], label="Beam", lw=1.5, color='g')
    axarr[0].plot([params['der_x']+params['beam_x'], params['der_x']+params['int_x']+params['beam_x']], [params['der_y']+params['beam_y'], params['der_y']+params['int_y']+params['beam_y']], label="Internal", lw=1.5, color='r')
    axarr[0].set_aspect('equal', adjustable='box')
    axarr[0].set_xlim(-r0,+r0)
    axarr[0].set_ylim(-r0,+r0)
    axarr[0].grid()
    axarr[0].legend(handler_map={line1: HandlerLine2D(numpoints=1)})
    #axarr[1].plot([0.0, params['der_x']], [0.0, params['der_y']], label="Derotator", lw=1.5, color='b')
    axarr[1].plot([0.0, params['beam_x']], [0.0, params['beam_y']], label="Beam", lw=1.5, color='g')
    axarr[1].plot([0.0, params['int_x']], [0.0, params['int_y']], label="Internal", lw=1.5, color='r')
    axarr[1].set_aspect('equal', adjustable='box')
    axarr[1].set_xlim(-r1,+r1)
    axarr[1].set_ylim(-r1,+r1)
    axarr[1].grid()
    axarr[1].legend()


def plot_data(axarr, data):
    x = data['x']
    y = data['y']
    angle = data['angle']
    axarr[0].plot(x, y, 'oc')
    for _x, _y, _angle in zip(x, y, angle):
        axarr[0].text(_x, _y, " {0} deg".format(_angle))


def plot_errors(axarr, data, params):
    x = data['x']
    y = data['y']
    model_x, model_y = model(params, data)
    sigma = np.mean(np.sqrt((x-model_x)**2+(y-model_y)**2))
    for x1, y1, x2, y2 in zip(x, y, model_x, model_y):
        axarr[0].plot([x1, x2], [y1, y2], 'k', lw=0.5)
    axarr[1].plot(sigma*np.cos(np.linspace(0.0,2*np.pi,100)), sigma*np.sin(np.linspace(0.0,2*np.pi,100)), 'k', lw=0.5)
