import scipy.optimize as opt
from collections import OrderedDict


def _wrapper(func):
    return lambda params, keys, fixed, data: func(dict(fixed.items() | zip(keys, params)), *data)


def fix(dict_params, fixed_key):
    if isinstance(dict_params, OrderedDict):
        list_params = []
        while dict_params:
            list_params.append(dict_params.popitem(last=False))
            if list_params[-1][0] == fixed_key:
                list_params[-1] = ('@'+fixed_key, list_params[-1][1])
        for key, value in list_params:
            dict_params[key] = value
    else:
        dict_params['@'+fixed_key] = dict_params.pop(fixed_key)


def leastsq(func, dict_params, args):
    """
    A modified least squares fitting routine that uses a dictionary as fitting parameters.
    Fitting parameters prepended with `@` are not optimized.
    """
    params = []
    keys = []
    fixed = {}
    for key, value in dict_params.items():
        if key[0] == '@':
            fixed[key[1:]] = value
        else:
            params.append(value)
            keys.append(key)
    params, status = opt.leastsq(_wrapper(func), params, args=(keys, fixed, args))
    params = dict(zip(keys, params))
    if isinstance(dict_params, OrderedDict):
        result = OrderedDict()
        for key in dict_params.keys():
            if key in params:
                result[key] = params[key]
            else:
                result[key.strip('@')] = dict_params[key]
    else:
        result = dict(fixed.items() | params.items())
    return result
