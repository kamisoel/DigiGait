import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d


def moving_avg(data, window=3):
    return np.apply_along_axis(lambda x: pd.Series(x).rolling(window).mean(),
                               arr = data, axis = 0)

def gauss_filter(data, sd=1):
    return np.apply_along_axis(lambda x: gaussian_filter1d(x,sd),
                               arr = data, axis = 0)

def time_normalize(y, steps=100):
    x = np.linspace(0, 1, len(y))
    new_x = np.linspace(0, 1, steps)
    if y.ndim > 1:
        return np.apply_along_axis(
            lambda y: interp1d(x, y, kind = 'cubic')(new_x),
            arr=y, axis=0
        )
    else:
        return interp1d(x, y, kind = 'cubic')(new_x)