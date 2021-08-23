import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import filtfilt, butter
from scipy.interpolate import interp1d
from scipy.stats import iqr


def minmax_scale(X, feature_range=(-1,1)):
    data_min = np.nanmin(X, axis=0)
    data_range = np.nanmax(X, axis=0) - data_min
    _scale = (feature_range[1] - feature_range[0]) / data_range
    _min = feature_range[0] - data_min * _scale
    return X * _scale + _min

def moving_avg(data, window=3):
    return np.apply_along_axis(lambda x: pd.Series(x).rolling(window).mean(),
                               arr = data, axis = 0)

def find_outliers(data, iqr_factor=1.5):
    return np.abs(data - np.median(data)) > iqr_factor * iqr(data)


def filter_outliers(data, iqr_factor=1.5):
    return data[~find_outliers(data, iqr_factor)]


def noise_filter(data, sd=1):
    return gaussian_filter1d(data, sd, axis=0)


def lp_filter(data, lp_freq=10, order=4, fs=100):
    nyq = 0.5 * fs
    b, a = butter(order, lp_freq/nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def interp_along_time(data, old_fs=50, new_fs=100):
    x = np.linspace(0, 1, len(data))
    new_x = np.linspace(0, 1, int(new_fs/old_fs * len(data)))
    return interp1d(x, data, 'cubic', axis=0)(new_x)

def time_normalize(y, steps=101):
    x = np.linspace(0, 1, len(y))
    new_x = np.linspace(0, 1, steps)
    if y.ndim > 1:
        return np.apply_along_axis(
            lambda y: interp1d(x, y, kind = 'cubic')(new_x),
            arr=y, axis=0
        )
    else:
        return interp1d(x, y, kind = 'cubic')(new_x)


def align_values(left, right, f='mean', tolerance=10, keep='none', start_left=False):
    """
    align elements of A and B by nearest neighbor with distance <= tolerance
    f is used to aggregate paired elements
    unpaired elements can be kept or ignored
    left: list of values, must be iterable and sorted
    right: list of values, must be iterable and sorted
    f: String ('mean', 'min', 'max', 'zip') for common function or custom function for aggregating pairs
    tolerance: maximum distance to be counted as pair
    keep: decides which unpaired values to keep; can be 'none', 'left', 'right' or 'both'
    returns: list of paired and aggregated values and kept unpaired one
    """
    A = iter(left)
    B = iter(right)
    C = []

    try:
        a = next(A)
        b = next(B)
        while True:
            if start_left and a >= b:
                b = next(B)
            elif abs(a-b) <= tolerance:
                if f == 'mean':
                    C.append((a+b)/2)
                elif f == 'min':
                    C.append(min(a,b))
                elif f == 'max':
                    C.append(max(a,b))
                elif f == 'diff':
                    C.append(abs(a-b))
                elif f == 'zip':
                    C.append([a,b])
                elif callable(f):
                    C.append(f(a,b))
                a = next(A)
                b = next(B)
            elif a <= b:
                if keep == 'left' or keep == 'both':
                    C.append(a)
                a = next(A)
            else:
                if keep == 'right' or keep == 'both':
                    C.append(b)
                b = next(B)
    except StopIteration:
        pass
    # one of the iterators may have leftover elements
    for a in A:
        if keep == 'left' or keep == 'both':
            C.append(a)
    for b in B:
        if keep == 'right' or keep == 'both':
            C.append(b)
    return np.array(C)




# Peak detection script converted from MATLAB script
# at http://billauer.co.il/peakdet.html
#    
# Returns two arrays
#    
# function [maxtab, mintab]=peakdet(v, delta, x)
# PEAKDET Detect peaks in a vector
# [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
# maxima and minima ("peaks") in the vector V.
# MAXTAB and MINTAB consists of two columns. Column 1
# contains indices in V, and column 2 the found values.
#     
# With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
# in MAXTAB and MINTAB are replaced with the corresponding
# X-values.
#
# A point is considered a maximum peak if it has the maximal
# value, and was preceded (to the left) by a value lower by
# DELTA.
#
# Eli Billauer, 3.4.05 (Explicitly not copyrighted).
# This function is released to the public domain; Any use is allowed.
def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(mintab), np.array(maxtab)
