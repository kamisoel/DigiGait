import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import iqr

import data.skeleton_helper as skeletons
from data.timeseries_utils import time_normalize

class GaitCycleDetector(object):

    def __init__(self, pose_format='mediapipe'):
        if pose_format == 'mediapipe':
            self.skel = skeletons.MediaPipeSkeleton()
            self.left_toe = self.skel.keypoint2index['left_foot_index']
            self.right_toe = self.skel.keypoint2index['right_foot_index']
        elif pose_format == 'openpose':
            self.skel = skeletons.OpenPoseSkeleton()
            self.left_toe = self.skel.keypoint2index['LBigToe']
            self.right_toe = self.skel.keypoint2index['RBigToe']
        elif pose_format == 'coco': #no foot kpts only ankle
            self.skel = skeletons.CocoSkeleton()
            self.left_toe = self.skel.keypoint2index['LAnkle']
            self.right_toe = self.skel.keypoint2index['RAnkle']
        elif pose_format == 'h36m': 
            #TODO better soluttion.. left/right has to be switched..
            #self.skel = skeletons.H36mSkeleton()
            self.right_toe = 6
            self.left_toe = 3
        else:
            raise ValueError('Illegal pose format')


    def avg_gait_phase(self, data, cycles):
        """
        data: landmark positions or joint angles as (Frame, Joint, X) numpy array
        peaks: start of the gait cycles, e.g. return value of simple_detection
        return: 
        """
        split_norm = self.normed_gait_phases(data, cycles)
        return np.mean(split_norm, axis=0), np.std(split_norm, axis=0)
    

    def normed_gait_phases(self, data, cycles):
        split_norm = self._split_and_filter(data, cycles, True)
        return np.stack(split_norm)


    def _split_and_filter(self, data, cycles, time_normalized = False):
        splits = []
        cycles = cycles.astype(int)

        lengths = cycles[1:] - cycles[:-1]
        filtered = np.where(np.abs(lengths - np.median(lengths) > 1.5 * iqr(lengths)))
        #filtered = np.where(np.abs(lengths - np.mean(lengths) > 2*np.std(lengths)))

        for i in range(0, len(cycles)-1):
            if not np.isin(i, filtered):
                start, end = cycles[i], cycles[i+1]
                split = data[start:end]
                if time_normalized:
                    split = time_normalize(split)
                splits.append(split)
        
        return splits


    def heel_strike_detection(self, pose, filter_sd=5, prominence=0.01):
        feet_dist = np.linalg.norm(pose[:, self.left_toe, -2:] - pose[:, self.right_toe, -2:],
                              axis=1)
        right_in_front = np.sign(pose[:, self.right_toe, -2] - pose[:, self.left_toe, -2])
        filtered_dist = gaussian_filter1d(feet_dist, filter_sd)

        #mins, maxs = _peakdet(filtered_dist, 0.5)
        #return maxs[:, 0], mins[:, 0]

        re_peaks, _ = find_peaks(filtered_dist * right_in_front, prominence=prominence)
        le_peaks, _ = find_peaks(filtered_dist * -right_in_front, prominence=prominence)
        return le_peaks, re_peaks


    def simple_detection(self, pose, filter_sd=5):
        """
        pose: 2d/3d pose landmarks as (Frame, Joint, 2/2) numpy array
        filter_sd: gaussian filter strength for smoothing the distance function
        sagital_axis: axis number spcifying the sagital axis in pose
        returns: peaks of the distance between left and right foot
        """
        toe_dist = np.linalg.norm(
                    pose[:, self.left_toe, -2:] - pose[:, self.right_toe, -2:],
                    axis=1)
        dist_org = toe_dist * np.sign(pose[:, self.right_toe, -2] - pose[:, self.left_toe, -2])
        filtered_dist = gaussian_filter1d(dist_org, filter_sd)

        mins, maxs = self._peakdet(filtered_dist, 0.5)

        return maxs[:,0].astype(np.int) #return just the frames of the peaks


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
    def _peakdet(self, v, delta, x = None):
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