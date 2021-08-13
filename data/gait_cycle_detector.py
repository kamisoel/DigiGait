import numpy as np
import vg
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import iqr

import data.skeleton_helper as skeletons
from data.h36m_skeleton_helper import H36mSkeletonHelper
from data.timeseries_utils import time_normalize

class GaitCycleDetector(object):

    def __init__(self, pose_format='mediapipe', rnn_support=False):
        if pose_format == 'mediapipe':
            self.skel = skeletons.MediaPipeSkeleton()
            self.lfoot = self.skel.keypoint2index['left_foot_index']
            self.rfoot = self.skel.keypoint2index['right_foot_index']
            self.lhip = self.skel.keypoint2index['left_hip']
            self.rhip = self.skel.keypoint2index['right_hip']
            self.mhip = -1
        elif pose_format == 'openpose':
            self.skel = skeletons.OpenPoseSkeleton()
            self.lfoot = self.skel.keypoint2index['LBigToe']
            self.rfoot = self.skel.keypoint2index['RBigToe']
            self.lhip = self.skel.keypoint2index['LHip']
            self.rhip = self.skel.keypoint2index['RHip']
            self.mhip = self.skel.keypoint2index['MidHip']
        elif pose_format == 'coco': #no foot kpts only ankle
            self.skel = skeletons.CocoSkeleton()
            self.lfoot = self.skel.keypoint2index['LAnkle']
            self.rfoot = self.skel.keypoint2index['RAnkle']
            self.lhip = self.skel.keypoint2index['LHip']
            self.rhip = self.skel.keypoint2index['RHip']
            self.mhip = -1
        elif pose_format == 'h36m': 
            #left/right has to be switched..
            self.skel = H36mSkeletonHelper()
            self.lfoot = self.skel.keypoint2index['LeftAnkle']
            self.rfoot = self.skel.keypoint2index['RightAnkle']
            self.lhip = self.skel.keypoint2index['LeftHip']
            self.rhip = self.skel.keypoint2index['RightHip']
            self.mhip = self.skel.keypoint2index['Hip']
        else:
            raise ValueError('Illegal pose format')+delta


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

        for i in range(0, len(cycles)-1):
            if not np.isin(i, filtered):
                start, end = cycles[i], cycles[i+1]
                split = data[start:end]
                if time_normalized:
                    split = time_normalize(split)
                splits.append(split)
        
        return splits

    def _norm_walking_dir(self, pose):
        x_axis = np.array([1,0,0])
        z_axis = np.array([0,0,1])
        origin = pose[:, [self.mhip]] if self.mhip != -1 else \
                0.5 * (pose[:, [self.rhip]] + pose[:, [self.lhip]])

        orient = vg.angle((pose[:, self.lhip] - pose[:, self.rhip]), x_axis, look=z_axis)
        new_pose = np.zeros_like(pose)
        for i in range(len(pose)):
            new_pose[i] = vg.rotate(pose[i] - origin[i], z_axis, orient[i])
        return new_pose + origin


    def filter_false_pos(self, cycles, knee_flex, threshold=50):
        kept = []
        for i in range(len(cycles)-1):
            split = knee_flex[cycles[i]:cycles[i+1]]
            # a flex > 50° in loading response is most likely a fp
            if np.max(split[:len(split)//4]) < threshold:
                kept.append(i)
        return cycles[kept]


    def simple_detection(self, pose, filter_sd=3, prominence=0.3, distance=25):
        norm_pose = self._norm_walking_dir(pose)

        filtered_rfoot = gaussian_filter1d(norm_pose[:, self.rfoot, 1], filter_sd)
        filtered_lfoot = gaussian_filter1d(norm_pose[:, self.lfoot, 1], filter_sd)

        rhs, _ = find_peaks(filtered_rfoot, distance=distance, prominence=prominence)
        rto, _ = find_peaks(-filtered_rfoot, distance=distance, prominence=prominence)
        lhs, _ = find_peaks(filtered_lfoot, distance=distance, prominence=prominence)
        lto, _ = find_peaks(-filtered_lfoot, distance=distance, prominence=prominence)
        return rto, rhs, lto, lhs


    def rnn_detection(self, pose, threshold=0.7):
        from data.gait_event_model import GaitEventModel
        norm_pose = self._norm_walking_dir(pose)

        model = GaitEventModel.load_pretrained()
        model.eval()

        y_hat = torch.sigmoid(model(norm_pose)).detach().cpu().numpy()
        y_hat[y_hat < threshold] = 0
        # peak detection?

        rhs = np.nonzero(y_hat[:, 0])
        lhs = np.nonzero(y_hat[:, 1])
        rto = np.nonzero(y_hat[:, 2])
        lto = np.nonzero(y_hat[:, 3])

        return rto, rhs, lto, lhs



    def heel_strike_detection(self, pose, filter_sd=5, delta=0.5):
        """
        pose: 2d/3d pose landmarks as (Frame, Joint, 2/3) numpy array
        filter_sd: gaussian filter strength for smoothing the distance function
        returns: heel strikes of right and left foot
        """
        if pose.shape[2] == 3:
            pose = self._norm_walking_dir(pose)
            right_in_front = np.sign(pose[:, self.lfoot, 1] - pose[:, self.rfoot, 1])
        else:
            right_in_front = np.sign(pose[:, self.rfoot, 0] - pose[:, self.lfoot, 0])

        feet_dist = np.linalg.norm(pose[:, self.lfoot] - pose[:, self.rfoot], axis=1)
        filtered_dist = gaussian_filter1d(feet_dist, filter_sd)

        _, re_peaks = self._peakdet(filtered_dist * right_in_front, delta)
        _, le_peaks = self._peakdet(filtered_dist * -right_in_front, delta)
        #return just the frames of the peaks
        return re_peaks[:,0].astype(int), le_peaks[:,0].astype(int) 

        #re_peaks, _ = find_peaks(filtered_dist * -right_in_front, prominence=prominence)
        #le_peaks, _ = find_peaks(filtered_dist * right_in_front, prominence=prominence)
        #return re_peaks, le_peaks
        


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