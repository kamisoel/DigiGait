import numpy as np
import vg
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks

import data.skeleton_helper as skeletons
from data.timeseries_utils import time_normalize, peakdet, align_values, lp_filter, find_outliers

class GaitCycleDetector(object):
    """
    Helper class to detect gait events (HS & TO) using the kinematic data

    This class implements several algorithms for gait event
    detection, incl.:
        - Foot velocity algorithm by O'Connor et al. (2007)
        - Horizontal Heel Distance (HHD) by Banks et al. (2015)
        - Rel. Foot Displacement (RFD) based on Zeni et al. (2008)
        - an essamble method using all 3 algorithms
    In addition to event detection, some utility methods for splitting
    data into strides and time normalizing the data are defined as well

    Methods
    -------
    normed_gait_phases(data: np.ndarray, strides: list)
        returns data as time normalized splits into strides

    detect(pose, mode)
        detect HS and TO gait events from the pose data using the alg.
        specified in mode
    """

    def __init__(self, pose_format='h36m'):
        """
        Parameters
        ----------
        pose_format : str
            define which pose topology is used
            can be 'mediapipe', 'openpose', 'coco' or 'h36m'

        Raises
        ------
        ValueError
            If another pose_format is specified then the ones above
        """

        self.pose_format = pose_format
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
            self.skel = skeletons.H36mSkeletonHelper()
            self.lfoot = self.skel.keypoint2index['LeftAnkle']
            self.rfoot = self.skel.keypoint2index['RightAnkle']
            self.lknee = self.skel.keypoint2index['LeftKnee']
            self.rknee = self.skel.keypoint2index['RightKnee']
            self.lhip = self.skel.keypoint2index['LeftHip']
            self.rhip = self.skel.keypoint2index['RightHip']
            self.mhip = self.skel.keypoint2index['Hip']
        else:
            raise ValueError(f'Illegal pose format: {pose_format}')
    

    def normed_gait_phases(self, data, strides):
        """
        splits the given data (e.g. angles) into seperate strides
        and time normalizing these splits to 101 points

        Parameters
        ----------
        data : np.ndarray
            raw data (e.g. knee angle) to split
        
        strides : list or array-like
            list of HS event timings to split the data on

        Returns
        -------
        np.ndarray
            stacked, time-normalized splits
        """

        split_norm = self._split_and_filter(data, strides, True)
        return np.stack(split_norm)



    def _split_and_filter(self, data, strides, time_normalized = False):
        splits = []
        strides = strides.astype(int)

        if len(strides) <= 1:
            return [data]

        lengths = strides[1:] - strides[:-1]
        is_filtered = find_outliers(lengths, 1.5)

        for i in range(0, len(strides)-1):
            if not is_filtered[i]:
                start, end = strides[i], strides[i+1]
                split = data[start:end]
                if time_normalized:
                    split = time_normalize(split)
                splits.append(split)
        
        return splits


    def _norm_walking_dir(self, pose):
        """
        Helper method to normalize the used coordinate system
        """

        if pose.shape[-1] == 2: # in 2D make sure movement is from left to right
            pelvis = 0.5 * (pose[:, self.rhip] + pose[:, self.lhip])
            v_pelvis = np.gradient(pelvis[..., 0]) # get movement speed
            # caculate running mean to keep 'direction' while turning
            N = 10
            cumsum = np.cumsum(np.insert(v_pelvis, 0, 0)) 
            running_mean = (cumsum[N:] - cumsum[:-N]) / float(N)
            # use the sign of the speed as movement direction and pad edges
            walking_dir = np.pad(np.sign(running_mean), N//2, mode='edge')
            pose = pose.copy()
            pose[...,1] *= walking_dir
            return pose

        elif pose.shape[-1] == 3: # in 3D keep hip parallel to the x-axis
            x_axis = np.array([1,0,0])
            z_axis = np.array([0,0,1])
            origin = pose[:, [self.mhip]] if self.mhip != -1 else \
                    0.5 * (pose[:, [self.rhip]] + pose[:, [self.lhip]])

            orient = vg.angle((pose[:, self.lhip] - pose[:, self.rhip]), x_axis, look=z_axis)
            new_pose = np.zeros_like(pose)
            for i in range(len(pose)):
                new_pose[i] = vg.rotate(pose[i] - origin[i], z_axis, orient[i])
            return new_pose + origin


    def filter_false_pos(self, strides, knee_flex, threshold=50):
        """
        sanity check to filter obvious false positive HS events
        """

        kept = []
        for i in range(len(strides)-1):
            split = knee_flex[strides[i]:strides[i+1]]
            # a flex > 50° in loading response is most likely a fp
            if np.max(split[:len(split)//4]) < threshold:
                kept.append(i)
        return strides[kept]


    def detect(self, pose, mode='auto', **kwargs):
        """
        Common method for the different event detection algorithms

        Parameters
        ----------
        pose : np.ndarray
            2D/3D pose data

        mode : str
            event detection algorithm to use
            'auto' -> ensemble method
            'rnn'  -> RNN based (not recommended for now)
            'fva'  -> foot velocity algorithm
            'hhd'  -> horizontal heel displacement
            'rfd'  -> rel. foot displacement

        **kwargs
            additional parameters, e.g. lp_freq to pass to the 
            detection method

        Returns
        -------
        tuple
            gait events as 4-tuple of lists (RHS, LHS, RTO, LTO)

        Raises
        ------
        ValueError
            If unknown mode is specified
        """

        if mode == 'auto':
            return self.combined_detection(pose, **kwargs)
        elif mode == 'rnn':
            return self.rnn_detection(pose, **kwargs)
        elif mode == 'fva':
            return self.fva_detection(pose, **kwargs)
        elif mode == 'hhd':
            return (*self.hhd_detection(pose, **kwargs), 
                    np.array([]), np.array([]))
        elif mode == 'rfd':
            return self.simple_detection(pose, **kwargs)
        else:
            raise ValueError(f'Unknow detection mode: {mode}')


    def combined_detection(self, pose, lp_freq=6, tolerance=15):
        """
        Ensemble method using RFD, FVA and HHD together
        """

        # use foot displacement algorithm as basis
        rhs, lhs, rto, lto = self.simple_detection(pose, lp_freq)
        # use hhd algorithm for better HS detection
        rhs2, lhs2 = self.hhd_detection(pose, lp_freq)
        # use fva algorithm for better TO detection
        _, _, rto2, lto2 = self.fva_detection(pose, lp_freq)

        rhs = align_values(rhs, rhs2, 'mean', tolerance, keep='left')
        lhs = align_values(lhs, lhs2, 'mean', tolerance, keep='left')
        rto = align_values(rto, rto2, 'mean', tolerance, keep='left')
        lto = align_values(lto, lto, 'mean', tolerance, keep='left')

        return np.round(rhs), np.round(lhs), np.round(rto), np.round(lto)


    def fva_detection(self, pose, lp_freq=6):
        """
        Foot velocity algorithm by O'Connor et al. (2007)
        """

        # finds heelstrike (hs) and toe off (to) from vertical position and velocity
        def _detect(vertical_foot_pos):
            # use finite estimate to calculate velocity
            foot_vel = np.gradient(vertical_foot_pos)
            # filter the curve using a gaussian kernel 
            filtered_foot_vel = lp_filter(foot_vel, lp_freq)

            # find minima and maxima
            # all significant local minima
            hs, _ = find_peaks(-filtered_foot_vel, height=-filtered_foot_vel.min()/10)
            # only the 'global' maxima
            to, _ = find_peaks(filtered_foot_vel, height=filtered_foot_vel.max()/2)
            if hs.size != 0:
                # "exclude the major trough in the foot velocity signal which occurs during the swing phase of gait"
                ground_dist = vertical_foot_pos - vertical_foot_pos.min(axis=0)
                close_to_ground = ground_dist < 0.35 * np.ptp(vertical_foot_pos)
                hs = hs[close_to_ground[hs]]
            return hs, to
    
        if pose.shape[-1] == 3: # if 3D
            pose = self._norm_walking_dir(pose)
        rhs, rto = _detect(pose[:, self.rfoot, -1])
        lhs, lto = _detect(pose[:, self.lfoot, -1])
        return rhs, lhs, rto, lto


    def hhd_detection(self, pose, lp_freq=6, delta=0.5):
        """
        Heel strike detection using the Horizontal Heel Distance (HHD) 
        algorithm by Banks et al. (2015)
        """

        pose = self._norm_walking_dir(pose)
        rheel, lheel = pose[:, self.rfoot, -2:], pose[:, self.lfoot, -2:] #ignore x-axis in 3D
        feet_dist = np.linalg.norm(rheel - lheel, axis=1) # horizontal heel distance
        is_right = np.sign(rheel[:, 0]) # right foot in front of pelvis
        filtered_dist = lp_filter(feet_dist * is_right, lp_freq)

        mins, maxs = peakdet(filtered_dist, 0.05) # get turning points aka peaks of the gradient
        rhs = mins[:, 0] if mins.size!= 0 else []
        lhs = maxs[:, 0] if maxs.size!= 0 else []
        return rhs, lhs


    def simple_detection(self, pose, lp_freq=6, delta=0.2):
        """
        Relative foot displacement algorithm for event detection on treatmills
        based on Zeni et al. (2008)
        """

        def _detect(heel_pos):
            y_pos = lp_filter(heel_pos[:, 0], lp_freq)
            #z_pos = heel_pos[:, 1]
            #ground_dist = z_pos - z_pos.min(axis=0)
            #close_to_ground = ground_dist < 0.35 * np.ptp(z_pos)
            to, hs = peakdet(y_pos, delta)

            hs = hs[:, 0] if hs.size!= 0 else []
            to = to[:, 0] if to.size!= 0 else []
            return hs, to
    
        pose = self._norm_walking_dir(pose)
        rto, rhs = _detect(pose[:, self.rfoot, -2:])
        lto, lhs = _detect(pose[:, self.lfoot, -2:])
        return rhs, lhs, rto, lto


    def rnn_detection(self, pose, threshold=0.1):
        if self.pose_format != 'h36m':
            raise ValueError('Topology {self.pose_format} is not supported for now!')
        from model.gait_event_model import GaitEventModel

        ## create model
        model = GaitEventModel.load_pretrained()
        pose = self._norm_walking_dir(pose)

        return model.predict_from_pose(pose)
     