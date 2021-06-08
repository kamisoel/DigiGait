import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import data.skeleton_helper as skeletons

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
		else:
			raise ValueError('Illegal pose format')


	def simple_detection(self, pose_2d, filter_sd=5):
		toe_dist = np.linalg.norm(
			    	pose_2d[:, self.left_toe] - pose_2d[:, self.right_toe],
			    	axis=1)
		dist_org = toe_dist * np.sign(pose_2d[:, self.right_toe, 0] - pose_2d[:, self.left_toe, 0])
		filtered_dist = gaussian_filter1d(dist_org, filter_sd)

		mins, maxs = self._peakdet(filtered_dist, 0.5)

		return maxs[:,0] #return just the frames of the peaks


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