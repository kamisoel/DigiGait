from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import mediapipe as mp

from model.estimator_2d import Estimator2D
from data.video_dataset import VideoDataset
from data.skeleton_helper import mediapipe2openpose
from data.data_utils import suggest_metadata

class MediaPipe_Estimator2D(Estimator2D):
    """2D human pose estimator using MediaPipe"""

    BATCH_SIZE = 64

    def __init__(self, device='cpu'):
        self.device = device
        self.mp_pose = mp.solutions.pose


    def _image_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        # Reverse camera frame normalization
        return (X + [1, h/w])*w/2

    def estimate(self, video):
        
        with self.mp_pose.Pose(static_image_mode=False) as pose:
            pose_2d = []
            for frame in video:
              result = pose.process(frame)
              pose_2d.append([[p.x, p.y] for p in result.pose_landmarks.landmark])
            pose_2d = np.vstack(pose_2d).reshape(-1, 33, 2)
            pose_2d = mediapipe2openpose(pose_2d)
            pose_2d = self._image_coordinates(pose_2d, *video.size)

            # create VideoPose3D-compatible metadata and keypoint structure
            metadata = suggest_metadata('coco')
            video_name = 'video'
            video_meta = {'w': video.size[0], 'h': video.size[1], 'fps': video.fps}
            metadata['video_metadata'] = {video_name: video_meta}
            keypoints = {video_name: {'custom': [pose_2d]}}

        return keypoints, metadata

