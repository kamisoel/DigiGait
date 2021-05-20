import subprocess
import os
import sys
from pathlib import Path

import numpy as np

from dash_app.config import Config

# use ffprobe to get the duration of a video
def get_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_asset(file):
	return os.path.join(Config.ASSETS_ROOT, file)


def random_upload_url(mkdir=False):
	from secrets import token_urlsafe
	url = Path(Config.UPLOAD_ROOT) / token_urlsafe(16)
	if mkdir:
		url.mkdir(parents=True, exist_ok=True)
	return url


def run_estimation(video_name='video.mp4', bbox_name='bboxes.npy', 
					in_dir=None, video_range=None):

	from data.person_detection import detect_person
	from model.lpn_estimator_2d import LPN_Estimator2D
	from model.videopose3d import VideoPose3D
	from data.video_dataset import VideoDataset
	from data.h36m_skeleton_helper import H36mSkeletonHelper
	from data.angle_helper import calc_common_angles

	if in_dir is None:
		in_dir = Config.UPLOAD_ROOT
		
	in_dir = Path(in_dir)
	video_file = in_dir / video_name
	bbox_file = in_dir / bbox_name

	if bbox_file.exists():
	    bboxes = np.load(bbox_file.resolve())
	else:
	    bboxes = detect_person('yolov5s', video_file, bbox_file, 
	                           video_out=in_dir / (video_file.stem + '_bboxes.mp4'))

	estimator_2d = LPN_Estimator2D()
	keypoints, meta = estimator_2d.estimate(video_file, bboxes)

	estimator_3d = VideoPose3D()
	pose_3d = estimator_3d.estimate(keypoints, meta)
	pose_3d = pose_3d[video_file.stem]

	knee_angles = calc_common_angles(pose_3d)

	#skeleton_helper = H36mSkeletonHelper()
	#angles = skeleton_helper.pose2euler(pose_3d)
	#knee_angles = {k: v[:,1] for k, v in angles.items() if k.endswith('Knee')}

	return pose_3d, knee_angles