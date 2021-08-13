import subprocess
import os
import sys
from iocursor import Cursor
from pathlib import Path
from collections.abc import Mapping
from collections import defaultdict

import numpy as np

from dash_app.config import DashConfig
from data.person_detection import detect_person
from model.videopose3d import VideoPose3D
from data.video import Video
from data.video_dataset import VideoDataset
from data.h36m_skeleton_helper import H36mSkeletonHelper
from data.angle_helper import calc_common_angles
from data.gait_cycle_detector import GaitCycleDetector

# use ffprobe to get the duration of a video
def ffprobe_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_duration(video_path):
    with Video(video_path) as video:
        return video.duration


def get_asset(file):
    return os.path.join(DashConfig.ASSETS_ROOT, file)


def random_upload_url(mkdir=False):
    from secrets import token_urlsafe
    url = Path(DashConfig.UPLOAD_ROOT) / token_urlsafe(16)
    if mkdir:
        url.mkdir(parents=True, exist_ok=True)
    return url

def calc_metrics(knee_angles, gait_events):
    pass

def get_demo_data():
    demo_path = Path(DashConfig.DEMO_DATA) / 'demo_data.npz'
    demo_data = np.load(demo_path, allow_pickle=True)
    demo_pose = demo_data['pose_3d']
    demo_angles = np.stack([demo_data['rknee_angle'], demo_data['lknee_angle']], axis=-1)
    gait_cycles = (demo_data['rcycles'], demo_data['lcycles'])
    return demo_pose, demo_angles, gait_cycles


def avg_gait_phase(angles, cycles):
    gcd = GaitCycleDetector()
    if isinstance(cycles, tuple):
        r_normed_phases = gcd.normed_gait_phases(angles[:,0], cycles[0])
        l_normed_phases = gcd.normed_gait_phases(angles[:,1], cycles[1])
        r_mean = np.mean(r_normed_phases[:5], axis=0)
        l_mean = np.mean(l_normed_phases[:5], axis=0)
        return r_mean, l_mean

    normed_phases = gcd.normed_gait_phases(angles, cycles)
    # use average of first five steps
    mean_phase = np.mean(normed_phases[:5], axis=0)
    return mean_phase[:,0], mean_phase[:,1]


def get_norm_data(clinical=True):
    norm_path = Path(DashConfig.DEMO_DATA) / 'norm_knee.npz'
    norm_data = np.load(norm_path)
    mean = np.rad2deg(norm_data['mean'])
    std = np.rad2deg(norm_data['std'])
    if not clinical:
        mean = 180 - mean
        std = 180 - std
    return dict(Knee=(mean, std))

def get_sagital_view(pose_3d):
    RHip, LHip = 1, 4
    hip = pose_3d[0, RHip] - pose_3d[0, LHip]
    return dict(x=1+hip[0], y=2.5, z=0.25)


def memory_file(content):
    return Cursor(content) # alt: io.BytesIO(content)


def run_estimation_file(video_name='video.mp4', bbox_name='bboxes.npy', 
                    in_dir=None, video_range=None):
    if in_dir is None:
        in_dir = DashConfig.UPLOAD_ROOT
        
    in_dir = Path(in_dir)
    video_file = in_dir / video_name
    bbox_file = in_dir / bbox_name

    if bbox_file.exists():
        bboxes = np.load(bbox_file.resolve())
    else:
        bboxes = detect_person('yolov5s', video_file, bbox_file, 
                               video_out=in_dir / (video_file.stem + '_bboxes.mp4'))

    return run_estimation(video_file, bboxes, video_range)


def run_estimation(video_path, video_range=None, 
                    pipeline='Mediapipe + VideoPose3D',
                    ops=defaultdict):
    with Video(video_path) as video:

        start, end = map(lambda x: int(x*video.fps), video_range)
        end = min(end, len(video))
        video = video[start:end] if video_range is not None else video

        if pipeline == 'lpn': #'LPN + VideoPose3D':
            from model.lpn_estimator_2d import LPN_Estimator2D
            estimator_2d = LPN_Estimator2D()
            estimator_3d = VideoPose3D(normalized_skeleton=ops['skel_norm'])
            #phase_detector = GaitCycleDetector(pose_format='coco')
        elif pipeline == 'mp_nf': #'MediaPipe + VideoPose3D (w/o feet)':
            from model.mediapipe_estimator import MediaPipe_Estimator2D
            estimator_2d = MediaPipe_Estimator2D(out_format='coco')
            estimator_3d = VideoPose3D(normalized_skeleton=ops['skel_norm'])
            #phase_detector = GaitCycleDetector(pose_format='coco')
        elif pipeline == 'mp_wf': #'MediaPipe + VideoPose3D (w/ feet)':
            from model.mediapipe_estimator import MediaPipe_Estimator2D
            estimator_2d = MediaPipe_Estimator2D(out_format='openpose')
            estimator_3d = VideoPose3D(openpose=True, normalized_skeleton=ops['skel_norm'])
            #phase_detector = GaitCycleDetector(pose_format='openpose')
        else:
            raise ValueError('Invalid Pipeline: ', pipeline)
        phase_detector = GaitCycleDetector(pose_format='h36m')
        
        keypoints, meta = estimator_2d.estimate(video)
        pose_2d = keypoints['video']['custom'][0]
        pose_3d = estimator_3d.estimate(keypoints, meta)
        pose_3d = next(iter(pose_3d.values()))

        angles = calc_common_angles(pose_3d, clinical=True)
        knee_angles = np.stack([angles['RKnee'], angles['LKnee']], axis=-1)
        if ops['debias']:
            knee_angles *= 1.2

        rcycles, lcycles = phase_detector.heel_strike_detection(pose_3d)
        rcycles = phase_detector.filter_false_pos(rcycles, angles['RKnee'])
        lcycles = phase_detector.filter_false_pos(lcycles, angles['LKnee'])

        #skeleton_helper = H36mSkeletonHelper()
        #angles = skeleton_helper.pose2euler(pose_3d)
        #knee_angles = {k: v[:,1] for k, v in angles.items() if k.endswith('Knee')}

        return pose_3d, knee_angles, (rcycles, lcycles)

