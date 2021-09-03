"""
Helper class to abstract the HPE models and the data processing
"""

import subprocess
import os
import io
import sys
from iocursor import Cursor
from pathlib import Path
from collections.abc import Mapping
from collections import defaultdict

import numpy as np
import pandas as pd

from data import c3d_helper
from data import nonlinear as nl
from dash_app.config import DashConfig
from data.person_detection import detect_person
from model.videopose3d import VideoPose3D
from data.video import Video
from data.video_dataset import VideoDataset
from data.h36m_skeleton_helper import H36mSkeletonHelper
from data.angle_helper import calc_common_angles
from data.gait_cycle_detector import GaitCycleDetector
from data.timeseries_utils import align_values, lp_filter, filter_outliers


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

def calc_nonlinear(angles):
    def _f(value):
        return f'{value:.2f}'

    embeddings = []
    nl_metrics = []
    for i in range(angles.shape[1]):
        data = angles[:,i]
        #delay = nl.estimate_delay(data)
        delay = 4
        dim = 3
        embeddings.append(nl.takensEmbedding(data, delay, dim))

        nl_metrics.append({
            'Lyapunov exponent': _f(nl.max_lyapunov_exp(data, delay, dim, 50)),
            'Sample entropy': _f(nl.sample_entropy(data, dim)),
            'Correlation Dimension': _f(nl.correlation_dim(data, dim))
        })
    embeddings = np.stack(embeddings, 0)
    nl_metrics = pd.DataFrame(nl_metrics, index=['Right side', 'Left side']).transpose()
    nl_metrics = nl_metrics.rename_axis('Metrics').reset_index()
    return embeddings, nl_metrics


def write_c3d_data(pose, angle, events):
    in_mem_file = io.BytesIO()
    c3d_helper.write_c3d(in_mem_file, pose, angle, events, freq=50)
    return in_mem_file.getvalue()


def load_c3d_data(file):
    #in_mem_file = io.BytesIO(data)
    pose, angles, events = c3d_helper.read_c3d(file)
    if len(pose) > 400: # long recording, more than 8 steps expected -> recalculate events
        gcd = GaitCycleDetector(pose_format='h36m')
        events = gcd.detect(pose, mode='auto')
    return pose, angles, events


def as_csv(pose, angle, events):
    pass


def get_asset(file):
    return os.path.join(DashConfig.ASSETS_ROOT, file)


def random_upload_url(mkdir=False):
    from secrets import token_urlsafe
    url = Path(DashConfig.UPLOAD_ROOT) / token_urlsafe(16)
    if mkdir:
        url.mkdir(parents=True, exist_ok=True)
    return url

def _normed_cycles(angles, events):
    gcd = GaitCycleDetector()
    rhs, lhs, *_ = events
    r_normed_phases = gcd.normed_gait_phases(angles[:,0], rhs)
    l_normed_phases = gcd.normed_gait_phases(angles[:,1], lhs)
    return np.stack([r_normed_phases, l_normed_phases], axis=0)


def calc_metrics(angles, events):

    def _mean_std_str(values, unit): # expect 1d-array (N)
        return f"{np.nanmean(values):.1f} ± {np.nanstd(values):.1f} {unit}"

    def _ratio_str(r, l):
        r, l= r.mean(), l.mean()
        sign = '+' if r/l > 1 else ''
        return sign + f"{(100 * r / l - 100):.2f}%"

    def _row(name, unit, right, left):
        right = filter_outliers(right, 2.5)
        left = filter_outliers(left, 2.5)
        return name, _mean_std_str(right, unit), \
               _mean_std_str(left, unit), _ratio_str(right, left)

    def _time(values, fps=50):
        return values / fps * 1000

    names = ['Metric', 'Right side', 'Left side', 'Right / Left ratio']

    rhs, lhs, rto, lto = events
    gcd = GaitCycleDetector()
    rcycles = gcd.normed_gait_phases(angles[:,0], rhs) # (N, 101)
    lcycles = gcd.normed_gait_phases(angles[:,1], lhs)

    rstance = align_values(rhs, rto, 'diff', tolerance=30, start_left=True)
    lstance = align_values(lhs, lto, 'diff', tolerance=30, start_left=True)
    
    rswing = align_values(rto, rhs, 'diff', tolerance=30, start_left=True)
    lswing = align_values(lto, lhs, 'diff', tolerance=30, start_left=True)

    rdouble = align_values(rhs, lto, 'diff', tolerance=10, start_left=True)
    ldouble = align_values(lhs, rto, 'diff', tolerance=10, start_left=True)


    metrics = [_row('Range of motion', '°', rcycles.ptp(axis=-1), lcycles.ptp(axis=-1)),
               _row('Max peak', '°', rcycles.max(axis=-1), lcycles.max(axis=-1)),
               _row('Max peak (loading response)', '°', rcycles[...,:33].max(axis=-1), lcycles[...,:33].max(axis=-1)),
               _row('Total step time', 'ms', _time(np.diff(rhs)), _time(np.diff(lhs))),
               _row('Stance time', 'ms', _time(rstance), _time(lstance)),
               _row('Swing time', 'ms', _time(rswing), _time(lswing)),
               _row('Double support time', 'ms', _time(rdouble), _time(ldouble)),
              ]
    return pd.DataFrame(metrics, columns=names)


def get_demo_data():
    demo_path = Path(DashConfig.DEMO_DATA) / 'demo_data.npz'
    demo_data = np.load(demo_path, allow_pickle=True)
    demo_pose = demo_data['pose_3d']
    demo_angles = 1.2 * np.stack([demo_data['rknee_angle'], demo_data['lknee_angle']], axis=-1)
    gait_events = (demo_data['rcycles'], demo_data['lcycles'], None, None)

    demo_pose = lp_filter(demo_pose, 7)
    demo_angles = lp_filter(demo_angles, 7)
    gait_events = GaitCycleDetector('h36m').detect(demo_pose, mode='auto')
    return demo_pose, demo_angles, gait_events


def calc_avg_stride(angles, events):
    gcd = GaitCycleDetector()
    rhs, lhs, _, _ = events
    if len(rhs) < 2 or len(lhs) < 2: # not enough events to calculate
        return np.zeros(100), np.zeros(100)

    r_normed_phases = gcd.normed_gait_phases(angles[:,0], rhs)
    l_normed_phases = gcd.normed_gait_phases(angles[:,1], lhs)
    r_mean = np.mean(r_normed_phases[:], axis=0)
    l_mean = np.mean(l_normed_phases[:], axis=0)
    return r_mean, l_mean


def get_norm_data(name='overground', joint=None, clinical=True):
    norm_path = Path(DashConfig.DEMO_DATA) / 'norm_data.npz'
    data = np.load(norm_path)
    keys = data['keys'].tolist() #index of key gives pos of joint
    norm_values = {}
    for i, k in enumerate(keys):
        norm_values[k] = data[name][:, i]
        if not clinical:
            norm_values[k] = 180 - norm_values[k]
    return norm_values

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
                    detection='auto', ops=defaultdict):
    with Video(video_path) as video:

        start, end = map(lambda x: int(x*video.fps), video_range)
        end = min(end, len(video))
        video = video[start:end] if video_range is not None else video

        if pipeline == 'lpn': #'LPN + VideoPose3D':
            from model.lpn_estimator_2d import LPN_Estimator2D
            estimator_2d = LPN_Estimator2D()
            estimator_3d = VideoPose3D(normalized_skeleton=ops['skel_norm'])
            #gcd = GaitCycleDetector(pose_format='coco')
        elif pipeline == 'mp_nf': #'MediaPipe + VideoPose3D (w/o feet)':
            from model.mediapipe_estimator import MediaPipe_Estimator2D
            estimator_2d = MediaPipe_Estimator2D(out_format='coco')
            estimator_3d = VideoPose3D(normalized_skeleton=ops['skel_norm'])
            #gcd = GaitCycleDetector(pose_format='coco')
        elif pipeline == 'mp_wf': #'MediaPipe + VideoPose3D (w/ feet)':
            from model.mediapipe_estimator import MediaPipe_Estimator2D
            estimator_2d = MediaPipe_Estimator2D(out_format='openpose')
            estimator_3d = VideoPose3D(openpose=True, normalized_skeleton=ops['skel_norm'])
            #gcd = GaitCycleDetector(pose_format='openpose')
        else:
            raise ValueError('Invalid Pipeline: ', pipeline)
        gcd = GaitCycleDetector(pose_format='h36m')
        
        keypoints, meta = estimator_2d.estimate(video)
        pose_2d = keypoints['video']['custom'][0]
        pose_3d = estimator_3d.estimate(keypoints, meta)
        pose_3d = next(iter(pose_3d.values()))
        pose_3d = lp_filter(pose_3d, 6)

        angles = calc_common_angles(pose_3d, clinical=True)
        knee_angles = np.stack([angles['RKnee'], angles['LKnee']], axis=-1)
        if ops['debias']:
            knee_angles *= 1.2

        gait_events = gcd.detect(pose_3d, mode=detection)

        #skeleton_helper = H36mSkeletonHelper()
        #angles = skeleton_helper.pose2euler(pose_3d)
        #knee_angles = {k: v[:,1] for k, v in angles.items() if k.endswith('Knee')}

        return pose_3d, knee_angles, gait_events

