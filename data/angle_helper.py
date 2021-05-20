import numpy as np


MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine, Neck, Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)

def moving_avg(data, window=3):
    return np.apply_along_axis(lambda x: pd.Series(x).rolling(window).mean(),
                               arr = data, axis = 0)

def gauss_filter(data, sd=1):
    from scipy.ndimage.filters import gaussian_filter1d
    return np.apply_along_axis(lambda x: gaussian_filter1d(x,sd),
                               arr = data, axis = 0)

def get_joint_angles(pose_3d, joint_idx):
    import vg
    xs = pose_3d[:, joint_idx[1]] - pose_3d[:, joint_idx[0]]
    if len(joint_idx) == 3:
        ys = pose_3d[:, joint_idx[1]] - pose_3d[:, joint_idx[2]]
    elif len(joint_idx) == 4:
        ys = pose_3d[:, joint_idx[3]] - pose_3d[:, joint_idx[2]]
    return vg.angle(xs, ys)

def calc_common_angles(pose_3d):
    MHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, \
    Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist = range(17)
    
    angles = {}
    #angles['RightKnee'] = np.zeros((pose_3d.shape[0], 3))
    angles['RightKnee'] = get_joint_angles(pose_3d, [RHip, RKnee, RAnkle]) # y
    angles['LeftKnee'] = get_joint_angles(pose_3d, [LHip, LKnee, LAnkle]) # y

    #angles['RightHip'] = get_joint_angles(pose_3d, [Spine, MHip, RKnee, RHip])
    #angles['LeftHip'] = get_joint_angles(pose_3d, [Spine, MHip, LKnee, LHip])
    return angles

