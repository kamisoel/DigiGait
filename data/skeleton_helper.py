import numpy as np

def mediapipe2openpose(keypoints):
	op_idx = OpenPoseSkeleton().keypoint2index
	mp_idx = MediaPipeSkeleton().keypoint2index

	op_kpts = np.zeros_like(keypoints)
	op_kpts[op_idx['Nose']] = keypoints[mp_idx['Nose']]
	op_kpts[op_idx['Neck']] = keypoints[mp_idx['']]
	op_kpts[op_idx['RShoulder']] = keypoints[mp_idx['right_shoulder']]
	op_kpts[op_idx['RElbow']] = keypoints[mp_idx['right_elbow']]
	op_kpts[op_idx['RWrist']] = keypoints[mp_idx['right_wrist']]
	op_kpts[op_idx['LShoulder']] = keypoints[mp_idx['left_shoulder']]
	op_kpts[op_idx['LElbow']] = keypoints[mp_idx['left_elbow']]
	op_kpts[op_idx['LWrist']] = keypoints[mp_idx['left_wrist']]
	op_kpts[op_idx['MidHip']] = keypoints[mp_idx['right_hip']]
	op_kpts[op_idx['RHip']] = keypoints[mp_idx['right_hip']] - keypoints[mp_idx['left_hip']]
	op_kpts[op_idx['RKnee']] = keypoints[mp_idx['right_knee']]
	op_kpts[op_idx['RAnkle']] = keypoints[mp_idx['right_ankle']]
	op_kpts[op_idx['LHip']] = keypoints[mp_idx['left_hip']]
	op_kpts[op_idx['LKnee']] = keypoints[mp_idx['left_knee']]
	op_kpts[op_idx['LAnkle']] = keypoints[mp_idx['left_ankle']]
	op_kpts[op_idx['REye']] = keypoints[mp_idx['right_eye']]
	op_kpts[op_idx['LEye']] = keypoints[mp_idx['left_eye']]
	op_kpts[op_idx['REar']] = keypoints[mp_idx['right_ear']]
	op_kpts[op_idx['LEar']] = keypoints[mp_idx['left_ear']]
	op_kpts[op_idx['LBigToe']] = keypoints[mp_idx['left_foot_index']]
	op_kpts[op_idx['LSmallToe']] = keypoints[mp_idx['left_foot_index']]
	op_kpts[op_idx['LHeel']] = keypoints[mp_idx['left_heel']]
	op_kpts[op_idx['RBigToe']] = keypoints[mp_idx['right_foot_index']]
	op_kpts[op_idx['RSmallToe']] = keypoints[mp_idx['right_foot_index']]
	op_kpts[op_idx['RHeel']] = keypoints[mp_idx['right_heel']]

	return op_kpts


class OpenPoseSkeleton(object):

    def __init__(self):
        self.root = 'MidHip'
        self.keypoint2index = {
            'Nose': 0,
            'Neck': 1,
            'RShoulder': 2,
            'RElbow': 3,
            'RWrist': 4,
            'LShoulder': 5,
            'LElbow': 6,
            'LWrist': 7,
            'MidHip': 8,
            'RHip': 9,
            'RKnee': 10,
            'RAnkle': 11,
            'LHip': 12,
            'LKnee': 13,
            'LAnkle': 14,
            'REye': 15,
            'LEye': 16,
            'REar': 17,
            'LEar': 18,
            'LBigToe': 19,
            'LSmallToe': 20,
            'LHeel': 21,
            'RBigToe': 22,
            'RSmallToe': 23,
            'RHeel': 24
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        self.children = {
            'MidHip': ['Neck', 'RHip', 'LHip'],
            'Neck': ['Nose', 'RShoulder', 'LShoulder'],
            'Nose': ['REye', 'LEye'],
            'REye': ['REar'],
            'REar': [],
            'LEye': ['LEar'],
            'LEar': [],
            'RShoulder': ['RElbow'],
            'RElbow': ['RWrist'],
            'RWrist': [],
            'LShoulder': ['LElbow'],
            'LElbow': ['LWrist'],
            'LWrist': [],
            'RHip': ['RKnee'],
            'RKnee': ['RAnkle'],
            'RAnkle': ['RBigToe', 'RSmallToe', 'RHeel'],
            'RBigToe': [],
            'RSmallToe': [],
            'RHeel': [],
            'LHip': ['LKnee'],
            'LKnee': ['LAnkle'],
            'LAnkle': ['LBigToe', 'LSmallToe', 'LHeel'],
            'LBigToe': [],
            'LSmallToe': [],
            'LHeel': [],
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent


class MediaPipeSkeleton(object):

    def __init__(self):
        self.root = 'Nose'
        self.keypoint2index = {
            'Nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32,
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)
