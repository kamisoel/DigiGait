import abc

class Estimator2D(object):
    """Base class of 2D human pose estimator."""

    def __init__(self):
        pass

    @abc.abstractclassmethod
    def estimate(self, video, bboxes=None):
        """
        Args:
            video: Array of images (N, BGR, H, W)
            bboxest: Array of bounding-box (left_top x, left_top y, 
                bbox_width, bbox_height).
        Return:
            keypoints: Array of 2d-keypoint position with confidence levels 
            (n_joints, 3)
            meta: VideoPose3D-compatible metadata object
        """
        pass