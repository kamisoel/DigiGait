import numpy as np

def adjust_aspect_ratio(bbox, aspect_ratio=3/4):
    x,y,w,h = bbox
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return np.array([x,y,w,h])


def xyxy2xywh(bbox):
    return np.hstack([bbox[..., :2], bbox[..., 2:]-bbox[..., :2]])

def xywh2xyxy(bbox):
    return np.hstack([bbox[..., :2], bbox[..., 2:]+bbox[..., :2]])

def xywh2cs(bbox, pixel_std=200):
    center = bbox[..., :2] + bbox[..., 2:] * 0.5
    scale = bbox[..., 2:] / pixel_std

    return center, scale

def cs2xywh2(c, s, pixel_std=200):
    bbox = np.zeros((len(c),4)) if c.ndim > 1 else np.zeros(4)
    bbox[..., 2:] = s[..., :] * pixel_std
    bbox[..., :2] = c[..., :] - bbox[..., 2:] / 2
    return bbox