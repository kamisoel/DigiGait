import torch
import imageio
import numpy as np
from time import time

import argparse

from data.video import Video

def get_bboxes(model, video, pred_every=1):
    # read video frame by frame
    bboxes = []

    num_frames = 0
    t = time()

    for i, frame in enumerate(video):
        if i % pred_every != 0:
            bboxes.append(np.full(4, np.nan))
            continue
        results = model(frame, size=320).pandas().xyxy[0]
        box = results.loc[results['confidence'].idxmax()].to_numpy()[:4].astype(int)
        bboxes.append(np.hstack([box[:2], box[2:]-box[:2]])) # (x,y,w,h)
        num_frames += 1

    print(f"{num_frames} frames processed in {(time()-t):.3f}s")
    return bboxes

def render_with_bboxes(video, bboxes, video_out):
    with imageio.get_writer(video_out, fps=video.fps) as writer:
        for i, frame in enumerate(video):
            draw_box(frame, bboxes[i])
            writer.append_data(frame)

def draw_box(img, box, s=5, c=[120,0,0]):
    x1, y1, w, h = box
    x2, y2 = x1+w, y1+h

    img[max(0, y1-s//2) : 1+y2+s//2,
        max(0, x1-s//2) : 1+x1+s//2] = c
    img[max(0, y1-s//2) : 1+y2+s//2,
        max(0, x2-s//2) : 1+x2+s//2] = c
    img[max(0, y1-s//2) : 1+y1+s//2,
        max(0, x1-s//2) : 1+x2+s//2] = c
    img[max(0, y2-s//2) : 1+y2+s//2,
        max(0, x1-s//2) : 1+x2+s//2] = c
    return img

def is_outlier(A, std=3.0):
    return np.abs(A - np.nanmean(A, axis=0)) > std * np.nanstd(A, axis=0)

def interpolate_outliers(A):
    def interp_along_axis(A):    
        from scipy import interpolate
        inds = np.arange(A.shape[0])
        is_good = np.isfinite(A) & (~is_outlier(A))

        if(np.all(is_good)):
            return A
      
        # linearly interpolate and then fill the extremes with the mean
        # (relatively similar to what kalman does)
        f = interpolate.interp1d(inds[is_good], A[is_good], kind="linear", bounds_error=False)
        B = np.where(is_good, A, f(inds))
        B = np.where(np.isfinite(B), B, np.nanmean(B))
        return B

    return np.apply_along_axis(interp_along_axis, arr=A, axis=0)

def detect_person(model_arch, video, bb_out="", video_out="", pred_every=1):
    # Create yolo model
    model = torch.hub.load('ultralytics/yolov5', model_arch, pretrained=True)
    model.classes = [0] # detect persons only 
    model.eval()

    bboxes = get_bboxes(model, video, pred_every)
    num_outliers = np.sum((~np.isfinite(bboxes) | is_outlier(bboxes)))
    print(f"Interpolating {num_outliers} parameters...")
    bboxes = interpolate_outliers(bboxes)
    if video_out:
        print(f"Save Video with added bboxes...")
        render_with_bboxes(video, bboxes, video_out)

    print("Done.")
    if bb_out:
        # save using numpy
        np.save(bb_out, np.vstack(bboxes))

    return bboxes


if __name__ == '__main__':
    #read args
    parser = argparse.ArgumentParser(description='Person detection')
    parser.add_argument('-i', '--input',
                        help='input video file',
                        type=str,
                        default="")
    parser.add_argument('-o', '--output',
                        help='output npz file',
                        type=str,
                        default="bboxes")
    parser.add_argument('-m', '--model',
                        help='specify yolo model (yolov5s, yolov5m, yolov5l, yolov5x)',
                        type=str,
                        default="yolov5s")
    parser.add_argument('-v', '--video_out',
                        help='if given, save video with added bboxes to',
                        type=str,
                        default="")
    parser.add_argument('-s', '--start_time',
                        help='start the video at spefified time in seconds',
                        type=float,
                        default=0)
    parser.add_argument('-l', '--length',
                        help='take only the subclip of specified length starting with <start_time>',
                        type=float,
                        default=-1)
    args = parser.parse_args()

    with Video(args.input) as video:
        start_frame = round(args.start_time * video.fps)
        if args.length > 0 and args.start_time + args.length < video.duration:
            end_frame = start_frame + round(args.length * video.fps)
            subclip = video[start_frame:end_frame]
        else:
            subclip = video[args.start_frame:]

        detect_person(args.model, subclip, args.output, args.video_out)
