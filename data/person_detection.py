
import sys
import torch
import imageio
import numpy as np
from time import time

import argparse 

def get_bboxes(model, video_in):
    # read video frame by frame
    bboxes = []
    
    with imageio.get_reader(video_in, 'ffmpeg') as video:
        num_frames = video.get_meta_data()['nframes']

        t = time()
        for frame in video.iter_data():
            results = model(frame, size=320).pandas().xyxy[0]
            box = results.loc[results['confidence'].idxmax()].to_numpy()[:4].astype(int)
            bboxes.append(np.hstack([box[:2], box[2:]-box[:2]])) # (x,y,w,h)

        print(f"{num_frames} frames processed in {(time()-t):.3f}s")
    return bboxes

def render_with_bboxes(video_in, bboxes, video_out):
    with imageio.get_reader(video_in, 'ffmpeg') as video:
        fps = video.get_meta_data()['fps']
        with imageio.get_writer(video_out, fps=fps) as writer:
            for i, frame in enumerate(video.iter_data()):
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

def detect_person(model_arch, input, output="", video_out=""):
    # Create yolo model
    model = torch.hub.load('ultralytics/yolov5', model_arch, pretrained=True)
    model.classes = [0] # detect persons only 
    model.eval()

    bboxes = get_bboxes(model, input)
    num_outliers = np.sum(~np.isfinite(bboxes) & is_outlier(bboxes))
    print(f"Interpolating {num_outliers} frames...")
    bboxes = interpolate_outliers(bboxes)
    if video_out:
        print(f"Save Video with added bboxes...")
        render_with_bboxes(input, bboxes, video_out)

    print("Done.")
    if output:
        # save using numpy
        np.save(output, np.vstack(bboxes))

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
    args = parser.parse_args()

    detect_person(args.model, args.input, args.output, args.video_out)
