import numpy as np
import imageio
import torch
from torch.utils.data import Dataset

from data.bbox_utils import xywh2xyxy 

class VideoDataset(Dataset):
    def __init__(self, path, bboxes=None, video_range=None, transform=None):
        self.path = path
        self.bboxes = bboxes
        self.transform = transform

        # explained in __getitem__
        self._reader = None

        reader = imageio.get_reader(self.path, 'ffmpeg')
        self.size = reader._meta['size']
        self.fps = reader._meta['fps']

        if video_range is None:
            self._start = 0
            self._length = reader.count_frames()
        else:
            start, end = video_range
            self._start = int(start * self.fps)
            self._length = int((end - start) * self.fps)

    def __getitem__(self, ix):
        # Below is a workaround to allow using `VideoDataset` with
        # `torch.utils.data.DataLoader` in multiprocessing mode.
        # `DataLoader` sends copies of the `VideoDataset` object across
        # processes, which sometimes leads to bugs, as `imageio.Reader`
        # does not support being serialized. Since our `__init__` set
        # `self._reader` to None, it is safe to serialize a
        # freshly-initialized `VideoDataset` and then, thanks to the if
        # below, `self._reader` gets initialized independently in each
        # worker thread.

        ix = self._start + ix

        if self._reader is None:
            self._reader = imageio.get_reader(self.path, 'ffmpeg')

        # this is a numpy ndarray in [h, w, channel] format
        frame = self._reader.get_data(ix)

        # cut out the frame part captured by the bbox
        if self.bboxes is not None:
            rect = xywh2xyxy(self.bboxes[ix]).astype(int)
            frame = frame[rect[1]:rect[3], rect[0]:rect[2]]

        # from rgb byte to [0,1]
        frame = frame / 255

        # PyTorch standard layout [channel, h, w]
        frame = torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32))

        # apply transform
        if self.transform is not None:
            frame = self.transform(frame)
        
        if self.bboxes is None:
            return frame
        else:
            return frame, self.bboxes[ix]

    def __len__(self):
        return self._length