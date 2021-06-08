import imageio
from slicerator import Slicerator 

@Slicerator.from_class
class Video(object):

    propagate_attrs = ['fps', 'duration', 'size', 'metadata']

    def __init__(self, video):
        self._video = video
        self._reader = imageio.get_reader(video, 'ffmpeg')
        self._meta = self._reader.get_meta_data()
        self._nframes = int(self._meta['duration'] * self._meta['fps'])

    @property
    def metadata(self):
        return self._meta

    @property
    def duration(self):
        return self._meta['duration']

    @property
    def fps(self):
        return self._meta['fps']

    @property
    def size(self):
        w,h = self._meta['size'] #is this w,h or h,w ??
        return w, h

    def close(self):
        self._reader.close()

    def __getitem__(self, i):
        return self._reader.get_data(i)

    def __len__(self):
        return self._nframes
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()