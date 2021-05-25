from pathlib import Path
from urllib import request
from urllib.error import URLError
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize

from data.data_utils import suggest_metadata
from common import camera
from common.model import TemporalModel
from model.estimator_3d import Estimator3D

class VideoPose3D (Estimator3D):
    """3D human pose estimator usign VideoPose3D"""

    CFG_FILE = "model/configs/videopose.yaml"
    CKPT_FILE = "model/checkpoints/pretrained_h36m_detectron_coco.bin"

    def __init__(self):
        if not Path(self.CKPT_FILE).exists():
            self.download_weights()

        with Path(self.CFG_FILE).open("r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        self.model = self.create_model(cfg, self.CKPT_FILE)
        self.causal = cfg['MODEL']['causal']

    def download_weights(self):
        weight_url = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
        try:
            url_request = request.urlopen(weight_url)
            path = Path(self.CKPT_FILE)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(url_request.read())
        except URLError:
            print("Could not download weight file. Please check your internet \
                connection and proxy settings")

    def post_process(self, pose_3d):
        #transform to world coordinates
        rot = np.array([0.1407056450843811, -0.1500701755285263, 
                        -0.755240797996521, 0.6223280429840088], dtype='float32')
        pose_3d = camera.camera_to_world(pose_3d, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        pose_3d[:, :, 2] -= np.min(pose_3d[:, :, 2])
        return pose_3d

    def create_model(self, cfg, checkpoint):        
        # specify models hyperparameters - loaded from config yaml
        model_params = cfg['MODEL']
        filter_widths = model_params['filter_widths'] #[3,3,3,3,3]
        dropout = model_params['dropout'] #0.25
        channels = model_params['channels'] #1024
        causal = model_params['causal'] #False

        n_joints_in = cfg['IN_FORMAT']['num_joints']
        n_joints_out = cfg['OUT_FORMAT']['num_joints']

        # create model and load checkpoint
        model_pos = TemporalModel(n_joints_in, 2, n_joints_out, filter_widths, 
                                  causal, dropout, channels)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'])
        model_pos.eval() # Important for dropout!

        # push to gpu
        if torch.cuda.is_available():
            model_pos = model_pos.cuda()
        model_pos.eval()

        return model_pos

    def post_process(self, pose_3d):
        pose_3d = np.ascontiguousarray(pose_3d)
        #transform to world coordinates
        rot = np.array([0.1407056450843811, -0.1500701755285263, 
                        -0.755240797996521, 0.6223280429840088], dtype='float32')
        pose_3d = camera.camera_to_world(pose_3d, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        pose_3d[:, :, 2] -= np.min(pose_3d[:, :, 2])
        return pose_3d

    def estimate(self, keypoints, meta):
        pad = (self.model.receptive_field() - 1) // 2 # Padding on each side
        causal_shift = pad if self.causal else 0

        predictions = {}
        for video in keypoints:
            kps = keypoints[video]['custom'][0]
            # Normalize camera frames to image size
            res = meta['video_metadata'][video]
            kps[..., :2] = camera.normalize_screen_coordinates(kps[..., :2], 
                                                               w=res['w'], h=res['h'])
            # Pad keypoints with edge mode
            kps = np.expand_dims(np.pad(kps, ((pad + causal_shift, pad - causal_shift), 
                                              (0, 0), (0, 0)), 'edge'), axis=0)

            # Run model
            with torch.no_grad():
              kps = torch.from_numpy(kps.astype('float32'))
              if torch.cuda.is_available():
                  kps = kps.cuda()
              predicted_3d_pos = self.model(kps).squeeze(0).detach().numpy()

              predictions[video] = self.post_process(predicted_3d_pos)

        return predictions
        