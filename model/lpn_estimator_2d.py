from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize

from model import lpn
from model.estimator_2d import Estimator2D
from data.video_dataset import VideoDataset
from data.bbox_utils import xywh2cs, adjust_aspect_ratio
from data.data_utils import suggest_metadata

class LPN_Estimator2D(Estimator2D):
    """2D human pose estimator using lpn-pytorch (https://github.com/zhang943/lpn-pytorch)"""

    PRETRAINED_GID = '1dldLwjOacXV_uGkbxfEIPPJEK_2A-Snp'
    CFG_FILE = 'model/configs/lpn50_256x192_gd256x2_gc.yaml'
    CKPT_FILE = 'model/checkpoints/lpn_50_256x192.pth'

    def __init__(self, device='cpu'):
        self.device = device
        # download pretrained weights if necessary
        if not Path(self.CKPT_FILE).exists():
            self.download_weights()

        # load pretrained lpn pose network
        self.model, self.cfg = self.create_lpn_model(
                                    Path(self.CFG_FILE).resolve(), 
                                    Path(self.CKPT_FILE).resolve())


    def download_weights(self):
        try:
            from google_drive_downloader import GoogleDriveDownloader as gdd
            gdd.download_file_from_google_drive(self.PRETRAINED_GID, self.CKPT_FILE)
        except ImportError as error:
            print('GoogleDriveDownloader has to be installed for automatic download' \
                'You can download the weights manually under: https://drive.google.com/file/d/1dldLwjOacXV_uGkbxfEIPPJEK_2A-Snp/view?usp=sharing')



    def create_lpn_model(self, cfg_file, ckp_file):
        # create Configs
        args = Namespace(cfg = cfg_file, modelDir='', logDir='')
        lpn.update_config(lpn.cfg, args)

        # Use cfg to create model
        pose_model = lpn.get_pose_net(lpn.cfg, is_train=False)

        # load pretrained weights
        if torch.cuda.is_available():
            pose_model = pose_model.cuda()
            pose_model.load_state_dict(torch.load(ckp_file), strict=False)
        else:
            checkpoint = torch.load(ckp_file, map_location=torch.device(self.device))
            pose_model.load_state_dict(checkpoint, strict=False)

        pose_model.eval()

        return pose_model, lpn.cfg


    def estimate(self, video_file, bboxes=None):

        # Convert bboxes to correct aspect ratio
        bboxes = np.apply_along_axis(adjust_aspect_ratio, 1, bboxes, aspect_ratio=3/4)

        # Create Preprocessing Pipeline for Video Frame
        transform = Compose([
            Resize((256, 192)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Create Dataset and DataLoader
        BATCH_SIZE = 64
        dataset = VideoDataset(video_file, bboxes, transform = transform)
        dl = DataLoader(dataset, batch_size=BATCH_SIZE)

        # Infer poses using the model
        with torch.no_grad():
            self.model.eval()
            pose_2d = []
            for i, (frame, bbox) in enumerate(dl):
                frame = frame.to(self.device)
                batch_heatmap = self.model(frame).cpu().numpy()

                c, s = xywh2cs(bbox.numpy())

                preds, maxvals = lpn.get_final_preds(self.cfg, batch_heatmap, c, s)
                pose_2d.append(preds)
            pose_2d = np.vstack(pose_2d)

        # create VideoPose3D-compatible metadata and keypoint structure
        video_name = Path(video_file).stem
        metadata = suggest_metadata('coco')
        video_meta = {'w': dataset.size[0], 'h': dataset.size[1], 'fps': dataset.fps}
        metadata['video_metadata'] = {video_name: video_meta}
        keypoints = {video_name: {'custom': [pose_2d]}}

        return keypoints, metadata


