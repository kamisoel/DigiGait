import torch
from torch import nn
import numpy as np
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from data.timeseries_utils import interp_along_time, minmax_scale

class GaitEventModel(nn.Module):
    def __init__(self, d_in, d_out=1, hidden_dim=128, n_layers=3,  
                 dropout=0.25, bidirectional=False):
        super(GaitEventModel, self).__init__()
        self.gru = nn.GRU(
            d_in, hidden_dim, n_layers, batch_first=True, 
            dropout=dropout, bidirectional=bidirectional
        )
        if bidirectional:
            self.fc = nn.Linear(2*hidden_dim, d_out)
        else:
            self.fc = nn.Linear(hidden_dim, d_out)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x) # (batch_size, seq_length, hidden_size)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

    def predict(self, x, threshold=0.1):
        self.eval()
        x = torch.Tensor(x)
        if x.dim() == 2:
            x = x.view(1, x.shape[0], x.shape[1])
        y_hat = torch.sigmoid(self.forward(x))
        y_hat = y_hat.detach().cpu().numpy()
        y_hat[y_hat < threshold] = 0
        # TODO: peakdetection

        return y_hat

    def predict_from_pose(self, pose, threshold=0.1):
        MHIP, RHIP, RKNEE, RHEEL, LHIP, LKNEE, LHEEL = range(7)

        def _debias(pose):
            pose = pose.copy()
            pose[:, RHIP:] = pose[:, RHIP:] - pose[:, [MHIP]]  # all joints relative to pelvis
            pose[:, MHIP] = pose[:, MHIP] - pose[[0], MHIP]    # pelvis relative to first frame

            femur_len_r = np.linalg.norm(pose[:, RHIP] - pose[:, RKNEE], axis=-1)
            femur_len_l = np.linalg.norm(pose[:, LHIP] - pose[:, LKNEE], axis=-1)
            femur_len = (femur_len_r + femur_len_l) / 2 # avg femur length in each frame
            pose = pose / femur_len[:, None, None]      # normalize each frame
            return pose

        def _create_features(pose, hip, knee, foot):
            features = np.zeros((len(pose), 6, 3))
            features[:, 0] = pose[:, hip]
            features[:, 1] = pose[:, knee]
            features[:, 2] = pose[:, foot]
            features[:, 3:] = np.gradient(features[:, :3], axis=0)
            features = lp_filter(features, 6)
            features = features.reshape((len(pose), -1))
            return features

        pose = interp_along_time(_debias(pose), 50, 100)
        rfeature = minmax_scale(_create_features(pose, self.rhip, self.rknee, self.rfoot))
        lfeature = minmax_scale(_create_features(pose, self.lhip, self.lknee, self.lfoot))

        rpredict = model.predict(rfeature, 0)
        rhs = rpredict[:, 0].nonzero()
        rto = rpredict[:, 1].nonzero()

        lpredict = model.predict(lfeature, 0)
        lhs = lpredict[:, 0].nonzero()
        lto = lpredict[:, 1].nonzero()

        return rhs, lhs, rto, lto


    @staticmethod
    def load_pretrained(device='cpu'):
        PRETRAINED_GID = '1WaA6JlarrVvN4kQtXtdRA3JSXQMaIiRL'
        CKPT_FILE = Path('model/checkpoints/gait_event_model.pth')
        #SCALE_FILE = Path('model/checkpoints/gait_event_min_scale.npy')

        if not CKPT_FILE.exists():
            # download pretrained weights
            gdd.download_file_from_google_drive(PRETRAINED_GID, CKPT_FILE)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = GaitEventModel(18, 2, hidden_dim=128, n_layers=2, dropout=0.15, bidirectional=True)
        model.to(device)
        model.load_state_dict(torch.load(CKPT_FILE, map_location=device))

        #min_, scale_ = np.load(SCALE_FILE)
        #scale_func = lambda x: x * scale_ + min_

        return model #, scale_func

