import torch
from torch import nn
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd

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


    @staticmethod
    def load_pretrained(device='cpu'):
        PRETRAINED_GID = '1WaA6JlarrVvN4kQtXtdRA3JSXQMaIiRL'
        CKPT_FILE = Path('model/checkpoint/gait_event_rnn.pth')

        if not CKPT_FILE.exists():
            # download pretrained weights
            gdd.download_file_from_google_drive(self.PRETRAINED_GID, self.CKPT_FILE)

        model = GaitEventModel(24, 4, hidden_dim=128, n_layers=2, dropout=0.15, bidirectional=True)
        model.load_state_dict(torch.load(CKPT_FILE)) #TODO load to device

        return model

