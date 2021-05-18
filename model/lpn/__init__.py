# ------------------------------------------------------------------------------
# Adapted from:
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from .config import _C as cfg
from .config import update_config
from .lpn import LPN, get_pose_net
from .inference import get_final_preds

