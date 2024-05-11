"""
An example for the model class
"""
import torch.nn as nn
import numpy as np
import logging

class HmipT(nn.Module):
    def __init__(self, config, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger

        # define layers        
        self.conv1 = nn.Conv2d(in_channels=5*32, out_channels=self.config.conv1_out, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(21*3, 32)
        self.linear2 = nn.Linear(32*2, 128)
        self.linear3 = nn.Linear(128, 64)

        transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.motion_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.attention = nn.MultiheadAttention(64, 2, batch_first=True)

        self.linear_mask1 = nn.Linear(5 * 64, 128)
        self.linear_mask2 = nn.Linear(128, 32)


    def forward(self, imgs: np.ndarray, poses: np.ndarray):
        self.logger.debug(imgs.shape) # 1, 5, 32, 60, 48
        self.logger.debug(poses.shape) # 1, 5, 2, 21, 3
        bs, tframes, channels, width, height = imgs.shape
        _, _, hands_num, landmarks_num, pos = poses.shape
        
        concat_imgs = imgs.view(bs, -1, width, height)
        imgs = self.conv1(concat_imgs)
        self.logger.debug(f"reduced {imgs.shape}") # 1, 64, 30, 24

        img_features = imgs.view(*imgs.shape[0:2], -1).permute(0, 2, 1)
        self.logger.debug(f"image feature: {img_features.shape}") # 1, 720, 64

        poses = poses.view(bs, tframes, hands_num, -1) # 1, 5, 2, 21*3
        pose_features: np.ndarray = self.linear1(poses)
        self.logger.debug(f"pose feature: {pose_features.shape}") # 1, 5, 2, 32
        
        hand_pose = pose_features.view(bs, tframes, -1) # 1, 5, 64
        hand_pose = self.linear2(hand_pose)
        hand_pose = self.relu(hand_pose)
        hand_pose_feature: np.ndarray = self.linear3(hand_pose)
        self.logger.debug(f"hand pose feature: {hand_pose_feature.shape}") # 1, 5, 64

        motion_feature: np.ndarray = self.motion_encoder(hand_pose_feature)
        self.logger.debug(f"motion feature {motion_feature.shape}") # 1, 5, 64

        mask_detector_features, _ = self.attention(motion_feature, img_features, img_features, need_weights=False)
        self.logger.debug(f"mask detector feature: {mask_detector_features.shape}") # 1, 5, 64

        mask_detector_features = mask_detector_features.reshape(bs, -1) # 1, 320
        detector = self.linear_mask1(mask_detector_features)
        detector = self.relu(detector)
        detector = self.linear_mask2(detector) 
        # self.logger.debug("detector:", detector.shape) # 1, 32
        
        return detector
