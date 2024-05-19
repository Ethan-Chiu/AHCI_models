"""
An example for the model class
"""
import torch
import torch.nn as nn
import numpy as np
import logging

class HmipT(nn.Module):
    def __init__(self, config, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger

        # define layers 
        self.relu = nn.ReLU()

        # Image       
        self.img_conv = nn.Conv2d(in_channels=5*32, out_channels=self.config.conv1_out, kernel_size=3, stride=2, padding=1)

        # Hand-Landmark
        self.landmark_embed_dim = 16
        self.landmark_linear_1 = nn.Linear(14, 64)
        self.landmark_linear_2 = nn.Linear(64, self.landmark_embed_dim)

        # Hand-pose
        self.pose_linear_1 = nn.Linear(24*self.landmark_embed_dim, 512)
        self.pose_linear_2 = nn.Linear(512, 256)
        self.pose_linear_3 = nn.Linear(256, 64)

        # Head
        self.head_linear_1 = nn.Linear(7, 16)
        self.head_linear_2 = nn.Linear(16, 32)

        # Join-pose
        self.join_linear_1 = nn.Linear(64+32, 128)
        self.join_linear_2 = nn.Linear(128, 64)

        # Motion
        transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.motion_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        # Cross attention
        self.attention_1 = nn.MultiheadAttention(64, 2, batch_first=True)
        self.norm_1 = nn.LayerNorm(64)
        self.attention_2 = nn.MultiheadAttention(64, 2, batch_first=True)
        self.norm_2 = nn.LayerNorm(64)

        # Detector
        self.mask_linear_1 = nn.Linear(5 * 64, 128)
        self.mask_linear_2 = nn.Linear(128, 32)


    def forward(self, imgs: np.ndarray, poses: np.ndarray, heads: np.ndarray):
        self.logger.debug(imgs.shape) # 1, 5, 32, 60, 48
        self.logger.debug(poses.shape) # 1, 5, 24, 14
        self.logger.debug(heads.shape) # 1, 5, 7
        bs, tframes, channels, width, height = imgs.shape
        _, _, landmarks_num, pos = poses.shape
        
        # Image
        concat_imgs = imgs.view(bs, -1, width, height)
        imgs = self.img_conv(concat_imgs)
        self.logger.debug(f"reduced {imgs.shape}") # 1, 64, w/2, h/2

        img_features = imgs.view(*imgs.shape[0:2], -1).permute(0, 2, 1)
        self.logger.debug(f"image feature: {img_features.shape}") # 1, 720, 64

        # Hand-Landmark
        landmark = self.landmark_linear_1(poses)
        landmark = self.relu(landmark)
        landmark = self.landmark_linear_2(landmark)
        landmark: np.ndarray = self.relu(landmark)
        self.logger.debug(f"landmark feature: {landmark.shape}")

        # Hand-Pose
        hand_pose = landmark.view(bs, tframes, -1) # 1, 5, 24*16
        hand_pose = self.pose_linear_1(hand_pose)
        hand_pose = self.relu(hand_pose)
        hand_pose = self.pose_linear_2(hand_pose)
        hand_pose = self.relu(hand_pose)
        hand_pose = self.pose_linear_3(hand_pose)
        hand_pose: np.ndarray = self.relu(hand_pose)
        self.logger.debug(f"hand pose feature: {hand_pose.shape}") # 1, 5, 64

        # Head
        head_pose = self.head_linear_1(heads)
        head_pose = self.relu(head_pose)
        head_pose = self.head_linear_2(head_pose)
        head_pose: np.ndarray = self.relu(head_pose)
        self.logger.debug(f"head pose feature: {head_pose.shape}") # 1, 5, 32

        # Join-pose
        join_pose: np.ndarray = torch.cat((hand_pose, head_pose), dim=-1)
        self.logger.debug(f"join pose feature: {join_pose.shape}") # 1, 5, 96

        join_pose = self.join_linear_1(join_pose)
        join_pose = self.relu(join_pose)
        join_pose = self.join_linear_2(join_pose)
        join_pose: np.ndarray = self.relu(join_pose)
        self.logger.debug(f"join pose feature: {join_pose.shape}") # 1, 5, 64

        # Motion
        motion_feature: np.ndarray = self.motion_encoder(join_pose)
        self.logger.debug(f"motion feature {motion_feature.shape}") # 1, 5, 64

        # Cross attention
        dtc_feat, _ = self.attention_1(motion_feature, img_features, img_features, need_weights=False)
        dtc_q = self.norm_1(dtc_feat + motion_feature)

        # detector_feat, _ = self.attention_2(dtc_q, img_features, img_features, need_weights=False)
        # detector_feat = self.norm_2(detector_feat + dtc_q)
        detector_feat = dtc_q
        self.logger.debug(f"mask detector feature: {detector_feat.shape}") # 1, 5, 64

        detector_feat = detector_feat.reshape(bs, -1) # 1, 320
        detector = self.mask_linear_1(detector_feat)
        detector = self.relu(detector)
        detector = self.mask_linear_2(detector) 
        self.logger.debug(f"detector: {detector.shape}") # 1, 32
        
        return detector
