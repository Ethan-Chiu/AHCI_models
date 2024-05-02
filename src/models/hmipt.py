"""
An example for the model class
"""
import torch.nn as nn

from .yolov9.models.common import DetectMultiBackend


class HmipT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define layers
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=self.config.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        # yolo
        weights_path = './pretrained_weights/gelan-c-seg.pt'
        self.yolo = DetectMultiBackend(weights_path, data='./src/models/yolov9/data/coco.yaml', fp16=False)
        self.yolo.warmup(imgsz=(1, 3, 640, 640))

    def forward(self, x):
        # x.shape (bs, 3, 256, 256)
        im = x / 255
        pred, proto = self.yolo(im)[:2]
        # print("pred shape", pred)
        # print("proto shape", proto)

        x = self.conv(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        fnn = nn.Linear(x.size(1), 2).cuda()
        x = fnn(x)

        out = x
        # out.shape (bs, 2)
        return out
