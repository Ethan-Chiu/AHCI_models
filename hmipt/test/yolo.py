import sys
import numpy as np
from PIL import Image
import os
import platform
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import torch
import io
from functools import reduce
import logging
import torch.nn as nn
from hmipt.utils.config import get_config_from_json
from hmipt.src.models.hmipt import HmipT
from torchvision import transforms
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append('./yolov9')

from hmipt.src.models.yolov9.models.common import DetectMultiBackend
from hmipt.src.models.yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from hmipt.src.models.yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from hmipt.src.models.yolov9.utils.plots import Annotator, colors, save_one_box
from hmipt.src.models.yolov9.utils.segment.general import masks2segments, process_mask
from hmipt.src.models.yolov9.utils.torch_utils import select_device, smart_inference_mode
from hmipt.src.models.yolov9.utils.augmentations import letterbox
from PIL import Image


def process_image():
    im0 = np.array(Image.open("../data/frames/data2/bottle/bottled_13/frame_000010.jpg"))
    im = letterbox(im0, imgsz)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = model(im, augment=augment, visualize=False)[:2]
    torch.save(proto[-1][0], "proto_my.pt")
    exit()

if __name__ == "__main__":
    weights='../pretrained_weights/gelan-c-seg.pt'
    data= '../src/models/yolov9/data/coco.yaml'
    imgsz=(480,480)
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    device='0'
    augment=False
    half=False
    dnn=False
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    result = process_image()