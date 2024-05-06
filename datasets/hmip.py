import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import PIL
from PIL import Image
import os
import scipy.io as sio
import numpy as np
import torchvision.transforms as transforms
import csv
from src.models.yolov9.models.common import DetectMultiBackend
from pathlib import Path


class HmiptDataset(Dataset):
    def __init__(self, mode, data_csv, data_root, mp_data_root, transform=None):
        self.data = self._load_csv(data_csv)
        self.mode = mode
        self.data_root = data_root
        self.mp_data_root = mp_data_root
        self.data_cache_dir = "./data/hmip/cache"
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)

        self.transform = transform

        weights_path = "./pretrained_weights/gelan-c-seg.pt"
        self.yolo = DetectMultiBackend(
            weights_path, data="./src/models/yolov9/data/coco.yaml", fp16=False
        )
        # TODO: check if cuda is available
        self.yolo.warmup(imgsz=(1, 3, 640, 640))
        self.pooling_layer = nn.AvgPool2d(kernel_size=2).cuda()


    def _load_csv(self, csv_filepath: str):
        data = []
        with open(csv_filepath, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                detector_str = row["detector"]
                detector = detector_str.split(" ")
                detector = [float(d) for d in detector]
                data.append(
                    (
                        [
                            row["img_path_1"],
                            row["img_path_2"],
                            row["img_path_3"],
                            row["img_path_4"],
                            row["img_path_5"],
                        ],
                        [
                            row["mp_path_1"],
                            row["mp_path_2"],
                            row["mp_path_3"],
                            row["mp_path_4"],
                            row["mp_path_5"],
                        ],
                        detector,
                    )
                )
        if len(data) == 0:
            raise RuntimeError("Found 0 images, please check the data set")
        return data
    

    def _save_proto(self, proto, filename):
        directory = self.data_cache_dir
        files = os.listdir(directory)
        if len(files) >= 20:
            oldest_file = min(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
            os.remove(os.path.join(directory, oldest_file))
        # Save your array
        np.save(os.path.join(directory, filename), proto.cpu())


    def __getitem__(self, index):
        
        img_paths, mp_paths, detector = self.data[index]

        mp = []
        for mp_path in mp_paths:
            mp_filepath = os.path.join(self.mp_data_root, mp_path)
            mp.append(np.load(mp_filepath))

        imgs = []
        for img_path in img_paths:
            cache_name = img_path.replace("/", "_").replace(".jpg", "") + ".npy"
            cache_proto = os.path.join(self.data_cache_dir, cache_name)
            if os.path.exists(cache_proto):
                proto = np.load(cache_proto)
                imgs.append(proto)
                continue
            
            img_filepath = os.path.join(self.data_root, img_path)
            img = Image.open(img_filepath).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            img = img / 255
            img = img.unsqueeze(0).cuda()
            _, proto = self.yolo(img)[:2]
            proto = proto[-1]
            # print(proto.shape)
            pooled_proto = self.pooling_layer(proto)
            self._save_proto(pooled_proto, cache_name)
            
            imgs.append(proto)

        return (imgs, mp), detector

    def __len__(self):
        return len(self.data)


class HmipDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ["train", "test", "random"]

        mean_std = ([128.0, 128.0, 128.0], [1.0, 1.0, 1.0])

        self.input_transform = transforms.Compose(
            [
                transforms.Resize((480, 384), interpolation=PIL.Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std),
            ]
        )

        if self.config.mode == "random":
            train_data = torch.randn(
                self.config.batch_size,
                self.config.input_channels,
                self.config.img_size,
                self.config.img_size,
            )
            train_labels = torch.ones(
                self.config.batch_size, self.config.img_size, self.config.img_size
            ).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (
                self.len_train_data + self.config.batch_size - 1
            ) // self.config.batch_size
            self.valid_iterations = (
                self.len_valid_data + self.config.batch_size - 1
            ) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(
                train, batch_size=config.batch_size, shuffle=True
            )
            self.valid_loader = DataLoader(
                valid, batch_size=config.batch_size, shuffle=False
            )

        elif self.config.mode == "train":
            train_set = HmiptDataset(
                "train",
                self.config.data_csv,
                self.config.data_root,
                mp_data_root=self.config.mp_data_root,
                transform=self.input_transform,
            )
            valid_set = HmiptDataset(
                "val",
                self.config.data_csv,
                self.config.data_root,
                mp_data_root=self.config.mp_data_root,
                transform=self.input_transform,
            )

            self.train_loader = DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.data_loader_workers,
                pin_memory=self.config.pin_memory,
            )
            self.valid_loader = DataLoader(
                valid_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                pin_memory=self.config.pin_memory,
            )
            self.train_iterations = (
                len(train_set) + self.config.batch_size
            ) // self.config.batch_size
            self.valid_iterations = (
                len(valid_set) + self.config.batch_size
            ) // self.config.batch_size

        elif self.config.mode == "test":
            test_set = HmiptDataset(
                "test", self.config.data_root, transform=self.input_transform
            )

            self.test_loader = DataLoader(
                test_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.data_loader_workers,
                pin_memory=self.config.pin_memory,
            )
            self.test_iterations = (
                len(test_set) + self.config.batch_size
            ) // self.config.batch_size

        else:
            raise Exception("Please choose a proper mode for data loading")

    def finalize(self):
        pass
