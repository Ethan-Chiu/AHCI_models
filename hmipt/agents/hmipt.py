import os
import shutil
import logging
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
import torch.optim as optim
import torch.nn.init as init

from .base import BaseAgent

from hmipt.src.models.hmipt import HmipT
from hmipt.datasets.hmip import HmipDataLoader
from hmipt.utils.misc import print_cuda_statistics
from hmipt.src.models.yolov9.models.common import DetectMultiBackend


# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)


cudnn.benchmark = True


class HimpTAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.logger.setLevel(logging.INFO)

        # define models
        self.hmipt = HmipT(config=config, logger=self.logger) 

        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # self.hmipt.apply(initialize_weights)

        # define data_loader
        self.data_loader = HmipDataLoader(config=config) 

        # define loss
        self.loss = nn.MSELoss() 

        # define optimizers for both generator and discriminator
        print(self.config.learning_rate)
        self.optimizer = optim.SGD(self.hmipt.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
 
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        torch.cuda.manual_seed_all(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        if self.is_cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.hmipt = self.hmipt.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        
        # Summary Writer
        self.summary_writer = None

        weights_path = "./pretrained_weights/gelan-c-seg.pt"
        self.yolo = DetectMultiBackend(
            weights_path, data="./src/models/yolov9/data/coco.yaml", fp16=False
        ).eval()
        self.yolo.warmup(imgsz=(1, 3, 640, 640))
        self.pooling_layer = nn.AvgPool2d(kernel_size=2).cuda()


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if file_name is None:
            return
        
        try:
            filepath = os.path.join(self.config.checkpoint_dir, file_name)

            self.logger.info("Loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.hmipt.load_state_dict(checkpoint['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        # Save model checkpoint
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.hmipt.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        # Save the state
        save_path = os.path.join(self.config.checkpoint_dir, file_name)
        self.logger.info(f"Saving checkpoint to {save_path}")

        torch.save(state, save_path)
        # print("State", state['state_dict'])

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            best_save_path = os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar')
            shutil.copyfile(save_path, best_save_path)

    def run(self):
        """
        The main operator
        :return:
        """
        print("Agent started running")
        try:
            if self.config.mode == "train":
                self.train()
            elif self.config.mode == "test":
                self.test()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
            self.current_epoch = epoch
            print(f"Epoch {epoch}/{self.config.max_epoch}")
            self.train_one_epoch()
            # loss = self.validate()

            # is_best = loss < self.best_metric
            # if is_best:
            #     self.best_metric = loss

            self.save_checkpoint(f"checkpoint_{epoch}.pth.tar", is_best=False)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.hmipt.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            imgs_raw, poses, heads = data

            # NOTE: test
            # if batch_idx != 0:
            #     continue
            # if batch_idx == 0:
            #     print(self.hmipt.state_dict())

            with torch.no_grad():
                img_shape = imgs_raw.shape
                imgs_raw = imgs_raw.reshape((img_shape[0] * img_shape[1], *img_shape[2:]))
                imgs_raw = imgs_raw.cuda()
                
                _, proto = self.yolo(imgs_raw, augment=False, visualize=False)[:2]
                proto: np.ndarray = proto[-1]

                # NOTE: test
                # test = proto[0]
                # torch.save(test, "./test/proto_test.pt")
                # print(paths)
                # print('test test test')
                # break

                pooled: np.ndarray = self.pooling_layer(proto)

                pooled_proto_shape = pooled.shape
                pooled = pooled.reshape((img_shape[0], img_shape[1], *pooled_proto_shape[1:]))

            imgs = pooled.detach()

            imgs, poses, heads, target = imgs.to(self.device), poses.to(self.device), heads.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.hmipt(imgs, poses, heads)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(target), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.hmipt.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader.valid_loader):
                imgs, poses, heads = data

                imgs, poses, heads, target = imgs.to(self.device), poses.to(self.device), heads.to(self.device), target.to(self.device)
                output = self.hmipt(imgs, poses, heads)
                val_loss += self.loss(output, target).item()  # sum up batch loss

        val_loss /= len(self.data_loader.valid_loader.dataset)
        self.logger.info('Val set: Average loss: {:.4f})\n'.format(
            val_loss
        )) 
        return val_loss
    

    def test(self):
        self.logger.info("Start testing...")
        self.hmipt.eval()
        test_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader.test_loader):
                imgs, poses, heads = data

                imgs, poses, heads, target = imgs.to(self.device), poses.to(self.device), heads.to(self.device), target.to(self.device)
                output = self.hmipt(imgs, poses, heads)
                test_loss += self.loss(output, target).item() 

        test_loss /= len(self.data_loader.test_loader)
        self.logger.info('Test set: Average loss: {:.4f})\n'.format(
            test_loss
        )) 
        return test_loss


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint("checkpoint_fianl.pth.tar")
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
