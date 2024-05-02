import os
import random
import shutil

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

# import your classes here
from src.models.hmipt import HmipT
from datasets.hmip import HmipDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class HimpTAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = HmipT(config=config) 

        # define data_loader
        self.data_loader = HmipDataLoader(config=config) 

        # define loss
        self.loss = nn.MSELoss() 

        # define optimizers for both generator and discriminator
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
 

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

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
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
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
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        # Save the state
        save_path = os.path.join(self.config.checkpoint_dir, file_name)
        self.logger.info(f"Saving checkpoint to {save_path}")

        torch.save(state, save_path)

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            best_save_path = os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar')
            shutil.copyfile(save_path, best_save_path)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

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
            loss = self.validate()

            is_best = loss < self.best_metric
            if is_best:
                self.best_metric = loss

            self.save_checkpoint(f"checkpoint_{epoch}.pth.tar", is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            # print("data", data)
            # print("target", target)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        val_loss = 0
        # correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss(output, target).item()  # sum up batch loss
                # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.data_loader.valid_loader.dataset)
        self.logger.info('Val set: Average loss: {:.4f})\n'.format(
            val_loss
        )) 
        return val_loss

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint("checkpoint_fianl.pth.tar")
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
