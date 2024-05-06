"""
An example for the model class
"""
import torch.nn as nn


class HmipT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define layers
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=self.config.num_filters, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        print(x.shape)
        # x.shape (bs, 3, 256, 256)
        
        x = self.conv(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        fnn = nn.Linear(x.size(1), 2).cuda()
        x = fnn(x)

        out = x
        # out.shape (bs, 2)
        return out
