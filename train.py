import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import utils

ap = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
power = pa.gpu
epochs = pa.epochs


trainloader, validationloader, testloader = utils.load_data(where)


model, optimizer, criterion = utils.setup_model(structure,dropout,lr,power)


utils.train_network(model, optimizer, criterion, epochs, 12, trainloader, power)


utils.save_checkpoint(path,structure,dropout,lr)
