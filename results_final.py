import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from models.ideal import IdealAddModule, IdealSubModule, IdealMemModule
from tasks.module import AddTask
import numpy as np

