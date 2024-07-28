# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline



# Deep Learning
import  torch
import  torch.nn            as nn
import  torch.nn.functional as F
from    torch.optim.optimizer import Optimizer
from    torch.optim.lr_scheduler import LRScheduler
from    torch.utils.data import DataLoader
from    torch.utils.tensorboard import SummaryWriter

from    torchmetrics.classification import MulticlassAccuracy
from    torchmetrics.regression import R2Score

import  torchvision
from    torchvision.transforms import v2 as TorchVisionTrns

import torchinfo

# Improve performance by benchmarking
torch.backends.cudnn.benchmark = True

# Miscellaneous
import copy
from enum import auto, Enum, unique
import math
import os
from platform import python_version
import random
import time

# Typing
from typing import Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Jupyter
from IPython import get_ipython
from IPython.display import HTML, Image
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, interact, IntSlider, Layout, SelectionSlider
from ipywidgets import interact