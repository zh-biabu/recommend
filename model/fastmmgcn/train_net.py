# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. 
In ACM MM`19,
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric


