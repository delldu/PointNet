from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
from datasets import PartDataset
from pointnet import PointNetDenseCls
import random


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')

opt = parser.parse_args()
print(opt)

d = PartDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    class_choice=['Chair'],
    npoints=2500,
    shuffle=False,
    train=False)

idx = opt.idx
print("model %d/%d" % (idx, len(d)))

point, seg = d[idx]


#resample
choice = [i for i in range(len(seg))]
random.shuffle(choice)
# print("choice:", choice)
point = point[choice, :]
seg = seg[choice]
print("point:", point)

classifier = PointNetDenseCls(k=4)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

point = point.transpose(1, 0).contiguous().unsqueeze_(0)

_, trans = classifier.feat(point)
point = point.transpose(2, 1)
point = torch.bmm(point, trans)
# point = point.transpose(2, 1)

point.squeeze_(0)
point_np = point.detach().numpy()

showpoints(point_np)

