from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F

import pdb


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = PartDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    train=False,
    npoints=opt.num_points)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print("Train data: ", len(dataset), "Test data:", len(test_dataset))
num_classes = len(dataset.classes)
print('Classes: ', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

# pdb.set_trace()
# (Pdb) pp classifier
# PointNetCls(
#   (feat): PointFeat(
#     (stn): STN3d(
#       (conv1): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
#       (conv2): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
#       (conv3): Conv1d(128, 1024, kernel_size=(1,), stride=(1,))
#       (fc1): Linear(in_features=1024, out_features=512, bias=True)
#       (fc2): Linear(in_features=512, out_features=256, bias=True)
#       (fc3): Linear(in_features=256, out_features=9, bias=True)
#       (relu): ReLU()
#       (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (bn3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (bn4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (bn5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (conv1): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
#     (conv2): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
#     (conv3): Conv1d(128, 1024, kernel_size=(1,), stride=(1,))
#     (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (bn3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (fc1): Linear(in_features=1024, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=16, bias=True)
#   (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU()
# )


for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        # pdb.set_trace()
        # (Pdb) points.shape, target.shape
        # (torch.Size([32, 2500, 3]), torch.Size([32, 1]))
        # pp target[:, 0]
        # tensor([15, 15, 15,  4,  8,  4,  0, 15, 12,  9, 15,  4, 15, 15, 15,  0,  0, 15,
        #          0,  4,  4,  0, 15, 15,  7,  0,  8,  8, 15, 15,  6,  0])
        points, target = Variable(points), Variable(target[:, 0])
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        # pdb.set_trace()
        # (Pdb) pp points.shape
        # torch.Size([32, 3, 2500])
        #  pp target.shape,target
        # (torch.Size([32]),
        #  tensor([10,  4,  7,  6, 15,  8,  3,  4, 15,  3,  3,  8, 15, 15,  8,  3, 15, 15,
        #          0, 15, 15,  0, 15,  1,  8,  4, 15, 15, 15,  3,  4,  3],
        #        device='cuda:0'))
        # (Pdb) pp pred.shape
        # torch.Size([32, 16])
        # (Pdb) pp pred_choice
        # tensor([10,  4,  7,  6, 15,  8,  3,  4, 15,  3,  3,  8, 15, 15,  8,  3, 15,  9,
        #          0, 15, 15,  0, 15,  1,  8,  4, 15, 14, 15,  3,  4,  2],
        #        device='cuda:0')

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target[:, 0])
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
