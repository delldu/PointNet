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
from pointnet import PointNetDenseCls
import torch.nn.functional as F

import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=16, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')


opt = parser.parse_args()
opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print(opt)


shape_name = 'Airplane'
dataset = PartDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=False,
    class_choice=[shape_name]
    )
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = PartDataset(
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    classification=False,
    class_choice=[shape_name],
    train=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print("Train data:", len(dataset), "Test data:", len(test_dataset))
num_classes = dataset.num_seg_classes
print('Classes: ', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

# pdb.set_trace()
# (Pdb) pp test_dataset.num_seg_classes
# 3
# (Pdb) pp test_dataset.catfile
# 'shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt'
# (Pdb) pp test_dataset.cat
# {'Chair': '03001627'}
# (Pdb) pp len(dataset)
# 3371

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target)
        # pdb.set_trace()
        # (Pdb) pp points.shape
        # torch.Size([16, 2500, 3])
        # (Pdb) pp target.shape
        # torch.Size([16, 2500])

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1

        # pdb.set_trace()
        # (Pdb) pred.shape
        # torch.Size([40000, 4])
        # (Pdb) target.shape
        # torch.Size([40000])

        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        # pdb.set_trace()

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))