from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import pdb


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        # pdb.set_trace()
        # (Pdb) a
        # self = STN3d(
        #   (conv1): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
        #   (conv2): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        #   (conv3): Conv1d(128, 1024, kernel_size=(1,), stride=(1,))
        #   (fc1): Linear(in_features=1024, out_features=512, bias=True)
        #   (fc2): Linear(in_features=512, out_features=256, bias=True)
        #   (fc3): Linear(in_features=256, out_features=9, bias=True)
        #   (relu): ReLU()
        #   (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )


    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([32, 3, 2500])

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        # pdb.set_trace()
        # (Pdb) iden
        # tensor([[1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 1.]], device='cuda:0')
        # (Pdb) iden.shape
        # torch.Size([32, 9])

        # (Pdb) pp x.shape
        # torch.Size([32, 3, 3])

        return x


class PointFeat(nn.Module):
    def __init__(self, global_feat = True):
        super(PointFeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat


    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([32, 3, 2500])
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)

        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        # ==> x.shape --> (32, 3, 2500)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # pdb.set_trace()
        # (Pdb) pp pointfeat.shape
        # torch.Size([32, 64, 2500])
        # (Pdb) x.shape
        # torch.Size([32, 1024])
        # (Pdb) trans.shape
        # torch.Size([32, 3, 3])
        # (Pdb) 

        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetCls, self).__init__()
        self.feat = PointFeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # pdb.set_trace()
        # k = 16

    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([32, 3, 2500])                
        x, trans = self.feat(x)
        # pdb.set_trace()
        # (Pdb) x.shape, trans.shape
        # (torch.Size([32, 1024]), torch.Size([32, 3, 3]))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([32, 16])

        return F.log_softmax(x, dim=0), trans

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointFeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # pdb.set_trace()
        # k = 4
        # self = PointNetDenseCls(
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
        #   (conv1): Conv1d(1088, 512, kernel_size=(1,), stride=(1,))
        #   (conv2): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
        #   (conv3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        #   (conv4): Conv1d(128, 4, kernel_size=(1,), stride=(1,))
        #   (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )


    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([16, 3, 2500])

        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        # pdb.set_trace()
        # (Pdb) x.shape
        # torch.Size([16, 2500, 4])
        # (Pdb) trans.shape
        # torch.Size([16, 3, 3])

        return x, trans


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointFeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointFeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())


