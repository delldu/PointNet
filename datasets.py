from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pdb

class PartDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 shuffle=True,
                 train=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification
        self.shuffle = shuffle

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # (Pdb) pp self.cat
        # {'Airplane': '02691156',
        #  'Bag': '02773838',
        #  'Cap': '02954340',
        #  'Car': '02958343',
        #  'Chair': '03001627',
        #  'Earphone': '03261776',
        #  'Guitar': '03467517',
        #  'Knife': '03624134',
        #  'Lamp': '03636649',
        #  'Laptop': '03642806',
        #  'Motorbike': '03790512',
        #  'Mug': '03797390',
        #  'Pistol': '03948459',
        #  'Rocket': '04099429',
        #  'Skateboard': '04225987',
        #  'Table': '04379243'}

        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))


        # (Pdb) pp  len(self.meta['Airplane']), self.meta['Airplane'][0]
        # (2421,
        #  ('shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/1021a0914a7207aff927ed529ad90a11.pts',
        #   'shapenetcore_partanno_segmentation_benchmark_v0/02691156/points_label/1021a0914a7207aff927ed529ad90a11.seg'))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)

        # pdb.set_trace()

    def __getitem__(self, index):
        fn = self.datapath[index]
        # ('Airplane', 'shapenetcore_partanno_segmentation_benchmark_v0/
        #    02691156/points/1021a0914a7207aff927ed529ad90a11.pts', 
        # 'shapenetcore_partanno_segmentation_benchmark_v0
        #    /02691156/points_label/1021a0914a7207aff927ed529ad90a11.seg')
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        #print(point_set.shape, seg.shape)
        if self.shuffle:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            #resample
            point_set = point_set[choice, :]
            seg = seg[choice]


        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
