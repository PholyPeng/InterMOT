import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from time import time
import numpy as np
from .pointnet_util import PointNetSetAbstraction,PointNetFeaturePropagation

class PointNet_v1(nn.Module):

    def __init__(self, in_channels, out_channels=512, use_dropout=False):
        super(PointNet_v1, self).__init__()
        self.feat = PointNetfeatGN(in_channels, out_channels)
        reduction = 512 // out_channels
        self.reduction = reduction
        self.conv1 = torch.nn.Conv1d(1088 // reduction, 512 // reduction, 1)
        self.conv2 = torch.nn.Conv1d(512 // reduction, out_channels, 1)
        self.bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.bn2 = nn.GroupNorm(16 // reduction, out_channels)
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_bn = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.dropout = None
        if use_dropout:
            print("Use dropout in pointnet")
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, point_split):
        #print(point_split)
        #f = open(r'C:\Users\Peter\Desktop\points.txt','a')
        #f.write(str(x.cpu()) + '\n')
        #f2= open(r'C:\Users\Peter\Desktop\split.txt','a')
        #f2.write(str(point_split.cpu()) + '\n')
        
        x, trans = self.feat(x, point_split)
        x = torch.cat(x, dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)

        max_feats = []
        for i in range(len(point_split) - 1):
            start = point_split[i].item()
            end = point_split[i + 1].item()
            max_feat = self.avg_pool(x[:, :, start:end])
            max_feats.append(max_feat.view(-1, 512 // self.reduction, 1))

        max_feats = torch.cat(max_feats, dim=-1)
        out = self.relu(self.bn2(self.conv2(max_feats))).transpose(
            -1, -2).squeeze(0)
        assert out.size(0) == len(point_split) - 1
        
        #f3 = open(r'C:\Users\Peter\Desktop\points_m.txt','a')
        #f3.write(str(out.cpu())+ '\n')
        return out, trans


class STN3d(nn.Module):

    def __init__(self, in_channels, out_size=3, feature_channels=512):
        super(STN3d, self).__init__()
        reduction = 512 // feature_channels
        self.reduction = reduction
        self.out_size = out_size
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, 64 // reduction, 1)
        self.bn1 = nn.GroupNorm(64 // reduction, 64 // reduction)
        self.conv2 = nn.Conv1d(64 // reduction, 128 // reduction, 1)
        self.bn2 = nn.GroupNorm(128 // reduction, 128 // reduction)
        self.conv3 = nn.Conv1d(128 // reduction, 1024 // reduction, 1)
        self.bn3 = nn.GroupNorm(1024 // reduction, 1024 // reduction)
        self.idt = nn.Parameter(torch.eye(self.out_size), requires_grad=False)

        self.fc1 = nn.Linear(1024 // reduction, 512 // reduction)
        self.fc_bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.fc2 = nn.Linear(512 // reduction, 256 // reduction)
        self.fc_bn2 = nn.GroupNorm(256 // reduction, 256 // reduction)

        self.output = nn.Linear(256 // reduction, out_size * out_size)
        nn.init.constant_(self.output.weight.data, 0)
        nn.init.constant_(self.output.bias.data, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -1, keepdim=True)[0]
        x = x.view(-1, 1024 // self.reduction)
        # print(x.shape) [1,1024]
        # model.eval()
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = self.relu(x)
        # x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.output(x).view(-1, self.out_size, self.out_size)

        x = x + self.idt
        #         idt = x.new_tensor(torch.eye(self.out_size))
        #         x = x + idt
        return x


class PointNetfeatGN(nn.Module):

    def __init__(self, in_channels=3, out_channels=512, global_feat=True):
        super(PointNetfeatGN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.stn1 = STN3d(in_channels, in_channels, out_channels)
        reduction = 512 // out_channels
        self.reduction = reduction
        self.conv1 = nn.Conv1d(in_channels, 64 // reduction, 1)
        self.bn1 = nn.GroupNorm(64 // reduction, 64 // reduction)

        self.conv2 = nn.Conv1d(64 // reduction, 64 // reduction, 1)
        self.bn2 = nn.GroupNorm(64 // reduction, 64 // reduction)
        self.stn2 = STN3d(64 // reduction, 64 // reduction, out_channels)

        self.conv3 = nn.Conv1d(64 // reduction, 64 // reduction, 1)
        self.bn3 = nn.GroupNorm(64 // reduction, 64 // reduction)

        self.conv4 = nn.Conv1d(64 // reduction, 128 // reduction, 1)
        self.bn4 = nn.GroupNorm(128 // reduction, 128 // reduction)
        self.conv5 = nn.Conv1d(128 // reduction, 1024 // reduction, 1)
        self.bn5 = nn.GroupNorm(1024 // reduction, 1024 // reduction)
        self.global_feat = global_feat
        print("use avg in pointnet feat")
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, point_split):
        conv_out = []
        trans = []

        trans1 = self.stn1(x)
        trans.append(trans1)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        x = x.transpose(2, 1)

        x = self.relu(self.bn1(self.conv1(x)))

        trans2 = self.stn2(x)
        trans.append(trans2)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2, 1)
        conv_out.append(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        max_feats = []
        for i in range(len(point_split) - 1):
            start = point_split[i].item()
            end = point_split[i + 1].item()
            max_feat = self.avg_pool(x[:, :, start:end])
            max_feats.append(
                max_feat.view(-1, 1024 // self.reduction,
                              1).repeat(1, 1, end - start))

        max_feats = torch.cat(max_feats, dim=-1)

        assert max_feats.size(-1) == x.size(-1)
        conv_out.append(max_feats)

        return conv_out, trans

class PointNet_v2(nn.Module):
    
    def __init__(self, in_channels, out_channels=512, use_dropout=False):
        super(PointNet_v2, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        reduction = 512 // out_channels
        self.reduction = reduction
        self.conv1 = torch.nn.Conv1d(1024 // reduction, 512 // reduction, 1)
        self.conv2 = torch.nn.Conv1d(512 // reduction, out_channels // reduction, 1)
        self.bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.bn2 = nn.GroupNorm(16 // reduction, out_channels)
        self.relu = nn.ReLU(inplace=True)
        print("use avg in pointnet feat")
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.stn1 = STN3d(in_channels, in_channels, out_channels)
        self.stn2 = STN3d(128 // reduction, 128 // reduction, out_channels)
        self.dropout = None
        if use_dropout:
            print("Use dropout in pointnet")
            self.dropout = nn.Dropout(p=0.5)       
           
    def multi_dets(self, xyz, points, point_split, radius, in_channel, mlp, group_all):
        new_xyz = []
        new_points = []
        split = [0]
        S0 = 0
        #print(len(point_split))
        
        for i in range(len(point_split) - 1):
            
            start = point_split[i].item()
            end = point_split[i + 1].item()
            #print("start:",start,",\tend:",end)
            xyzd = xyz[:, :3, start:end]
            pointsd = points[:, :, start:end]
            n_points = end - start
            #print("n_points:\t", n_points, "start:\t", start, "end:\t", end)
            num = int(math.sqrt(n_points))
            #print("xyz.shape:\t",xyzd.shape)
            #print("pointsd.shape:\t",pointsd.shape)
            sa = PointNetSetAbstraction(npoint = int(n_points//2) + 2, radius = radius, nsample = int(num)+1, in_channel = in_channel, mlp = mlp, group_all = group_all)
            sa = sa.cuda()
            l_xyz, l_points = sa(xyzd, pointsd) # l_xyz:[B, C, S] l_points:[B, D', S]
            #print("xyz_m.shape:\t",l_xyz.shape)
            #print("pointsd_m.shape:",l_points.shape)
            S = l_xyz.shape[2]
            split.append(S0 + S)
            S0 = S + S0
            new_xyz.append(l_xyz)
            new_points.append(l_points)
        new_xyz = torch.cat(new_xyz, dim = -1)      # new_xyz:[B, C, S * M = L'] M is the number of the detections
        new_points = torch.cat(new_points, dim = -1)
        split = torch.tensor(split, dtype= torch.int)
        return new_xyz, new_points, split
    
    def forward(self, points, point_split):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """ 
        if self.dropout is not None:
            points = self.dropout(points)
        
        xyz = points[:, :3, : ]
        #print(xyz)
        trans = []

        trans1 = self.stn1(xyz)
        trans.append(trans1)
        xyz = xyz.transpose(2, 1)
        xyz = torch.bmm(xyz, trans1)
        xyz = xyz.transpose(2, 1)
        points = points.transpose(2, 1)
        points = torch.bmm(points, trans1)
        points = points.transpose(2, 1)
        
        
        l1_xyz, l1_points, point_split1 = self.multi_dets(xyz, points, point_split, 
                               radius = 0.4, in_channel = self.in_channel + 3, 
                               mlp=[64, 64, 128], group_all=False)
        
        #print(l1_xyz)
        #print(l1_points)
        trans2 = self.stn2(l1_points)
        trans.append(trans2)
        l1_points = l1_points.transpose(2, 1)
        l1_points = torch.bmm(l1_points, trans2)
        l1_points = l1_points.transpose(2, 1)
        
        l2_xyz, l2_points, point_split2 = self.multi_dets(l1_xyz, l1_points, point_split1,
                               radius = 0.8, in_channel=128 + 3, 
                               mlp=[128, 128, 256], group_all=False)
        
        l3_xyz, l3_points, point_split3 = self.multi_dets(l2_xyz, l2_points, point_split2,
                               radius = 1.2, in_channel=256 + 3, 
                               mlp=[256, 512, 1024], group_all=False)
        #print(point_split3)
        l3_points = self.relu(self.bn1(self.conv1(l3_points)))
        
        feats = []
        for i in range(len(point_split3) - 1):
            start = point_split3[i].item()
            end = point_split3[i + 1].item()
            feat = self.avg_pool(l3_points[:, :, start:end])
            feats.append(feat.view(-1, 512, 1))
        
        feats = torch.cat(feats, dim = -1)
        out = self.relu(self.bn2(self.conv2(feats))).transpose(-1, -2).squeeze(0)
        
        assert out.size(0) == len(point_split) - 1
        
        return out, trans
