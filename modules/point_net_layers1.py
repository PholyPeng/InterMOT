import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from time import time
import numpy as np
from .pointnet_util import PointNetSetAbstraction,PointNetFeaturePropagation,PointNetSetAbstractionMsg


def get_n_pt(npoints):
    counter = 2
    while(counter < npoints):
        counter *= 2
    return counter//2

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
    

class PointNetpp(nn.Module):
    
    def __init__(self, radius, in_channel, mlp, group_all):
        super(PointNetpp, self).__init__()
        self.radius = radius
        self.in_channel = in_channel
        self.mlp = mlp
        self.group_all = group_all


    def forward(self, xy, xyz, points, point_split):
        new_xy = []
        new_xyz = []
        new_points = []
        split = [0]
        S0 = 0
        
        for i in range(len(point_split) - 1):
            
            start = point_split[i].item()
            end = point_split[i + 1].item()
            xyd = xy[:,:2,start:end]
            xyzd = xyz[:, :3, start:end]
            pointsd = points[:, :, start:end]
            n_points = end - start
            npoint = get_n_pt(n_points)
            
            sa = PointNetSetAbstraction(npoint = npoint, radius = self.radius, nsample = 32, in_channel = self.in_channel, mlp = self.mlp, group_all = self.group_all)
            sa.cuda()
            l_xy, l_xyz, l_points = sa(xyd, xyzd, pointsd) # l_xyz:[B, 3, S] l_points:[B, D', S]

            S = l_xyz.shape[2]
            split.append(S0 + S)
            
            S0 = split[-1]
            #S0 = S + S0
            new_xy.append(l_xy)
            new_xyz.append(l_xyz)
            new_points.append(l_points)
            
        new_xy = torch.cat(new_xy,dim =-1)
        new_xyz = torch.cat(new_xyz, dim = -1)
        new_points = torch.cat(new_points, dim = -1)
        split = torch.tensor(split, dtype= torch.int)
        return new_xy, new_xyz, new_points, split

class PointFP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointFP, self).__init__()
        self.in_channel = in_channel
        self.mlp = mlp
    
    def forward(self, xyz1, points1, point_split1, xyz2, points2,  point_split2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        split = [0]
        S0 = 0
        new_points = []
        for i in range(len(point_split1) - 1):
            
            start1 = point_split1[i].item()
            end1 = point_split1[i + 1].item()
            xyzd1 = xyz1[:, :3, start:end]
            pointsd1 = points1[:, :, start:end]
            
            if points1 is None:
                pointsd1 = None
            else:
                start2 = point_split2[i].item()
                end2 = point_split2[i + 1].item()
                xyzd2 = xyz2[:, :3, start:end]
                pointsd2 = points2[:, :, start:end]

            fp = PointNetFeaturePropagation(in_channel = self.in_channel,mlp = self.mlp)
            fp.cuda()

            l_points = fp(xyzd1, xyzd2, pointsd1, pointsd2)
            S = l_points.shape[2]
            split.append(S0 + S) 
            S0 = split[-1]
            new_points.append(l_points)

        new_points = torch.cat(new_points, dim = -1)

        return new_points

class PointNetpp_msg(nn.Module):
    
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetpp_msg, self).__init__()
        self.radius_list = radius_list
        self.in_channel = in_channel
        self.mlp_list = mlp_list
        self.nsample_list = nsample_list


    def forward(self, xy, xyz, points, point_split):
        new_xy = []
        new_xyz = []
        new_points = []
        split = [0]
        S0 = 0
        
        for i in range(len(point_split) - 1):
            
            start = point_split[i].item()
            end = point_split[i + 1].item()
            xyd = xy[:,:2,start:end]
            xyzd = xyz[:, :3, start:end]
            pointsd = points[:, :, start:end]
            n_points = end - start
            
            sa = PointNetSetAbstractionMsg(npoint = int(n_points//2) + 2, radius_list = self.radius_list, nsample_list = self.nsample_list,in_channel = self.in_channel, mlp_list = self.mlp_list)
            sa.cuda()
            l_xy, l_xyz, l_points = sa(xyd, xyzd, pointsd) # l_xyz:[B, 3, S] l_points:[B, D', S]

            S = l_xyz.shape[2]
            split.append(S0 + S)
            
            S0 = split[-1]
            #S0 = S + S0
            new_xy.append(l_xy)
            new_xyz.append(l_xyz)
            new_points.append(l_points)
            
        new_xy = torch.cat(new_xy,dim =-1)
        new_xyz = torch.cat(new_xyz, dim = -1)      # new_xyz:[B, C, S1 + S2 + S3 + ������+ Sm = L'] m is the number of the detections
        new_points = torch.cat(new_points, dim = -1)
        split = torch.tensor(split, dtype= torch.int)
        return new_xy,new_xyz, new_points, split

def make_ssg_layer( radius,
                        in_channel,
                        mlp,
                        group_all):
    return PointNetpp(radius, in_channel, mlp, group_all)

def make_msg_layer( radius_list,
                      nsample_list,
                        in_channel,
                        mlp_list
                        ):
    return PointNetpp_msg(radius_list, nsample_list, in_channel, mlp_list)

def make_fp_layers(in_channel,mlp):
    return PointFP(in_channel,mlp)

def make_stn_layers(in_channels, out_size, feature_channels):
    return STN3d(in_channels, out_size, feature_channels)

def parse_pointnet_layers(in_channel, 
                          out_channels=512, 
                          use_dropout=False):
    reduction = 512 // out_channels
    # radius_list = [1,2,3,5],
    # in_channel_list = [3 + 3,128 + 3,256 + 3,256 + 3],
    # mlp_list = [[64, 128],[128,256],[256],[512,1024]], 
    # group_all_list = [False, False, False,False]
    
    stn_layers = []
    stn_layers.append(make_stn_layers(3, 3, 512))
    stn_layers.append(make_stn_layers(128, 128, 512))
    
    layers = []
    fp_layers = []
    layers.append(make_ssg_layer(radius=0.2, in_channel=in_channel+3, mlp=[64, 64, 128], group_all=False))
    layers.append(make_ssg_layer(radius=0.4, in_channel = 128 + 3, mlp = [128, 256] , group_all = False))
    layers.append(make_ssg_layer(radius= 0.6,in_channel = 256 + 3, mlp = [256,512] , group_all = False))
    layers.append(make_ssg_layer(radius= 1.0,in_channel = 512 + 3, mlp = [512, 1024] , group_all = False))
    #layers.append(make_ssg_layer(radius=0.5, in_channel=in_channel+3, mlp=[64, 64, 128], group_all=False))
    #layers.append(make_msg_layer([0.1, 0.2, 0.4], [16, 32, 128], 128 ,[[32, 32, 64], [64, 64, 128], [64, 96, 128]]))
    #layers.append(make_msg_layer([0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]]))
    #layers.append(make_ssg_layer(1.0, 640 + 3, [256, 512, 1024], False))
    fp_layers.append(make_fp_layers(1024 + 512,[512,512]))#l3 + l4 -> l3
    fp_layers.append(make_fp_layers(512 + 256,[512,512])) #l2 + l3 -> l2
    fp_layers.append(make_fp_layers(512 + 128,[512,256])) #l1 + l2 -> l1
    fp_layers.append(make_fp_layers(256,[256,256,256]))
    
    conv1 = nn.Conv1d(256 // reduction, 512 // reduction, 1)
    conv2 = nn.Conv1d(512 // reduction, out_channels // reduction, 1)
    #fc1 = nn.Linear(1024 // reduction, 512 // reduction)
    #fc2 = nn.Linear(512 // reduction, out_channels // reduction)
    #bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
    #bn2 = nn.GroupNorm(16 // reduction, out_channels)
    bn1 = nn.BatchNorm1d(512)
    bn2 = nn.BatchNorm1d(512)
    relu = nn.ReLU(inplace=True)
    avg_pool = nn.AdaptiveAvgPool1d(1)
    conv_bn1 = [conv1,bn1,relu]
    layers.append(nn.Sequential(*conv_bn1))
    conv_bn2 = [conv2,bn2,relu]
    layers.append(nn.Sequential(*conv_bn2))
    layers.append(avg_pool)
      
    return nn.ModuleList(stn_layers),nn.ModuleList(layers),nn.ModuleList(fp_layers)

'''radius_list = [10,20,30,50]
in_channel_list = [3,128,256,512]
mlp_list = [[64, 64, 128],[128, 128, 256],[256, 512, 512],[512,1024]]
group_all_list = [False, False, False,False]
stn, layers_all = parse_pointnet_layers(in_channels = 3)
print(stn)
print(layers_all)'''