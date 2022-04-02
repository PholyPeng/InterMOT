import unittest
import torch
from torch.autograd import Variable, Function

from vision_sandbox import _C
knn = _C.knn

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_group(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, S]
    Return:
        new_points:, indexed points data, [B, N, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm?
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(k, xyz, new_xyz, radius = None):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz: (B, N, C) float32 array, input points
        new_xyz: (B, S, C) float32 array, query points
    Output:
        idx: (B, S, k) int32 array, indices to input points
    '''
    
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    dist = square_distance(new_xyz, xyz)
    dist_sorted,idx = dist.sort(dim = -1)
    if radius is not None:
        idx[dist_sorted > radius ** 2] = N
        
    idx = idx[:,:,:k]
    #TODO:if k > N,size of idx is [B,S,N]
    if k > N:
        group_first = idx[:, :, 0].view(B, S, 1).repeat([1, 1, N])
    else:
        group_first = idx[:, :, 0].view(B, S, 1).repeat([1, 1, k])
    mask = idx == N
    idx[mask] = group_first[mask]
    return idx

"""
ref [batch, dim, num_ref]

query [batch, dim, num_query]

inds [batch, k, num_query]
"""

def knn_point_cuda(k, xyz, new_xyz, radius = None):
    """
    :param k: int32, number of k in k-nn search
    :param xyz: (B, N, C) float32 array, input points
    :param new_xyz: (B, S, C) float32 array, query points
    :return: (B, S, k) int32 array, indices to input points
    ref [batch, dim, num_ref]
    query [batch, dim, num_query]
    inds [batch, k, num_query]
    """
    #print(new_xyz)
    device = xyz.device
    B, S, C = new_xyz.shape
    ref = xyz.permute(0,2,1).contiguous() #(B, C, N)
    query = new_xyz.permute(0,2,1).contiguous()   #(B, C, S) 
    ref = ref.to(device)
    query = query.to(device) 
     
    inds = torch.empty(B, k, S).long()
    inds = inds.to(device)
    knn(ref, query, inds)
    inds = inds.permute(0,2,1).contiguous()
    inds = inds - 1
    
    return inds
    
class TestKNearestNeighbor(unittest.TestCase):

    def test_forward(self):
        D, N, M = 3, 1000, 500
        xyz = torch.rand(1, D, N)
        new_xyz = torch.rand(1, D, M)
        ref = Variable(xyz)
        query = Variable(new_xyz)
        ref = ref.float()
        query = query.float()
        #print(ref.shape) # [1, 3, 1000]

        k = 16
        inds = torch.empty(1, k, M).long()
        print("C++")
        knn(ref, query, inds)
        print(inds.shape)      #(B, k, S)
        inds = inds.permute(0,2,1)
        # sorted_idx1, _ = torch.sort(inds, dim = 1)
        # print(sorted_idx1)
        print(inds)

        print("Original")
        idx1 = knn_point(k, xyz.permute(0,2,1), new_xyz.permute(0,2,1))
        print(idx1)
    
        # sorted_idx1, _ = torch.sort(idx1, dim = 2)
        # print(sorted_idx1)
    
        print("CUDA")
        ref = ref.cuda()
        query = query.cuda()
        inds = inds.cuda()
        knn(ref, query, inds)
        inds = inds.permute(0,2,1)
        print(inds)

def test():
    # N:100000, M:1000 --- 3861MB
    # N:100000, M:2000 --- 6913MB
    # N:100000, M:4000 ---13017MB
    # N:200000, M:1000 --- 6913MB
    D, N, M = 3, 200000, 2000
    ref = torch.rand(1, N, D)
    #query = torch.rand(1, M, D)
    perm = torch.randperm(N)
    indices = perm[:M]
    indices = indices.unsqueeze(0)

    query = index_points(ref,indices)
    k = 5
    #idx1 = knn_point_cuda(k, ref, query)

    ref = ref.cuda()
    query = query.cuda()

    idx2 = knn_point(k, ref, query)
    #print(idx2)

if __name__ == '__main__':

    n_epoch = 10
    for i in range(n_epoch):
        print("Processing {}/{} iterations".format(i+1, n_epoch))
        test()