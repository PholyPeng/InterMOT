import torch

from vision_sandbox import _C
knn = _C.knn

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

def test_knn():
    D, N, M = 3, 100, 10
    ref = torch.rand(1, N, D)
    query = torch.rand(1, M, D)
    k = 5
    
    idx2 = knn_point_cuda(k, ref, query)
    print(idx2)
   
if __name__ == '__main__':
    test_knn()