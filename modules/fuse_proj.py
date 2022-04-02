import torch
import torch.nn as nn


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device

    B, _, C = points.shape
    cat = torch.zeros(B, 1, C)
    cat = cat.cuda()
    points_cat = torch.cat([points, cat], dim=1)
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(
        B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points_cat[batch_indices, idx, :]
    return new_points


def Grid_generator(w, h, merged_channels=True):
    """
    Generate the grid for the image plane
    W = 224
    H = 224
    """
    grid_x, grid_y = torch.meshgrid(torch.linspace(0.5, w - 0.5, w),
                                    torch.linspace(0.5, h - 0.5, h))
    if merged_channels:
        xy = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)),
                       1).unsqueeze(0)
    else:
        xy = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)),
                       1).unsqueeze(0).reshape(-1, w, h, 2)

    return xy


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
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
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

def query_circle_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 2]
        new_xyz: query points, [B, S, 2]
        S:npoints,the number of sampled points
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    #print("device",device)
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(
        1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if nsample > N:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, N])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class Feat_extractor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(Feat_extractor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        """
        Input:
            x: input data where to extract features, [B, npoint, nsample, C]
        Output:
            out:  extracted features [B, npoint, C']   
        """
        B, npoint, nsample, C = x.shape
        if x is not None:
            x = x.permute(0, 3, 1, 2)  # [B, C, npoint, nsample]
        #print("x.shape:",x.shape)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = self.relu(bn(conv(x)))

        extracted = self.avg_pool(x)  # [B, C', npoint, 1]
        out = extracted.squeeze(-1).permute(0, 2, 1)  # [B, npoint, C']

        assert out.size(1) == npoint

        return out  # [B, npoint, C']


class fusion2to1(nn.Module):
    def __init__(self, in_channel, mlp):
        super(fusion2to1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = self.relu(bn(conv(x)))
        return x


class fusion221(nn.Module):
    def __init__(self, appear_len, point_len, out_channels):
        super(fusion221, self).__init__()

        self.appear_len = appear_len
        self.point_len = point_len
        self.gate_p = nn.Sequential(
            nn.Conv1d(point_len, point_len, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_i = nn.Sequential(
            nn.Conv1d(appear_len, appear_len, 1, 1),
            nn.Sigmoid(),
        )
        self.input_p = nn.Sequential(
            nn.Conv1d(point_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.input_i = nn.Sequential(
            nn.Conv1d(appear_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):

        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        raw_feats = feats[:1]
        group_feats = feats[1:]
        B, D, L = raw_feats.shape

        mask = group_feats.byte().any(dim=1)
        mask = mask.view(B, 1, L).repeat([1, D, 1])

        gate_p = self.gate_p(feats[:1])
        gate_i = self.gate_i(feats[1:])
        obj_fused = gate_p.mul(self.input_p(feats[:1])) + gate_i.mul(self.input_i(feats[1:]))
        obj_feats = obj_fused.div(gate_p + gate_i)

        res = torch.where(mask == 0, raw_feats, obj_feats)

        return res


class fuse_extract(nn.Module):
    def __init__(self):
        super(fuse_extract, self).__init__()
        self.point_center_extractor = Feat_extractor(in_channel=512, mlp=[512, 1024])
        self.pixel_center_extractor = Feat_extractor(in_channel=1024, mlp=[1024, 512])
        self.point_center_merge = fusion221(1024,1024,1024)
        self.pixel_center_merge = fusion221(512,512,512)

    def points_centered_sample_group(self, projected_xy, points, img, M_matrix,
                                     radius, nsample):

        Bi, C, H, W = img.shape
        img_d = img.permute(0, 2, 3, 1).contiguous().reshape(
            Bi, -1, C)  #[1, C, H, W] -> [1, H, W, C] -> [1, W * H, C]
        Bp, D, L = points.shape  # [1, D, L]

        z_cord = torch.ones(1, 1, L).cuda()
        #projected_xy:the original postion on the full image plane 1 x 2 x L

        projected_xyz = torch.cat([projected_xy, z_cord], dim=1)  # 1 x 3 x L

        projected_on_img_points = torch.bmm(M_matrix,
                                            projected_xyz)  #1 x 3 x L
        projected_on_img_points = projected_on_img_points[:, :2, :]  #1 x 2 x L
        projected_on_img_points = projected_on_img_points.permute(
            0, 2, 1)  #1 x L x 2

        img_xy = Grid_generator(W, H)  #[1, W * H, 2]
        img_xy = img_xy.cuda()
        #idx = query_circle_point(radius, nsample, img_xy, projected_on_img_points)  #1 x L x nsample
        idx = knn_point(nsample, img_xy, projected_on_img_points)  #1 x L x nsample
        #img : [1, W * H, 3]
        grouped_img_xy = index_points(img_xy, idx)  # [1, L, nsample, 2]
        grouped_img = index_points(img_d, idx)  # [1, L, nsample, C]
   
        feats = [] 
        feats.append(points)  # [1, D, L]      


        appear_extracted = self.point_center_extractor(grouped_img)  # expected output:[1, L, D]
        appear_extracted = appear_extracted.transpose(-2, -1)  #[1, D, L]

        feats.append(appear_extracted)

        out = torch.cat(feats, dim=1)  # output:[1, 2*D, L]

        #merge = fusion2to1(in_channel = 2*D, mlp = [D])
        out = self.point_center_merge(out)  #[1,D,L]
        return out

    def pixels_centered_sample_group(self, projected_xy, points, img, M_matrix,
                                     radius, nsample):

        Bi, C, H, W = img.shape
        img_d = img.permute(0, 2, 3, 1).contiguous().reshape(
            Bi, -1, C)  #[1, C, H, W] -> [1, W, H, C] -> [1, W * H, C]
        Bp, D, L = points.shape  # 1xDxL
        points_d = points.permute(0, 2, 1).contiguous()  #1xLxD
        z_cord = torch.ones(1, 1, L).cuda()
        projected_xyz = torch.cat([projected_xy, z_cord], dim=1)  #1 x 3 x L

        projected_on_img_points = torch.bmm(M_matrix,
                                            projected_xyz)  #1 x 3 x L
        projected_on_img_points = projected_on_img_points[:, :
                                                          2, :]  # 1 x 2 x L
        projected_on_img_points = projected_on_img_points.permute(
            0, 2, 1)  #1 x L x 2

        img_xy = Grid_generator(W, H).cuda()  #[1, W * H, 2]

        #idx = query_circle_point(radius, nsample, projected_on_img_points, img_xy)  # [1, W * H, nsample]
        idx = knn_point(nsample, projected_on_img_points, img_xy)  # [1, W * H, nsample]
        grouped_points = index_points(points_d, idx)  # [1, W*H, nsample, D]

        feats = []
        feats.append(img)  # [1, C, H, W]

        #print("grouped_points.shape",grouped_points.shape)
        points_extracted = self.pixel_center_extractor(grouped_points)  # [1, W*H, C]

        points_extracted = points_extracted.permute(
            0, 2, 1).contiguous().reshape(Bp, -1, H, W)  # [1, C, H, W]
        feats.append(points_extracted)

        out = torch.cat(feats, dim=1)  # [1, 2*C, H, W]
        out = torch.reshape(out, (Bi, 2 * C, -1))  # [1, 2*C, H*W]
        #merge = fusion2to1(in_channel= 2*C,mlp = [C])
        
        out = self.pixel_center_merge(out)  # [1, C, H*W]
        out = out.reshape(Bi, C, H, W)  # [1, C, H, W]

        return out

    def condense_net(self, xy, points, img, M_matrix, radius, nsample):

        feat1 = self.points_centered_sample_group(xy, points, img, M_matrix,
                                                  radius, nsample)

        feat2 = self.pixels_centered_sample_group(xy, points, img, M_matrix,
                                                  radius, nsample)

        return feat1, feat2

    def forward(self,
                xys,
                imgs,
                points,
                points_split,
                M_matrixes,
                radius=10,
                nsample=16):
        """
        xys: points position on the image plane 1 x 2 x L
        imgs: N+M x C x H x W
        M_matrixes:N+M x 3 x 3
        """
        #divide the imgs into N+M tensors of the same size(1 x C x H x W)
        imgs_list = list(imgs.split(split_size=1, dim=0))

        M_matrix_list = list(M_matrixes.split(split_size=1, dim=0))
        #print("M_matrix_list.shape",M_matrix_list[0].shape)
        points_list = []
        xy_list = []
        for i in range(len(points_split) - 1):
            start = points_split[i].item()
            end = points_split[i + 1].item()
            points_list.append(points[:, :, start:end])
            xy_list.append(xys[:, :, start:end])

        fused_img_list = []
        fused_points_list = []
        for (xy, img, points, M_matrix) in zip(xy_list, imgs_list, points_list,
                                               M_matrix_list):
            fused_points_tensor, fused_img_tensor = self.condense_net(
                xy, points, img, M_matrix, radius,
                nsample)  #[1,D,L'] [1, C, H, W]
            fused_points_list.append(fused_points_tensor)
            fused_img_list.append(fused_img_tensor)

        fused_imgs = torch.cat(fused_img_list, dim=0)  # [N+M, C, H, W]
        fused_points = torch.cat(fused_points_list, dim=-1)  #[1,D,L]

        return fused_imgs, fused_points
