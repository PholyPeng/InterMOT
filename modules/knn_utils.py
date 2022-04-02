import copy
import numpy as np
import torch

class DistIndex(object):
    def __init__(self, dist, idx):
        self.distance = dist
        self.index = idx
    
    def __str__(self):
        output = '(%d, %.2f)' % (self.index, self.distance)
        return output

    def __lt__(self, other):
        return self.index < other.index
    
    def __gt__(self, other):
        return self.index > other.index
    
    def __le__(self, other):
        return self.index < other.index or self.index == other.index

    def __ge__(self, other):
        return self.index > other.index or self.index == other.index
     
class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []
        self.index_list = []
        #self.knn_idx_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))
            #self.knn_idx_list.append(0)
        self.comparison_counter = 0
        
    def size(self):
        return self.count
    
    def full(self):
        return self.count == self.capacity
    
    def worstDist(self):
        return self.worst_dist
    
    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.worst_dist:
            return
        
        if self.count < self.capacity:
            self.count += 1
            
        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                #self.knn_idx_list[i] = self.knn_idx_list[i-1]
                i -= 1
            else:
                break
        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity-1].distance
        #self.knn_idx_list[i] = index
    
    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output

    def get_index(self):
        if len(self.dist_index_list) == 0:
            return None

        for dist_index in self.dist_index_list:
            self.index_list.append(dist_index.index)
        
        return self.index_list

                
class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []
        self.index_list = []
        self.comparison_counter = 0
        
    def size(self):
        return self.count
    
    def worstDist(self):
        return self.radius
    
    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return
        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))
    
    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
            % (self.count, self.radius, self.comparison_counter)
        return output
    
        
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf
        
    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output
    
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]
    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
            depth[0] -= 1
            
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None
    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        root.is_leaf = False
        children_point_indices = [[] for i in range(8)]
        for point_idx in point_indices:
            point_db = db[point_idx]
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)
        factor = [-0.5, 0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = 0.5 * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            root.children[i] = octree_recursive_build(root.children[i],
            db,
            child_center,
            child_extent,
            children_point_indices[i],
            leaf_size,
            min_extent)
    return root

def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)

def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True
    
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)
    
    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius

def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    
    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius

def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False
    if contains(query, result_set.worstDist(), root):
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.worstDist(), root)
    
    for c, child in enumerate(root.children):
        if child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search_fast(child, db, result_set, query):
            return True
    return inside(query, result_set.worstDist(), root)

def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False
    
    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.worstDist(), root)
    
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
        
    if octree_radius_search(root.children[morton_code], db, result_set, query):
        return True
    
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_radius_search(child, db, result_set, query):
            return True
    
    return inside(query, result_set.worstDist(), root)

def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False
    
    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.worstDist(), root)
    
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
        
    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True
    
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.worstDist(), child):
            continue
        if octree_knn_search(child, db, result_set, query):
            return True
    return inside(query, result_set.worstDist(), root)

def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = db_np_min + db_extent
    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)), leaf_size, min_extent)
    return root


def knn_point_save_memory(k, xyz, new_xyz, radius = None):
    """
    :param k: int32, number of k in k-nn search
    :param xyz: (B, N, C) float32 array, input points
    :param new_xyz: (B, S, C) float32 array, query points
    :return: (B, S, k) int32 array, indices to input points
    """
    leaf_size = 4
    min_extent = 0.0001

    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    if k > N:
        k = N
    if C == 2:  
        new_xyz = torch.cat([new_xyz, torch.zeros(B, S, 1).cuda()], dim = -1)   
        xyz = torch.cat([xyz, torch.zeros(B, N, 1).cuda()], dim = -1)
    queries = new_xyz.squeeze(0).cpu().numpy()
    db = xyz.squeeze(0).cpu().numpy()
    root = octree_construction(db, leaf_size, min_extent)
    idx = []
    for i, query in enumerate(queries):
        result_set = KNNResultSet(capacity=k)     
        octree_knn_search(root, db, result_set, query)
        idx.append(result_set.get_index())
        #idx.append(result_set.knn_idx_list)
    
    idx = torch.LongTensor(idx)
    idx = idx.unsqueeze(0)
    return idx


    
