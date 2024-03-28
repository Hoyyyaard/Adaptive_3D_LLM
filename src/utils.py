import sys
sys.path.append('/home/admin/Projects/LL3DA/')
import numpy as np
from tqdm import tqdm
import utils.pc_util as pc_util
import open3d
import time
from copy import deepcopy

def _farthest_point_sampling(points, num_points):
    """
    Perform farthest point sampling (FPS) on a point cloud.

    Args:
        points (numpy array): Input point cloud with shape (N, D), where N is the number of points and D is the dimensionality.
        num_points (int): Number of points to be sampled.

    Returns:
        selected_points (numpy array): Sampled points with shape (num_points, D).
    """
    N = points.shape[0]
    selected_indices = []
    selected_points = []

    # Randomly select the first point
    first_index = np.random.randint(0, N)
    selected_indices.append(first_index)
    selected_points.append(points[first_index])

    # Calculate distance to the first point
    distances = np.linalg.norm(points[:,:3] - points[first_index][:3], axis=1)

    for _ in tqdm(range(1, num_points),desc='FPS'):
        # Find the point farthest from the selected points
        farthest_index = np.argmax(distances)

        # Update distances
        new_distances = np.linalg.norm(points[:,:3] - points[farthest_index][:3], axis=1)
        distances = np.minimum(distances, new_distances)

        selected_indices.append(farthest_index)
        selected_points.append(points[farthest_index])

    selected_points = np.array(selected_points)

    return selected_points

## USer: below are used for dense or inflate pointcloud
def dense_pointclouds(raw_pointclouds, instance_pointclouds, target_obj_id_list, object_size, object_num, num_points):
    '''
        raw_pointclouds: [N*10] xyz rgb other_fts
        instance_pointclouds: [N*1] 
    '''
    TOTAL_POINT_NUM = 40000
    ADAPATIVE_POINT_NUM = 10000
    TGT_OBJ_PROB = 0.5
    INFLATE_PROB = 0.3
    REMAIN_PROB = 0.2
    
    if object_size <= 0.001:
        scale = 10
    elif 0.01 >= object_size > 0.001:
        scale = 5
    elif 0.1 >= object_size > 0.01:
        scale = 2
    elif object_size > 0.1:
        scale = 1
    
    scale = int(scale/object_num) if scale/object_num > 1 else 1
    
    
    def _get_inflate_axis_aligned_bounding_box(pcs, remaining_pcd, scale=scale, ):
        '''
            pcs & remaining_pcd : [N*3]
        '''
        tmp_pc = open3d.geometry.PointCloud()
        tmp_pc.points = open3d.utility.Vector3dVector(pcs)
        bbox = tmp_pc.get_axis_aligned_bounding_box()
        # 扩大边界框的尺寸n倍
        center = bbox.get_center()
        bbox.scale(scale, center)
        # 获取扩大后的边界框的最小和最大坐标
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        # TODO: 这里的高度边界应该选择整个场景的边界
        # min_bound[-1] = point_cloud_dims_min[-1]
        # max_bound[-1] = point_cloud_dims_max[-1]
        # 选择边界框内的余下点云的点
        indices_within_bbox = []
        for i, point in enumerate(remaining_pcd):
            if np.all(min_bound <= point) and np.all(point <= max_bound):
                indices_within_bbox.append(i)
        return indices_within_bbox
        
    remaining_pcd = None
    ## tgt object 点云信息
    object_pcd_list = []
    instance_object_pcd_list = []
    ## tgt object 膨胀点云信息（不包括tgt obj点云）
    inflate_pcd_list = []
    instance_inflate_pcd_list = []
    for target_obj_id in target_obj_id_list:
        ## 得到特定物体类别的点云
        obj_idx = instance_pointclouds == (target_obj_id + 1)
        
        ## 分离tgt obj点云和余下的点云
        tgt_obj_indices = np.where(obj_idx)[0]
        remaining_indices = list(set(range(raw_pointclouds.shape[0])) - set(tgt_obj_indices))

        ## 包含物体所有点云信息
        tmp_tgt_object_pcd = raw_pointclouds[tgt_obj_indices, :] 
        object_pcd_list.append(tmp_tgt_object_pcd)
        remaining_pcd = raw_pointclouds[remaining_indices, :] 
        
        tmp_instance_pointclouds = instance_pointclouds[tgt_obj_indices, :]
        inflate_pcd_list.append(tmp_instance_pointclouds) 
        instance_remaining_pcd = instance_pointclouds[remaining_indices, :] 
        
        ## 选择剩余点云中膨胀后边界框内点
        st = time.time()
        indices_within_bbox = _get_inflate_axis_aligned_bounding_box(tmp_tgt_object_pcd[:,:3], remaining_pcd[:,:3])
        # print(f'_get_inflate_axis_aligned_bounding_box time: {time.time()-st}')
        
        tmp_inflate_pcd = remaining_pcd[indices_within_bbox, :] 
        inflate_pcd_list.append(tmp_inflate_pcd)
        
        tmp_inflate_instacne_pcd = instance_remaining_pcd[indices_within_bbox, :] 
        instance_inflate_pcd_list.append(tmp_inflate_instacne_pcd)
        
        
        ## 在剩余点云中剔除膨胀点云
        remaining_index = range(len(remaining_pcd))
        remaining_index = [idx for idx in remaining_index if not idx in indices_within_bbox]
        remaining_pcd = remaining_pcd[remaining_index, :]
        instance_remaining_pcd = instance_remaining_pcd[remaining_index, :]
    
    all_tgt_object_pcds = np.concatenate(object_pcd_list, axis=0)
    all_inflate_pcds = np.concatenate(inflate_pcd_list, axis=0)
    
    all_instance_tgt_object_pcds = np.concatenate(instance_object_pcd_list, axis=0)
    all_instance_inflate_pcds = np.concatenate(instance_inflate_pcd_list, axis=0)
    
    tgt_object_prob = np.ones_like(all_tgt_object_pcds[:,0]) * TGT_OBJ_PROB
    inflate_prob = np.ones_like(all_inflate_pcds[:,0]) * INFLATE_PROB
    
    all_remaining_pcds = np.array(remaining_pcd)
    all_instance_remaining_pcds = np.array(instance_remaining_pcd)
    adaptive_pcds = np.concatenate([all_tgt_object_pcds, all_inflate_pcds], axis=0)
    instance_activate_pcds = np.concatenate([all_instance_tgt_object_pcds, all_instance_inflate_pcds], axis=0)
    
    # adaptive_pcds = all_tgt_object_pcds
    adaptive_prob = np.concatenate([tgt_object_prob, inflate_prob], axis=0)
    
    adaptive_pcds, choice = pc_util.random_sampling(adaptive_pcds, ADAPATIVE_POINT_NUM, return_choices=True)
    adaptive_prob = adaptive_prob[choice]
    instance_activate_pcds = instance_activate_pcds[choice]
    ## Only supervise the dense region detection
    point_vote_tgt_obj_list = np.unique(instance_activate_pcds)
    
    point_num_offset = int(TOTAL_POINT_NUM - len(adaptive_pcds) )
    # all_remaining_pcds = _farthest_point_sampling(all_remaining_pcds, point_num_offset)
    all_remaining_pcds, choices = pc_util.random_sampling(all_remaining_pcds, point_num_offset, return_choices=True)
    remain_prob = np.ones_like(all_remaining_pcds[:,0]) * REMAIN_PROB
    all_instance_remaining_pcds = all_instance_remaining_pcds[choices]
    
    adaptive_pcds = np.concatenate([adaptive_pcds, all_remaining_pcds], axis=0)
    adaptive_prob = np.concatenate([adaptive_prob, remain_prob], axis=0)
    # adaptive_pcds, _ = pc_util.random_sampling(np.concatenate([adaptive_pcds, all_remaining_pcds], axis=0), TOTAL_POINT_NUM, return_choices=True)
    # assert len(apdaptive_pcds) == TOTAL_POINT_NUM
    instance_adaptive_pcds = np.concatenate([instance_activate_pcds, all_instance_remaining_pcds], axis=0)
    
    point_votes = np.zeros([num_points, 3])
    point_votes_mask = np.zeros(num_points)
    for i_instance in point_vote_tgt_obj_list:            
        # find all points belong to that instance
        ind = np.where(instance_adaptive_pcds == i_instance)[0]
        x = adaptive_pcds[ind,:3]        ## [num_pcs, 3]
        center = 0.5*(x.min(0) + x.max(0))
        point_votes[ind, :] = center - x
        point_votes_mask[ind] = 1.0
    point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
    
    vote_label = point_votes.astype(np.float32)
    vote_label_mask = point_votes_mask.astype(np.int64)    
    
    ## 可视化
    # objects_pcd = open3d.geometry.PointCloud()
    # inflate_pcd = open3d.geometry.PointCloud()
    # remains_pcd = open3d.geometry.PointCloud()
    # objects_pcd.points = open3d.utility.Vector3dVector(adaptive_pcds[:,:3])
    # objects_pcd.colors = open3d.utility.Vector3dVector(adaptive_pcds[:,3:6])
    # inflate_pcd.points = open3d.utility.Vector3dVector(all_inflate_pcds[:,:3])
    # inflate_pcd.colors = open3d.utility.Vector3dVector(all_inflate_pcds[:,3:6])
    # remains_pcd.points = open3d.utility.Vector3dVector(all_remaining_pcds[:,:3])
    # # remains_pcd.colors = open3d.utility.Vector3dVector(all_remaining_pcds[:,3:6])
    # remains_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # open3d.visualization.draw_geometries([objects_pcd])
    
    return adaptive_pcds, adaptive_prob, vote_label, vote_label_mask


## USer: below are used for dense or inflate pointcloud
def dense_pointclouds_from_bbox(raw_pointclouds, tgt_bbox, object_size, object_num=1):
    '''
        raw_pointclouds: [N*10] xyz rgb other_fts
    '''
    TOTAL_POINT_NUM = 40000
    ADAPATIVE_POINT_NUM = 10000
    TGT_OBJ_PROB = 0.5
    INFLATE_PROB = 0.3
    REMAIN_PROB = 0.2
    
    if object_size <= 0.001:
        inflate_scale = 10
    elif 0.01 >= object_size > 0.001:
        inflate_scale = 5
    elif 0.1 >= object_size > 0.01:
        inflate_scale = 2
    elif object_size > 0.1:
        inflate_scale = 1
    
    inflate_scale = int(inflate_scale/object_num) if inflate_scale/object_num > 1 else 1
    
    def _9dof_to_box(box):
        if isinstance(box, list):
            box = np.array(box)
        center = box[:3].reshape(3, 1)
        scale = box[3:6].reshape(3, 1)
        rot = box[6:].reshape(3, 1)
        rot_mat = open3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
            rot)
        geo = open3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
        geo.color = [1,0,0]
        return geo

    def point_inside_obb(point, obb, scale):
        center = obb.get_center()
        obb.scale(scale, center)
        
        # 获取定向边界框的中心点、尺寸和旋转矩阵
        center = np.array(obb.center)
        extent = np.array(obb.extent)
        R = np.array(obb.R)

        # 将点变换到局部坐标系
        point_local = np.dot((point - center), np.linalg.inv(R))

        # 检查点是否在局部坐标系内的边界框内
        for i in range(3):
            if abs(point_local[i]) > extent[i]:
                return False
        return True
        
    remaining_pcd = None
    ## tgt object 点云信息
    object_pcd_list = []
    ## tgt object 膨胀点云信息（不包括tgt obj点云）
    inflate_pcd_list = []
    ## 得到特定物体类别的点云
    tgt_obj_indices = []
    indices_within_bbox = []
    bbox_9dof = _9dof_to_box(tgt_bbox)
    
    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(raw_pointclouds[:,:3])
    # pcd.colors = open3d.utility.Vector3dVector(raw_pointclouds[:,3:6])
    # open3d.visualization.draw_geometries([pcd, bbox_9dof])
    
    for i in range(raw_pointclouds.shape[0]):
        if point_inside_obb(raw_pointclouds[i,:3], deepcopy(bbox_9dof), 1):
            tgt_obj_indices.append(i)
        elif point_inside_obb(raw_pointclouds[i,:3], deepcopy(bbox_9dof), inflate_scale):
            indices_within_bbox.append(i)
            
    tgt_obj_indices = np.array(tgt_obj_indices)
        
    ## 分离tgt obj点云和余下的点云
    remaining_indices = list(set(range(raw_pointclouds.shape[0])) - set(tgt_obj_indices) - set(indices_within_bbox))

    ## 包含物体所有点云信息
    tmp_tgt_object_pcd = raw_pointclouds[tgt_obj_indices, :] 
    object_pcd_list.append(tmp_tgt_object_pcd)
    
    tmp_inflate_pcd = raw_pointclouds[indices_within_bbox, :] 
    inflate_pcd_list.append(tmp_inflate_pcd)
    
    remaining_pcd = raw_pointclouds[remaining_indices, :] 
    
    all_tgt_object_pcds = np.concatenate(object_pcd_list, axis=0)
    all_inflate_pcds = np.concatenate(inflate_pcd_list, axis=0)
    
    tgt_object_prob = np.ones_like(all_tgt_object_pcds[:,0]) * TGT_OBJ_PROB
    inflate_prob = np.ones_like(all_inflate_pcds[:,0]) * INFLATE_PROB
    
    all_remaining_pcds = np.array(remaining_pcd)
    adaptive_pcds = np.concatenate([all_tgt_object_pcds, all_inflate_pcds], axis=0)
    # adaptive_pcds = all_tgt_object_pcds
    adaptive_prob = np.concatenate([tgt_object_prob, inflate_prob], axis=0)
    
    adaptive_pcds, choice = pc_util.random_sampling(adaptive_pcds, ADAPATIVE_POINT_NUM, return_choices=True)
    adaptive_prob = adaptive_prob[choice]
    
    point_num_offset = int(TOTAL_POINT_NUM - len(adaptive_pcds) )
    # all_remaining_pcds = _farthest_point_sampling(all_remaining_pcds, point_num_offset)
    all_remaining_pcds, choices = pc_util.random_sampling(all_remaining_pcds, point_num_offset, return_choices=True)
    remain_prob = np.ones_like(all_remaining_pcds[:,0]) * REMAIN_PROB
    
    adaptive_pcds = np.concatenate([adaptive_pcds, all_remaining_pcds], axis=0)
    adaptive_prob = np.concatenate([adaptive_prob, remain_prob], axis=0)
    # adaptive_pcds, _ = pc_util.random_sampling(np.concatenate([adaptive_pcds, all_remaining_pcds], axis=0), TOTAL_POINT_NUM, return_choices=True)
    # assert len(apdaptive_pcds) == TOTAL_POINT_NUM
    
    ## 可视化
    # objects_pcd = open3d.geometry.PointCloud()
    # objects_pcd.points = open3d.utility.Vector3dVector(adaptive_pcds[:,:3])
    # objects_pcd.colors = open3d.utility.Vector3dVector(adaptive_pcds[:,3:6])
    # center = bbox_9dof.get_center()
    # bbox_9dof.scale(inflate_scale, center)
    # open3d.visualization.draw_geometries([objects_pcd, bbox_9dof])
    
    return adaptive_pcds, adaptive_prob