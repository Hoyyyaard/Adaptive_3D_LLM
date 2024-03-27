import os, json
import torch
import sys
sys.path.append('/home/admin/Projects/LL3DA/')
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_qa import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT
from tqdm import tqdm
import utils.pc_util as pc_util
import open3d
import matplotlib.pyplot as plt
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scanqa', help='dataset name')
    return parser.parse_args()

args = parse_args()

## USer: below are used for dense or inflate pointcloud
def dense_pointclouds(raw_pointclouds, instance_pointclouds, target_obj_id_list, scale):
    '''
        raw_pointclouds: [N*10] xyz rgb other_fts
        instance_pointclouds: [N*1] 
    '''
    TOTAL_POINT_NUM = 40000
    ADAPATIVE_POINT_NUM = 10000
    TGT_OBJ_PROB = 0.5
    INFLATE_PROB = 0.3
    REMAIN_PROB = 0.2
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
    ## tgt object 膨胀点云信息（不包括tgt obj点云）
    inflate_pcd_list = []
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
        
        ## 选择剩余点云中膨胀后边界框内点
        st = time.time()
        indices_within_bbox = _get_inflate_axis_aligned_bounding_box(tmp_tgt_object_pcd[:,:3], remaining_pcd[:,:3])
        print(f'_get_inflate_axis_aligned_bounding_box time: {time.time()-st}')
        
        tmp_inflate_pcd = remaining_pcd[indices_within_bbox, :] 
        inflate_pcd_list.append(tmp_inflate_pcd)
        
        ## 在剩余点云中剔除膨胀点云
        remaining_index = range(len(remaining_pcd))
        remaining_index = [idx for idx in remaining_index if not idx in indices_within_bbox]
        remaining_pcd = remaining_pcd[remaining_index, :]
    
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
    
    return adaptive_pcds, adaptive_prob

class Preprocess_Adaptive_Pointcloud_Dataset(ScanNetBaseDataset):
    def __init__(
        self,
    ):
        super().__init__(
            None,
            DatasetConfig(),
            split_set='train',
            num_points=40000,
            use_color=True,
            use_normal=True,
            use_multiview=False,
            use_height=True,
            augment=False,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        self.data_path = 'data/scannet/scannet_data'
        
        anno_dict = {
            "obecjt_caption" : 'data/ScanObjectCaption/ScanRefer_filtered_val_qa_format.json',
        }
        
        for task_name, anno_p in anno_dict.items():
            
            annotations = json.load(open(anno_p, 'r'))
            
            for anno in tqdm(annotations):
                
                ret_dcit = self._get_scan_data(anno['scene_id'])
                target_obj_id = int(anno['object_id'])
                
                match_mask = (ret_dcit["gt_object_ids"] == target_obj_id).astype(np.float32)
                match_mask = match_mask * ret_dcit["gt_box_present"]
        
                # use box to identify an object
                ref_gt_box_corner = \
                    ret_dcit["gt_box_corners"][match_mask == 1].reshape(8, 3).astype(np.float32)

                # 定义边界框的边，即点之间的连接
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]
                ]

                # 根据点和线创建线段集合
                colors = [[1, 0, 0] for i in range(len(lines))]  # 定义所有线条的颜色，这里设置为红色
                line_set = open3d.geometry.LineSet(
                    points=open3d.utility.Vector3dVector(ref_gt_box_corner),
                    lines=open3d.utility.Vector2iVector(lines),
                )
                line_set.colors = open3d.utility.Vector3dVector(colors)
                
                
                dense_ret_dict = self._get_scan_data_adaptive(anno['scene_id'])
                raw_pointclouds = dense_ret_dict["point_clouds"]
                instance_labels = dense_ret_dict["instance_labels"]
                
                obj_idx = instance_labels == (target_obj_id + 1)
                objects_pcd = open3d.geometry.PointCloud()
                objects_pcd.points = open3d.utility.Vector3dVector(raw_pointclouds[:,:3][obj_idx ])
                bbox = objects_pcd.get_axis_aligned_bounding_box()
                bbox_size = [bbox.max_bound[i] - bbox.min_bound[i] for i in range(len(bbox.max_bound))]
                object_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
                uid = anno['global_ann_id']
                object_name = anno['object_name']
                scene_id = anno['scene_id']
                print(f'Scene: {scene_id} Unique id: {uid} Object name: {object_name} Object size: {object_size}')
                
                if object_size <= 0.01:
                    bbox_wireframe = open3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                    bbox_wireframe.paint_uniform_color([1, 0, 0]) 
                    objects_pcd.points= open3d.utility.Vector3dVector(ret_dcit['point_clouds'][:,:3])
                    objects_pcd.colors= open3d.utility.Vector3dVector(ret_dcit['point_clouds'][:,3:6])
                    # open3d.visualization.draw_geometries([objects_pcd, line_set])
                    open3d.visualization.draw_geometries([objects_pcd, bbox_wireframe])
                    
                    if object_size <= 0.001:
                        scale = 10
                    elif 0.01 >= object_size > 0.001:
                        scale = 5
                    elif 0.1 >= object_size > 0.01:
                        scale = 2
                    elif object_size > 0.1:
                        scale = 1
                    
                    
                    import copy
                    center = bbox.get_center()
                    sacle_bbox = copy.deepcopy(bbox)
                    sacle_bbox.scale(scale, center)
                    sacle_bbox_wireframe = open3d.geometry.LineSet.create_from_axis_aligned_bounding_box(sacle_bbox)
                    sacle_bbox_wireframe.paint_uniform_color([0, 1, 0]) 
                    
                    raw_pointclouds, _ = dense_pointclouds(dense_ret_dict["point_clouds"], dense_ret_dict["instance_labels"], [int(anno['object_id'])], scale)
                    objects_pcd.points= open3d.utility.Vector3dVector(raw_pointclouds[:,:3])
                    objects_pcd.colors= open3d.utility.Vector3dVector(raw_pointclouds[:,3:6])
                    open3d.visualization.draw_geometries([objects_pcd, bbox_wireframe, sacle_bbox_wireframe])
        
if __name__ == '__main__':
    tmp = Preprocess_Adaptive_Pointcloud_Dataset()