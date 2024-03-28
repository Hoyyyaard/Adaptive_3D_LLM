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
            "scanqa" : 'data/ScanQA/ScanQA_v1.0_val.json',
            "scanrefer" : 'data/ScanRefer/ScanRefer_filtered_val.json',
            "nr3d" : 'data/Nr3D/nr3d_val.json'
        }
        
        for task_name, anno_p in anno_dict.items():
            
            annotations = json.load(open(anno_p, 'r'))
            
            for anno in tqdm(annotations):
                st = time.time()
                dense_ret_dict = self._get_scan_data_adaptive(anno['scene_id'])
                # print(f'get_scan_data_adaptive time: {time.time()-st}')
                
                if 'object_ids' in anno.keys():
                    target_obj_id_list = anno['object_ids']
                else:
                    target_obj_id_list = [int(anno['object_id'])]
                
                target_obj_id = random.choice(target_obj_id_list)
                raw_pointclouds = dense_ret_dict["point_clouds"]
                instance_labels = dense_ret_dict["instance_labels"]
                object_num = 1
                obj_idx = instance_labels == (target_obj_id + 1)
                objects_pcd = open3d.geometry.PointCloud()
                objects_pcd.points = open3d.utility.Vector3dVector(raw_pointclouds[:,:3][obj_idx ])
                bbox = objects_pcd.get_axis_aligned_bounding_box()
                bbox_size = [bbox.max_bound[i] - bbox.min_bound[i] for i in range(len(bbox.max_bound))]
                object_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
                
                st = time.time()
                from src.utils import dense_pointclouds
                dense_point_clouds, sample_prob = dense_pointclouds(dense_ret_dict["point_clouds"], dense_ret_dict["instance_labels"], target_obj_id_list, object_size, object_num, self.num_points)
                # print(f'dense_pointclouds time: {time.time()-st}')
                
                uni_idx = anno['question_id'] if 'question_id' in anno.keys() else f'{anno["scene_id"]}_{anno["object_id"]}_{anno["object_name"]}'
                
                adaptive_pcd_op_dir = f'{self.args.cache_dir}/{task_name}/{uni_idx}'
                if not os.path.exists(adaptive_pcd_op_dir):
                    os.makedirs(adaptive_pcd_op_dir)
                
                np.save(f'{adaptive_pcd_op_dir}/adaptive_pcd.npy', dense_point_clouds)
                np.save(f'{adaptive_pcd_op_dir}/sample_prob.npy', sample_prob)
                
                
        
if __name__ == '__main__':
    tmp = Preprocess_Adaptive_Pointcloud_Dataset()