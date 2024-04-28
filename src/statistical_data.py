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
# import open3d
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


class Preprocess_Adaptive_Pointcloud_Dataset(ScanNetBaseDataset):
    def __init__(
        self,
    ):
        super().__init__(
            None,
            DatasetConfig(),
            split_set='val',
            num_points=40000,
            use_color=True,
            use_normal=True,
            use_multiview=False,
            use_height=True,
            augment=False,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        # self.data_path = 'data/scannet/scannet_data'
        
        # anno_dict = {
        #     "scanqa" : 'data/ScanQA/ScanQA_v1.0_val.json',
        #     "scanrefer" : 'data/ScanRefer/ScanRefer_filtered_val.json',
        #     "nr3d" : 'data/Nr3D/nr3d_val.json'
        # }
        # size_statistic_dict = {
        #     "scanqa" : [],
        #     "scanrefer": [],
        #     "nr3d": []
        # }
        
        # for task_name, anno_p in anno_dict.items():
            
        #     annotations = json.load(open(anno_p, 'r'))
            
        #     for anno in tqdm(annotations):
        #         dense_ret_dict = self._get_scan_data_adaptive(anno['scene_id'])
        #         point_clouds = dense_ret_dict["point_clouds"][:, :3]  # x, y, z
                
        #         if 'object_ids' in anno.keys():
        #             target_obj_id_list = anno['object_ids']
        #         else:
        #             target_obj_id_list = [int(anno['object_id'])]
                
        #         instance_size_list = []
        #         for target_obj_id in target_obj_id_list:
        #             obj_idx = dense_ret_dict["instance_labels"] == (target_obj_id + 1)
        #             ## PLan A
        #             # object_points = point_clouds[obj_idx]
        #             # o3d_pcd = open3d.geometry.PointCloud()
        #             # o3d_pcd.points = open3d.utility.Vector3dVector(object_points)
        #             # axis_aligned_bounding_box = o3d_pcd.get_axis_aligned_bounding_box()
        #             # bbox_size = [axis_aligned_bounding_box.max_bound[i] - axis_aligned_bounding_box.min_bound[i] for i in range(len(axis_aligned_bounding_box.max_bound))]
        #             # instance_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
        #             ## PLan B
        #             instance_point_vote = dense_ret_dict['vote_label'][obj_idx][:,:3]
        #             instance_size = (instance_point_vote.max(0)-instance_point_vote.min(0))
        #             instance_size = instance_size[0] * instance_size[1] * instance_size[2]
        #             instance_size_list.append(instance_size)
        #         instance_size = instance_size.max()
        #         size_statistic_dict[task_name].append(instance_size)

        # sizes1 = size_statistic_dict['scanqa']
        # sizes2 = size_statistic_dict['scanrefer']
        # sizes3 = size_statistic_dict['nr3d']
        
        # def count_sizes(sizes):
        #     def find_nearest(array, value):
        #         idx = (np.abs(array - value)).argmin()
        #         return array[idx]
            
        #     min_size = round(min(sizes),4)
        #     max_size = round(max(sizes),4)
        #     print(min_size)
        #     print(max_size)
        #     size_counts = {size: 0 for size in np.arange(min_size - 10, max_size + 10, 1e-4)}
        #     for size in tqdm(sizes):
        #         try:
        #             size_counts[find_nearest(np.array(list(size_counts.keys())),round(size,4))] += 1
        #         except:
        #             # print(round(size,5))
        #             pass
        #     return size_counts

        # # 统计每个列表中的物体数量
        # count1 = count_sizes(sizes1)
        # np.save('results/size_statistic/scanqa.npy', count1)
        # count2 = count_sizes(sizes2)
        # np.save('results/size_statistic/scanrefer.npy', count2)
        # count3 = count_sizes(sizes3)
        # np.save('results/size_statistic/scannr3d.npy', count3)

        count1 = np.load('results/size_statistic/scanqa.npy', allow_pickle=True).tolist()
        count2 = np.load('results/size_statistic/scanrefer.npy', allow_pickle=True).tolist()
        count3 = np.load('results/size_statistic/scannr3d.npy', allow_pickle=True).tolist()
        
        xticks = [10e-5,10e-4,10e-3,10e-2, 10e-1, 10e0, 10, 10e1]
        # 将数据归类到横坐标区间中
        classified_data1 = {key: 0 for key in xticks}
        for key, value in count1.items():
            for i in range(len(xticks) - 1):
                if xticks[i] <= key < xticks[i + 1]:
                    classified_data1[xticks[i]] += value
                    break
        classified_data2 = {key: 0 for key in xticks}
        for key, value in count2.items():
            for i in range(len(xticks) - 1):
                if xticks[i] <= key < xticks[i + 1]:
                    classified_data2[xticks[i]] += value
                    break
        classified_data3 = {key: 0 for key in xticks}
        for key, value in count3.items():
            for i in range(len(xticks) - 1):
                if xticks[i] <= key < xticks[i + 1]:
                    classified_data3[xticks[i]] += value
                    break
                
        print(classified_data1)
        print(classified_data2)
        print(classified_data3)
        
        # Preparing data for plotting
        labels = [f"{key}" for key in classified_data1.keys()]
        values = list(classified_data1.values())

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.title('Scanqa')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # Display the plot
        plt.tight_layout()
        plt.show()
    
        
        labels = [f"{key}" for key in classified_data2.keys()]
        values = list(classified_data2.values())

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.title('Scanrefer')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # Display the plot
        plt.tight_layout()
        plt.show()
        
        
        labels = [f"{key}" for key in classified_data3.keys()]
        values = list(classified_data3.values())

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.title('nr3d')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # Display the plot
        plt.tight_layout()
        plt.show()
                
        
if __name__ == '__main__':
    tmp = Preprocess_Adaptive_Pointcloud_Dataset()