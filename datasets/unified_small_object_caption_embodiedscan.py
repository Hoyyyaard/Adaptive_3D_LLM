import os, json, random
import torch
import sys
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_object_caption import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT
from tqdm import tqdm
# import open3d
import utils.pc_util as pc_util


class Dataset(ScanNetBaseDataset):
    
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        self.args = args
        
        self.task_name = 'object_caption'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.split = split_set
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len
        self.eval_func = evaluate
        
        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        ## load annotations
        assert split_set in ["train", "val"]
        
        self.scanrefer = json.load(open('results/train_small_than_1e-3_wdes.json', 'r'))
            
        
        self.annotations = self.scanrefer
        for i in range(len(self.annotations)):
            self.annotations[i]['global_ann_id'] = str(i)
        ## USer: below is not compatible with the generate size filter dataset
        # if self.split != 'train' and not args.special_dataset:
        #     self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
        # elif self.split != 'train' and args.special_dataset:
        #     scene_ids = []
        #     for anno in self.annotations:
        #         if anno['scene_id'] not in scene_ids:
        #             scene_ids.append(anno['scene_id'])
        #     self.annotations = [{'scene_id': scene_id} for scene_id in scene_ids]
        # if self.split != 'train':
        #     self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
            
        ## USer: below are used to filter gt objects that less than certain number of points
        
        # tmp_ret_dicts = {scene_id:self._get_scan_data(scene_id) for scene_id in self.scan_names}
        # size_filter_annotations = []
        # max_point_num = 1e-9
        # min_point_num = 1e9
        # max_bbox_size = 1e-9
        # min_bbox_size = 1e9
        # thres_point_num = 50
        # thres_bbox_size = 5e-2
        # for anno in tqdm(self.scanrefer,desc='filtering annotations by size'):
        #     tmp_ret_dict = tmp_ret_dicts[anno['scene_id']]
        #     target_obj_id = int(anno['object_id'])
        #     point_clouds = tmp_ret_dict["point_clouds"][:, :3]  # x, y, z
        #     object_points = point_clouds[tmp_ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
        #     o3d_pcd = open3d.geometry.PointCloud()
        #     o3d_pcd.points = open3d.utility.Vector3dVector(object_points)
        #     axis_aligned_bounding_box = o3d_pcd.get_axis_aligned_bounding_box()
        #     bbox_size = [axis_aligned_bounding_box.max_bound[i] - axis_aligned_bounding_box.min_bound[i] for i in range(len(axis_aligned_bounding_box.max_bound))]
        #     point_clouds_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
        #     if point_clouds_size > max_bbox_size:
        #         max_bbox_size = point_clouds_size
        #     if point_clouds_size < min_bbox_size:
        #         min_bbox_size = point_clouds_size
        #     point_nums = len(object_points)
        #     if point_nums > max_point_num:
        #         max_point_num = point_nums
        #     if point_nums < min_point_num:
        #         min_point_num = point_nums
        #     # if point_nums <= thres_point_num:
        #     #     size_filter_annotations.append(anno)
        #     if point_clouds_size <= thres_bbox_size:
        #         size_filter_annotations.append(anno)
        # print(f'split:{split_set} filter annotation num:{len(size_filter_annotations)}/{len(self.annotations)}')    
            
        
        self._tag_dataset(self.annotations, 'densecap')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

    
    def _tag_dataset(self, corpus, task_name): 
        for anno in corpus:
            anno['task_name'] = task_name
        return 
    
    def _encode_box_coords(self, annotation_mask, ret_dict):
        center_normalized = ret_dict['gt_box_centers_normalized']
        size_normalized = ret_dict['gt_box_sizes_normalized']
        box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = box_normalized[annotation_mask == 1]
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        scan_name = self.annotations[idx]['scan_id'].split('/')[-1]
        object_id = self.annotations[idx]['target_id']
        object_name = self.annotations[idx]['target']
        # print(object_name)
        # print(scan_name)
        task_name = self.annotations[idx]['task_name']
        ret_dict = self._get_scan_data(scan_name)
        
        if self.split == 'train':
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            prompt = deepcopy(TASK_PROPMT[task_name][0])
            
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        tgt_box = self.annotations[idx]['tgt_bbox']
        ## USer: below is used for only input answer related pcs to model
        if self.args.adaptive_pcd_input:
            try:
                # cache_dir = 'results/process_datasets/adaptive_pcds_adapt_scale_1w/small_object_caption'
                # if not os.path.exists(cache_dir):
                #     os.makedirs(cache_dir)
                # exist_npy = os.listdir(cache_dir)
                # exist_npy = [npy.split(".")[0] for npy in exist_npy]
                # uni_key = f"{scan_name}_{object_id}_{object_name}"
                # cache_path = f'{cache_dir}/{uni_key}'

                # dense_ret_dict = self._get_scan_data_adaptive(scan_name)
                source_dir = 'data/scannet/scannet_data'
                mesh_vertices = np.load(os.path.join(f'{source_dir}_dense', scan_name) + "_aligned_vert.npy")

                point_cloud = mesh_vertices[:, 0:6]
                MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
                point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
                normals = mesh_vertices[:,6:9]
                point_cloud = np.concatenate([point_cloud, normals], 1)
                    
                floor_height = np.percentile(point_cloud[:, 2], 0.99)
                height = point_cloud[:, 2] - floor_height
                point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
                
                
                from utils_adaptive import dense_pointclouds_from_bbox
                # if uni_key in exist_npy:
                #     ret_dict['point_clouds'], ret_dict['sample_prob'] = np.load(f'{cache_path}.npy',allow_pickle=True).tolist()['point_clouds'], np.load(f'{cache_path}.npy',allow_pickle=True).tolist()['sample_prob']
                # else:
                ret_dict['point_clouds'], ret_dict['sample_prob'] = dense_pointclouds_from_bbox(point_cloud, tgt_box , tgt_box[3]*tgt_box[4]*tgt_box[5])
                # np.save(cache_path, {'point_clouds':ret_dict['point_clouds'], 'sample_prob':ret_dict['sample_prob']})
                
                ret_dict["point_cloud_dims_min"] = ret_dict["point_clouds"][..., :3].min(axis=0)
                ret_dict["point_cloud_dims_max"] = ret_dict["point_clouds"][..., :3].max(axis=0)
            except Exception as e:
                ret_dict['sample_prob'] = np.ones_like(ret_dict['point_clouds'][:,0])
                print(e)
        
        
        ## reference object

        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        tgt_box = np.array([tgt_box]).astype(np.float32)
        box_centers = tgt_box[:, 0:3]
        raw_sizes = tgt_box[:, 3:6] ## [w,h,l] of tgt objects
        raw_angles = np.zeros((1,), dtype=np.float32)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)
        
        if self.args.caption_box_query:
            # use box to identify an object
            ref_gt_box_corner = box_corners.astype(np.float32)
            box_query[0] = ref_gt_box_corner
            box_mask[0] = 1
        else:
            # use click to identify an object
            click_query[0] = box_centers.reshape(3,).astype(np.float32)
            click_mask[0] = 1
        
        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)
            
        return ret_dict
   