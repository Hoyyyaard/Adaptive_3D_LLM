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
        
        self.task_name = 'scanqa'
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
        
        if args.special_dataset:
            print("Use User special dataset for val !!!!!")
            annotation_file = args.special_dataset
        else:
            annotation_file = os.path.join(BASE, 'data', 'ScanQA', f'ScanQA_v1.0_{split_set}.json')
        self.annotations = json.load(open(annotation_file, 'r'))
        
        
        ## USer: below are used to filter gt objects that less than certain number of points
        # scene_ids = []
        # for anno in self.annotations:
        #     if anno['scene_id'] not in scene_ids:
        #         scene_ids.append(anno['scene_id'])
        # tmp_ret_dicts = {f'{scene_id}':self._get_scan_data(scene_id) for scene_id in scene_ids}
        # size_filter_annotations = []
        # max_point_num = 1e-9
        # min_point_num = 1e9
        # max_bbox_size = 1e-9
        # min_bbox_size = 1e9
        # thres_point_num = 50
        # thres_bbox_size = 5
        # for anno in tqdm(self.annotations,desc='filtering annotations by size'):
        #     # print(anno['question'])
        #     # print(anno['answers'])
        #     tmp_ret_dict = tmp_ret_dicts[anno['scene_id']]
        #     target_obj_id = random.choice(anno['object_ids'])
        #     point_clouds = tmp_ret_dict["point_clouds"][:, :3]  # x, y, z
        #     colors = np.ones_like(tmp_ret_dict["point_clouds"][:, 3:6])*0.5  # x, y, z
        #     obj_idx = tmp_ret_dict["instance_labels"] == (target_obj_id + 1)
        #     colors[obj_idx] = tmp_ret_dict["point_clouds"][:, 3:6][obj_idx]
        #     object_points = point_clouds[tmp_ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
        #     o3d_pcd = open3d.geometry.PointCloud()
        #     o3d_pcd.points = open3d.utility.Vector3dVector(object_points)
        #     axis_aligned_bounding_box = o3d_pcd.get_axis_aligned_bounding_box()
            
        #     # o3d_pcd.points = open3d.utility.Vector3dVector(point_clouds)
        #     # o3d_pcd.colors = open3d.utility.Vector3dVector(colors)
            
        #     # open3d.visualization.draw_geometries([o3d_pcd])
        #     bbox_size = [axis_aligned_bounding_box.max_bound[i] - axis_aligned_bounding_box.min_bound[i] for i in range(len(axis_aligned_bounding_box.max_bound))]
        #     point_clouds_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
            
        #     # instance_point_vote = tmp_ret_dict['vote_label'][obj_idx][:,:3]
        #     # instance_size = (instance_point_vote.max(0)-instance_point_vote.min(0))
        #     # instance_size = instance_size[0] * instance_size[1] * instance_size[2]
            
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
        #     if point_clouds_size <= 5e-3:
        #         size_filter_annotations.append(anno)
        # print(f'split:{split_set} filter annotation num:{len(size_filter_annotations)}/{len(self.annotations)}')

        self._tag_dataset(self.annotations, 'qa')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

        ## USer
        self.dense_train_info = {}
        self.dense_ret_dicts = {}
        from src.openscene_dense_pcd_fts_cache import OpenScene_Fts_Cache, LL3DA_Fts_Cache
        self.openscene_fts_cache = OpenScene_Fts_Cache(cache_dir=args.openscene_cache_dir)
        self.ll3da_fts_cache = LL3DA_Fts_Cache()
    
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
        
        scan_name = self.annotations[idx]['scene_id']
        task_name = self.annotations[idx]['task_name']
        qs_id = self.annotations[idx]['question_id']
        ret_dict = self._get_scan_data(scan_name)
        
        # load question and answer
        question = self.annotations[idx]['question'].lower()
        answer = random.choice(self.annotations[idx]['answers'])
        
        ## USer: below is used for only input answer related pcs to model
        if self.args.adaptive_pcd_input:
            cache_dir = f'{self.args.cache_dir}/scanqa'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            exist_npy = os.listdir(cache_dir)
            exist_npy = [npy.split(".")[0] for npy in exist_npy]
            uni_key = f"{scan_name}_{qs_id}"
            cache_path = f'{cache_dir}/{uni_key}'

            dense_ret_dict = self._get_scan_data_adaptive(scan_name)
            
            raw_pointclouds = dense_ret_dict["point_clouds"]
            instance_labels = dense_ret_dict["instance_labels"]
            target_obj_id = random.choice(self.annotations[idx]['object_ids'])
            object_num = len(self.annotations[idx]['object_ids'])
            obj_idx = instance_labels == (target_obj_id + 1)
            objects_pcd = open3d.geometry.PointCloud()
            objects_pcd.points = open3d.utility.Vector3dVector(raw_pointclouds[:,:3][obj_idx])
            bbox = objects_pcd.get_axis_aligned_bounding_box()
            bbox_size = [bbox.max_bound[i] - bbox.min_bound[i] for i in range(len(bbox.max_bound))]
            object_size = bbox_size[0] * bbox_size[1] * bbox_size[2]
            
            from src.utils import dense_pointclouds
            if uni_key in exist_npy:
                ret_dict = np.load(f'{cache_path}.npy',allow_pickle=True).tolist()
            else:
                ret_dict = dense_pointclouds(dense_ret_dict["point_clouds"], dense_ret_dict["instance_labels"], self.annotations[idx]['object_ids'], object_size, object_num, self.dataset_config, scan_name, self.center_normalizing_range)
                np.save(cache_path, ret_dict)

        ## ==== reference object
        target_obj_id = np.asarray(self.annotations[idx]['object_ids'])
        match_mask = ret_dict["gt_object_ids"][:, None] == target_obj_id[None, :]   # NUM_MAX_OBJ x nobj
        match_mask = (match_mask.sum(-1) > 0).astype(np.float32)                    # NUM_MAX_OBJ
        match_mask = match_mask * ret_dict["gt_box_present"]
        boxes = self._encode_box_coords(match_mask, ret_dict)
        
        # build prompts
        if self.split == 'train':
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            prompt = deepcopy(TASK_PROPMT[task_name][0])
            boxes = ''
        prompt['instruction'] = prompt['instruction'].format(locations=boxes, question=question)
        
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        ## ==== ground truth response
        response = prompt['answer'].format(locations=boxes, answer=answer)
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        
        if self.split == 'train' and random.random() < 0.25 and not self.args.abl_ll3da_w_openscene_token:
            try:
                target_obj_id = random.choice(self.annotations[idx]['object_ids'])
                try:
                    point_clouds = ret_dict["point_clouds"][:, :3]  # x, y, z
                    object_points = point_clouds[ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
                    click_query[0] = random.choice(object_points)
                except:
                    match_mask = (ret_dict["gt_object_ids"] == target_obj_id).astype(np.float32)
                    match_mask = match_mask * ret_dict["gt_box_present"]
                    click_query[0] = ret_dict["gt_box_centers"][match_mask == 1].reshape(3,).astype(np.float32)
                click_mask[0] = 1
            except Exception as e:
                print(e)
                click_mask[0] = 0

        
        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)
        
        ## below are used for training only
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        ## below are used for both training and evaluation
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        
        # objects_pcd = open3d.geometry.PointCloud()
        # objects_pcd.points = open3d.utility.Vector3dVector(ret_dict['point_clouds'][:,:3])
        # objects_pcd.colors = open3d.utility.Vector3dVector(ret_dict['point_clouds'][:,3:6])
        # open3d.visualization.draw_geometries([objects_pcd])
        
        if self.args.finetune_flex_opt or self.args.abl_ll3da_w_openscene_token:

            openscene_ret_dict = (self.openscene_fts_cache.get_openscene_scan_datas(scan_name, preprocess=self.args.token_preprocess))
            pcd = openscene_ret_dict['openscene_point_clouds']
            instance_labels = openscene_ret_dict['openscene_instance_labels']
            
            openscene_ret_dict['point_cloud_dims_min'] = pcd[..., :3].min(axis=0)
            openscene_ret_dict['point_cloud_dims_max'] = pcd[..., :3].max(axis=0)
            
            click_query = np.zeros((1, 3))
            click_mask = np.zeros((1,))
            
            ## 1 代表不mask
            token_instance_mask = np.ones(openscene_ret_dict['scene_tokens'].shape[0]).astype(np.float32)
            if self.split == 'train' and random.random() < 0.25:
                for target_obj_id in self.annotations[idx]['object_ids']:
                    # target_obj_id = random.choice(self.annotations[idx]['object_ids'])
                    has_instance = np.unique(openscene_ret_dict['token_instance_label'])
                    if target_obj_id + 1 in has_instance:
                        token_instance_mask[openscene_ret_dict['token_instance_label'] == target_obj_id + 1] = 0
                if (token_instance_mask == 0).sum() > 0:
                    ## reverse mask
                    token_instance_mask = 1 - token_instance_mask
                    ## aug
                    # total_activate_token_num = 50
                    # activate_token_num = (token_instance_mask == 1).sum()
                    # if total_activate_token_num > activate_token_num:
                    #     zero_index = np.where(token_instance_mask == 0)[0]
                    #     select_zero_index = np.random.choice(zero_index, total_activate_token_num-activate_token_num, replace=False)
                    #     token_instance_mask[select_zero_index] = 1
            
                try:
                    object_points = pcd[instance_labels == (target_obj_id + 1)]    # npt x 3
                    click_query[0] = random.choice(object_points)
                    click_mask[0] = 1
                except:pass
                
            openscene_ret_dict['token_instance_mask'] = token_instance_mask
            
            openscene_ret_dict['click_query'] = click_query.astype(np.float32)
            openscene_ret_dict['click_mask'] = click_mask.astype(np.float32)
            
            openscene_ret_dict['input_ids'] = ret_dict['input_ids']
            openscene_ret_dict['attention_mask'] = ret_dict['attention_mask']
            openscene_ret_dict['gradient_mask'] = ret_dict['gradient_mask']
            
            ## below are used for both training and evaluation
            openscene_ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
            openscene_ret_dict['instruction'] = ret_dict['instruction']
            openscene_ret_dict['instruction_mask'] = ret_dict['instruction_mask']
            
            del openscene_ret_dict['openscene_point_clouds']
            del openscene_ret_dict['openscene_instance_labels']
            
            ret_dict = openscene_ret_dict
        
        ret_dict['scan_name'] = scan_name
        if self.args.preprocess_dense_token:
            ret_dict['task_name'] = 'scanqa'
            ## TODO: 这里处理只选其中一个物体可能会导致性能下降
            ret_dict['tgt_obj_id'] = np.array(random.choice(self.annotations[idx]['object_ids'])).astype(np.int64)
        
        ## 为LL3DA-FLEX准备每个EPISODE的GT DENSE TOKEN
        if self.args.use_gt_dense_token and self.split == 'train':
            cache_dir = f"results/process_datasets/ll3da_flex_gt_dense_token/{self.split}/scanqa/{scan_name.split('_')[0]}/{idx}.pt"
            flex_gt_dense_token = torch.load(cache_dir, map_location='cpu').numpy().astype(np.float32)
            ret_dict['flex_gt_dense_token'] = flex_gt_dense_token

        # del ret_dict["instance_labels"]
        
        return ret_dict
   
   



