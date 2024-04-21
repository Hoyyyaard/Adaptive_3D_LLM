import os, json
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_dialogue import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT

    
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
        self.task_name = '3dllm-dialogue'
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
        
        annotation_file = os.path.join(BASE, 'data', '3D_LLM', f'3d_llm_embodied_dialogue_filtered_{split_set}.json')
        self.annotations = json.load(open(annotation_file, 'r'))
        self._tag_dataset(self.annotations, 'chat')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

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
        
        # load question and answer
        question = self.annotations[idx]['question'].lower()
        answer = random.choice(self.annotations[idx]['answers'])
        
        ret_dict = self._get_scan_data(scan_name)
        prompt = {
            'instruction': question,
            'answer': answer
        }
        
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        ## ==== ground truth response
        response = prompt['answer'].format(locations='', answer=answer)
        llm_inputs = self.tokenizer.batch_encode_plus(
            [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
            **self.tokenizer_config
        )
        
        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)
        
        ## the below are for LLaMA training
        ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
        ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['gradient_mask'] = \
            (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        ## the below are for QFormer and LLaMA evaluation
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        if self.args.finetune_flex_opt or self.args.abl_ll3da_w_openscene_token:
            if self.args.use_ll3da_scene_token:
                ret_dict.update(self.ll3da_fts_cache.get_ll3da_scan_datas(scan_name))
            else:
                openscene_ret_dict = (self.openscene_fts_cache.get_openscene_scan_datas(scan_name, preprocess=self.args.token_preprocess))
                pcd = openscene_ret_dict['openscene_point_clouds']
                instance_labels = openscene_ret_dict['openscene_instance_labels']
                
                openscene_ret_dict['point_cloud_dims_min'] = pcd[..., :3].min(axis=0)
                openscene_ret_dict['point_cloud_dims_max'] = pcd[..., :3].max(axis=0)
                
                ## 1 代表不mask
                token_instance_mask = np.ones(openscene_ret_dict['scene_tokens'].shape[0]).astype(np.float32)
                openscene_ret_dict['token_instance_mask'] = token_instance_mask
                
                click_query = np.zeros((1, 3))
                click_mask = np.zeros((1,))
                openscene_ret_dict['click_query'] = click_query.astype(np.float32)
                openscene_ret_dict['click_mask'] = click_mask.astype(np.float32)
                
                openscene_ret_dict['input_ids'] = ret_dict['input_ids']
                openscene_ret_dict['attention_mask'] = ret_dict['attention_mask']
                openscene_ret_dict['gradient_mask'] = ret_dict['gradient_mask']
                
                ## below are used for both training and evaluation
                openscene_ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
                openscene_ret_dict['instruction'] = ret_dict['instruction']
                openscene_ret_dict['instruction_mask'] = ret_dict['instruction_mask']
                ret_dict = openscene_ret_dict
                del openscene_ret_dict['openscene_point_clouds']
                del openscene_ret_dict['openscene_instance_labels']
            ret_dict['scan_name'] = scan_name
        
    
        return ret_dict
   