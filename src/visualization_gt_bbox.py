import sys
sys.path.append('/home/dell/Projects/LL3DA')
import re
import os, json, random
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_ovdet import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT
from plyfile import PlyData, PlyElement


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
        self.task_name = 'scanrefer_ov_det'
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
        
        self.scanrefer = json.load(
            open(os.path.join(BASE, 'data', 'ScanRefer', f'ScanRefer_filtered_{split_set}.json'), 'r')
        )
        
        with open(os.path.join(BASE, 'data', 'ScanRefer', f'ScanRefer_filtered_{split_set}.txt'), 'r') as f:
            self.scan_names = f.read().splitlines()
        
        self.annotations = self.scanrefer
        if self.split != 'train':
            self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
        self._tag_dataset(self.annotations, 'ov-det')
        
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
    
    def _decode_box_coords(self, pred_str, ret_dict):

        match = re.search(r'<obj>(.*?)</obj>', pred_str)
        numbers_str = match.group(1)
        numbers_list = numbers_str.split(', ')
        numbers = [int(num) for num in numbers_list]
        numbers_array = np.array(numbers)
        box_normalized = numbers_array / self.grid_size_3d

        box_centers_normalized = box_normalized[0:3]
        box_size_normalized = box_normalized[3:6]

        point_cloud_dims_min = ret_dict['point_cloud_dims_min']
        point_cloud_dims_max = ret_dict['point_cloud_dims_max']
        point_cloud_len = point_cloud_dims_max - point_cloud_dims_min

        box_centers = box_centers_normalized * point_cloud_len + point_cloud_dims_min
        box_size = box_size_normalized * point_cloud_len
        box = np.hstack((box_centers, box_size)) 

        return box
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        scan_name = self.annotations[idx]['scene_id']
        task_name = self.annotations[idx]['task_name']
        ret_dict = self._get_scan_data(scan_name)
        
        if self.split == 'train':
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            prompt = deepcopy(TASK_PROPMT[task_name][0])

        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        if self.split == 'train':
            
            target_obj_id = int(self.annotations[idx]['object_id'])
            category_name = ' '.join(self.annotations[idx]['object_name'].split('_'))
            caption = ' '.join(self.annotations[idx]['token'])
            
            ## reference object
            match_mask = (ret_dict["gt_object_ids"] == target_obj_id).astype(np.float32)
            match_mask = match_mask * ret_dict["gt_box_present"]
            vg_gt_box_corners = ret_dict["gt_box_corners"][match_mask==1]
            vg_box_angles = ret_dict["gt_box_angles"][match_mask==1]

            # <obj>cx, cy, cz, w, h, l</obj>
            vg_gt_box_discretized = self._encode_box_coords(match_mask, ret_dict)
            # cx, cy, cz, w, h, l
            vg_gt_box_recover_from_discretized = self._decode_box_coords(vg_gt_box_discretized, ret_dict)
            vg_gt_box_corners_recover_from_discretized = self.dataset_config.box_parametrization_to_corners_np(
                                                            vg_gt_box_recover_from_discretized[0:3][None, ...],
                                                            vg_gt_box_recover_from_discretized[3:6][None, ...],
                                                            vg_box_angles.astype(np.float32)[None, ...],
                                                                        )

            response = prompt['answer'].format(category=category_name, locations=vg_gt_box_discretized, caption=caption)
            
            ## input_ids as labels for LLM
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            box_query = np.zeros((self.max_prompts, 8, 3))
            box_mask = np.zeros((self.max_prompts,))
            click_query = np.zeros((1, 3))
            click_mask = np.zeros((1,))
            
            # (M, 8, 3)
            ref_gt_box_corner = \
                ret_dict["gt_box_corners"][ret_dict["gt_box_present"] == 1].astype(np.float32)
            
            object_numbers = ref_gt_box_corner.shape[0]
            if object_numbers > self.max_prompts:
                indices = np.random.choice(object_numbers, self.max_prompts, replace=False)
                ref_gt_box_corner = ref_gt_box_corner[indices]

            num_boxes = min(object_numbers, self.max_prompts)
            box_query[:num_boxes] = ref_gt_box_corner[:num_boxes]
            box_mask[:num_boxes] = 1

            # box_query[0] = ref_gt_box_corner
            # box_mask[0] = 1

            # try:
            #     point_clouds = ret_dict["point_clouds"][:, :3]  # x, y, z
            #     object_points = point_clouds[ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
            #     click_query[0] = random.choice(object_points)
            # except:
            #     click_query[0] = ret_dict["gt_box_centers"][match_mask == 1].reshape(3,).astype(np.float32)
            # click_mask[0] = 1

            ret_dict['vg_gt_box_corners'] = vg_gt_box_corners.astype(np.float32)
            ret_dict['vg_gt_box_discretized'] = vg_gt_box_discretized
            ret_dict['vg_gt_box_corners_recover_from_discretized'] = vg_gt_box_corners_recover_from_discretized.astype(np.float32)
            ret_dict['vg_object_name'] = category_name

            ret_dict['box_query'] = box_query.astype(np.float32)
            ret_dict['box_mask'] = box_mask.astype(np.float32)
            ret_dict['click_query'] = click_query.astype(np.float32)
            ret_dict['click_mask'] = click_mask.astype(np.float32)
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        return ret_dict
    
import argparse
def make_args_parser():
    parser = argparse.ArgumentParser("LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    # DISABLE warmup learning rate during dense caption training
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    # only ACTIVATE during dense caption training
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    parser.add_argument("--pretrained_weights", default=None, type=str)
    
    
    ##### Model #####
    # input based parameters
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    parser.add_argument(
        "--detector", default="detector_Vote2Cap_DETR", 
        help="folder of the detector"
    )
    parser.add_argument(
        "--captioner", default=None, type=str, help="folder of the captioner"
    )
    # training strategy
    parser.add_argument(
        "--freeze_detector", default=False, action='store_true', 
        help="freeze all parameters other than the caption head"
    )
    parser.add_argument(
        "--freeze_llm", default=False, action='store_true', 
        help="freeze the llm for caption generation"
    )
    # caption related hyper parameters
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument(
        "--max_des_len", default=128, type=int, 
        help="maximum length of object descriptions."
    )
    parser.add_argument(
        "--max_gen_len", default=32, type=int, 
        help="maximum length of object descriptions."
    )
    
    ##### Dataset #####
    parser.add_argument("--max_prompts", default=16, type=int, help="number of visual interactions")
    parser.add_argument("--dataset", default='scannet', help="dataset list split by ','")
    parser.add_argument("--grid_size_3d", default=255, type=int, help="grid size of the 3D scene")    
    parser.add_argument('--vocab', default="llama-hf/7B", type=str, help="The LLM backend")
    parser.add_argument('--qformer_vocab', default="bert-base-uncased", type=str, help="The QFormer backend")
    
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)
    parser.add_argument("--eval_every_iteration", default=4000, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument(
        "--criterion", default='CiDEr', type=str,
        help='metrics for saving the best model'
    )
    parser.add_argument("--test_ckpt", default="", type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--save_every", default=4000, type=int)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--filter_name", default='captioner.transformer.', type=str)
    
    ##### Distributed #####
    parser.add_argument("--ngpus", default=1, type=int, help='number of gpus')
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)
    
    args = parser.parse_args()
    args.use_height = not args.no_height
    
    return args

def write_ply(verts, colors, indices, output_file):
        if colors is None:
            colors = np.zeros_like(verts)
        if indices is None:
            indices = []

        file = open(output_file, 'w')
        file.write('ply \n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {:d}\n'.format(len(verts)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('element face {:d}\n'.format(len(indices)))
        file.write('property list uchar uint vertex_indices\n')
        file.write('end_header\n')
        for vert, color in zip(verts, colors):
            file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        for ind in indices:
            file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
        file.close()

def write_bbox_corners(corners, mode, output_file):
    """
    bbox: (8 * 3)
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges
    
    
    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0], # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)

def write_ply_rgb(points, colors, filename, text=True):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """

    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if points.shape[1] == 4: # the first dimension is b_id
         points = points[:, 1:]

    if isinstance(colors, torch.Tensor):
        colors = colors.numpy()

    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)  

def write_bbox(bbox, mode, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0) # 8 x 3

        return corners

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0], # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)


if __name__ == "__main__":

    args = make_args_parser()
    
    def count_max_object_number_in_a_scene():
        from datasets.scannet_base_dataset import DatasetConfig
        dataset_config = DatasetConfig()
        args = make_args_parser()

        dataset = Dataset(args, dataset_config)

        obj_num_list = []
        from tqdm import tqdm
        for i in tqdm(range(len(dataset))):
            data = dataset.__getitem__(i)
            obj_num = data['gt_box_present'].sum()
            if obj_num not in obj_num_list:
                obj_num_list.append(obj_num)
        pass
    
    def vis_box_and_vg_box():
        from datasets.scannet_base_dataset import DatasetConfig
        MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

        dataset_config = DatasetConfig()
        args = make_args_parser()
        args.max_prompts = 100
        args.use_color = True
        args.use_normal = True

        dataset = Dataset(args, dataset_config, use_color=True, use_normal=True, augment=False)
        save_dir_base = "results/temp/visualization/vg"

        from tqdm import tqdm
        for i in tqdm(range(len(dataset))):
            data = dataset.__getitem__(i)
            idx = int(data["scan_idx"])
            save_dir = os.path.join(save_dir_base, str(idx))
            os.makedirs(save_dir, exist_ok=True)

            xyz = data['point_clouds'][:, 0:3]
            rgb = data['point_clouds'][:, 3:6] * 256 + MEAN_COLOR_RGB
            scene_save_path = os.path.join(save_dir, "scene.ply")
            if not os.path.exists(scene_save_path):
                write_ply_rgb(xyz, rgb, scene_save_path)
            
            for j in range(data['box_query'].shape[0]):
                # box_corner = data['gt_box_corners'][j]
                # box_centers = data['gt_box_centers'][j]
                # box_sizes = data['gt_box_sizes'][j]
                # box = np.hstack([box_centers, box_sizes])
                # box_save_path = os.path.join(save_dir, "obj_{}.ply".format(j))
                # if box_corner.mean() != 0:
                #     write_bbox(box, mode=0, output_file=box_save_path)

                box_corner = data['box_query'][j]
                box_save_path = os.path.join(save_dir, "obj_{}.ply".format(j))
                if box_corner.mean() != 0:
                    write_bbox_corners(box_corner, mode=0, output_file=box_save_path)
            
            vg_gt_box_corners = data['vg_gt_box_corners']
            vg_object_name = data['vg_object_name']
            vg_save_path = os.path.join(save_dir, "vg_{}.ply".format(vg_object_name))
            write_bbox_corners(vg_gt_box_corners[0], mode=1, output_file=vg_save_path)

            if i >= 0:
                exit()
            

    vis_box_and_vg_box()
            

