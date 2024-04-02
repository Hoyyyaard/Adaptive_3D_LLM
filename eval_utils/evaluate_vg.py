import os, time, json
import torch
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    is_primary, 
    barrier,
    all_gather_dict
)

import re
import numpy as np
from plyfile import PlyData, PlyElement

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

@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    
    # prepare ground truth caption labels
    print("Evaluating visual grounding ...")
    scene_list = dataset_loader.dataset.scan_names
    # corpus, object_id_to_name = prepare_corpus(dataset_loader.dataset.scanrefer)
    task_name = dataset_loader.dataset.task_name
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    barrier()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    eval_metrics = {
        "mIoU": 0.0,
        "acc@0.25": 0.0,
        "acc@0.50": 0.0,
    }

    IoU = 0.0
    p_25 = 0.0
    p_50 = 0.0

    IoU_gt_noise = 0.0
    p_25_gt_noise = 0.0
    p_50_gt_noise = 0.0

    IoU_gt= 0.0
    p_25_gt = 0.0
    p_50_gt = 0.0
    
    total_time = 0
    count = 0
    from tqdm import tqdm
    pbar = tqdm(total=len(dataset_loader))
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        pbar.update(1)
        begin_time = time.time()
        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask': batch_data_label['qformer_attention_mask'],
            'box_query': batch_data_label['box_query'],
            'click_query': batch_data_label['click_query'],
        }

        vis_pc = batch_data_label['point_clouds'].clone().cpu().numpy()
        
        vis_pc[:, :, [0,1,2]] = vis_pc[:, :, [0,2,1]]
        vis_pc[:, :, 1] *= -1

        coords = vis_pc[:, :, 0:3]
        colors = vis_pc[:, :, 3:6]
        # (B, N, 8, 3)
        query_box = batch_data_label['box_query'].clone().cpu().numpy()

        outputs = model(model_input, is_eval=True, task_name='vg')

        outputs = dict(
            output_ids=outputs["output_ids"],
        )
        
        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)

        output_ids = outputs["output_ids"]  # batch x nqueries x max_length
        batch_size = output_ids.shape[0]
        captions = tokenizer.batch_decode(
            output_ids.reshape(-1, output_ids.shape[-1]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        ### replace box corners
        caption_box_coords = np.zeros((len(captions), 7))
        caption_box_mask = np.zeros((len(captions),))
        
        for cap_id, caption in enumerate(captions):
            try:
                # try to decode the caption into 3D boxes
                coord_str = re.findall(r'<obj>(.*?)</obj>', caption)[0]
                x, y, z, w, h, l = map(float, coord_str.split(','))
                caption_box_coords[cap_id, :6] = x, y, z, w, h, l
                caption_box_mask[cap_id] = 1
            except:
                continue
        
        point_cloud_dims_min = batch_data_label['point_cloud_dims_min']
        point_cloud_dims_max = batch_data_label['point_cloud_dims_max']
        
        caption_box_coords = caption_box_coords.reshape(batch_size, 1, 7)
        caption_box_coords = torch.from_numpy(caption_box_coords).to(net_device)
        caption_box_mask = caption_box_mask.reshape(batch_size, 1)
        caption_box_mask = torch.from_numpy(caption_box_mask).to(net_device)
        
        # batch x 1 x 7
        caption_box_coords = caption_box_coords / args.grid_size_3d
        caption_box_center = caption_box_coords[..., :3]
        caption_box_size = caption_box_coords[..., 3:6]
        
        scene_scale = (point_cloud_dims_max - point_cloud_dims_min).reshape(batch_size, 1, 3)
        scene_floor = point_cloud_dims_min.reshape(batch_size, 1, 3)
        caption_box_center = caption_box_center * scene_scale + scene_floor
        caption_box_size = caption_box_size * scene_scale
        caption_box_angle = caption_box_coords[..., -1].reshape(batch_size, 1)
        
        # batch x 1 x 8 x 3
        caption_box_corners = dataset_config.box_parametrization_to_corners(
            caption_box_center,     # batch x 
            caption_box_size, 
            caption_box_angle
        )

        # (batch, )
        match_box_ious = box3d_iou_batch_tensor(
            (caption_box_corners \
             .unsqueeze(2) \
             .repeat(1, 1, 1, 1, 1) \
             .view(-1, 8, 3) 
             ),
            (batch_data_label['vg_gt_box_corners'] \
             .unsqueeze(1) \
             .repeat(1, 1, 1, 1, 1) \
             .view(-1, 8, 3)
             )
        )
        match_box_ious_gt_discretized = box3d_iou_batch_tensor(
            (batch_data_label['vg_gt_box_corners_recover_from_discretized'].squeeze(1) \
             .unsqueeze(2) \
             .repeat(1, 1, 1, 1, 1) \
             .view(-1, 8, 3) 
             ),
            (batch_data_label['vg_gt_box_corners'] \
             .unsqueeze(1) \
             .repeat(1, 1, 1, 1, 1) \
             .view(-1, 8, 3)
             )
        )

        # match_box_ious_gt = box3d_iou_batch_tensor(
        #     (torch.from_numpy(query_box).to(batch_data_label['vg_gt_box_corners'].device) \
        #      .unsqueeze(2) \
        #      .repeat(1, 1, 1, 1, 1) \
        #      .view(-1, 8, 3) 
        #      ),
        #     (batch_data_label['vg_gt_box_corners'] \
        #      .unsqueeze(1) \
        #      .repeat(1, 1, 1, 1, 1) \
        #      .view(-1, 8, 3)
        #      )
        # )

        IoU += match_box_ious.sum().item()
        p_25 += (match_box_ious>0.25).sum().item()
        p_50 += (match_box_ious>0.50).sum().item()

        IoU_gt_noise += match_box_ious_gt_discretized.sum().item()
        p_25_gt_noise += (match_box_ious_gt_discretized>0.25).sum().item()
        p_50_gt_noise += (match_box_ious_gt_discretized>0.50).sum().item()

        # IoU_gt += match_box_ious_gt.sum().item()
        # p_25_gt += (match_box_ious_gt>0.25).sum().item()
        # p_50_gt += (match_box_ious_gt>0.50).sum().item()

        count += batch_size
        end_time = time.time()
        batch_time = round(end_time-begin_time, 1)
        total_time += batch_time

        if is_primary() and curr_iter % 10 == 0:
            logout(
                f"\n----------------------Evaluation-----------------------\n"
                f"Pred: mIoU: {IoU / count}, acc@25: {p_25 / count}, acc@50: {p_50 / count}, time: {total_time}, step: {curr_iter}/{len(dataset_loader)}\n"
                f"GT with discretized noise: mIoU: {IoU_gt_noise / count}, acc@25: {p_25_gt_noise / count}, acc@50: {p_50_gt_noise / count}, time: {total_time}, step: {curr_iter}/{len(dataset_loader)}\n"
                # f"GT : mIoU: {IoU_gt / count}, acc@25: {p_25_gt / count}, acc@50: {p_50_gt / count}, time: {total_time}, step: {curr_iter}/{len(dataset_loader)}"
            )

    eval_metrics['mIoU'] = IoU / count
    eval_metrics['acc@0.25'] = p_25 / count
    eval_metrics['acc@0.50'] = p_50 / count

    if is_primary():
            logout(
                f"\n----------------------Evaluation-----------------------\n"
                f"Result: mIoU: {eval_metrics['mIoU']}, acc@25: {eval_metrics['acc@0.25']}, acc@50: {eval_metrics['acc@0.50']}, time: {total_time}\n"
                f"GT with discretized noise: mIoU: {IoU_gt_noise / count}, acc@25: {p_25_gt_noise / count}, acc@50: {p_50_gt_noise / count}, time: {total_time}, step: {curr_iter}/{len(dataset_loader)}\n"
                # f"GT : mIoU: {IoU_gt / count}, acc@25: {p_25_gt / count}, acc@50: {p_50_gt / count}, time: {total_time}, step: {curr_iter}/{len(dataset_loader)}"
            )

    return eval_metrics