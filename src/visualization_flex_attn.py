import open3d as o3d
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


attn_dir = 'results/attn_vis_flex/test/'
# exp_dir = 'results/toy_exp/nipus_exp/unified_scanqa/finetune_model/encoder-openscene-maskformer-axis-align-w-sm-obj-wocausal/4layer/finetune_flex_self_attn/1epoch/qa_corpus_val.json'

## FIXME
# new_qa_pred_gt = {}
# with open(exp_dir, 'r') as f:
#     qa_pred_gt = json.load(f)
# for k,v in qa_pred_gt.items():
#     question_id = k.split('-')[0] + '-' + k.split('-')[1] + '-' + k.split('-')[2]
#     new_qa_pred_gt[question_id] = v


def _get_scan_data(scan_name,):
    data_path = 'data/scannet/scannet_data_w_sm_obj_dense'
    mesh_vertices = np.load(os.path.join(data_path, scan_name) + "_aligned_vert.npy")
    instance_labels = np.load(
        os.path.join(data_path, scan_name) + "_ins_label.npy"
    )
    ## bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]
    instance_bboxes = np.load(os.path.join(data_path, scan_name) + "_aligned_bbox.npy")


    point_cloud = mesh_vertices[:, 0:6]
    point_cloud[:, 3:] = (point_cloud[:, 3:] - np.array([109.8, 97.2, 83.8])) / 256.0
    pcl_color = point_cloud[:, 3:]
    
    normals = mesh_vertices[:,6:9]
    point_cloud = np.concatenate([point_cloud, normals], 1)
    
    
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
    
    ret_dict = {}
    ret_dict["point_clouds"] = point_cloud.astype(np.float32)
    ret_dict["pcl_color"] = pcl_color
    
    ret_dict['instance_labels'] = instance_labels.astype(np.int32) 
    
    return ret_dict

sphere_colors = [
    [0, 0, 1],   # blue
    [1, 0, 0],   # red
    [0, 1, 0],   # green
    [1, 1, 0],   # yellow
    [1, 0, 1],   # purple
    [0, 1, 1],   # cyan
    [0, 0, 0],   # black
]

## load annotation
anno = list(json.load(open('data/ScanQA/ScanQA_v1.0_val.json','r')))

attn_p_list = os.listdir(attn_dir)
# random.shuffle(attn_p_list)


for episode in tqdm(attn_p_list):
    
    anno_info = anno[int(episode)]
    scene_id = anno_info['scene_id']
    object_ids = anno_info['object_ids']
    object_names = anno_info["object_names"]
    print(anno_info['question'])
    print(anno_info['answers'])
    
    # pred_info = new_qa_pred_gt[anno_info['question_id']]
    # print("Pred: ", pred_info['pred'])
    # pred_seq = pred_info['pred'][0].split(' ')
    
    # score = pred_info['score']
    # if not score['CiDEr'] > 0.7:
    #     continue
    
    ## get scene pcd
    scene_pcd_info = _get_scan_data(anno_info['scene_id'])
    point_clouds = scene_pcd_info["point_clouds"][:, :3]  
    colors = scene_pcd_info["point_clouds"][:, 3:6]
    obj_idx = [scene_pcd_info["instance_labels"] == (target_obj_id + 1) for target_obj_id in object_ids]
    tmp_pcd = o3d.geometry.PointCloud()
    axis_aligned_bounding_box_list = []
    for oid in obj_idx:
        object_points = point_clouds[oid]    # npt x 3
        tmp_pcd.points = o3d.utility.Vector3dVector(object_points)
        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        axis_aligned_bounding_box_list.append(aabb)
    
    instrest_sphere_list = []
    new_tokens = os.listdir(epi_p:=os.path.join(attn_dir, episode))[:-1]
    for token_idx, last_token_attn_p in enumerate(new_tokens):
        # print(pred_seq[token_idx])
        attn_infos = torch.load(os.path.join(epi_p, last_token_attn_p), map_location='cpu')
        
        scan_name = attn_infos['scan_name'][0]
        assert anno_info['scene_id'] == scan_name, f'{scene_id} != {scan_name}'
        
        ## [layers(24), nhead(32), scen_token(512)+text_token(?)]
        attn_weight = attn_infos['attn_weight']
        ## only perserve last k layers
        attn_weight = attn_weight[-8:, ...]
        ## only perserve scene token
        attn_weight = attn_weight[..., :512]
        
        ## [bs(1), scen_token(512), 3]
        xyz = attn_infos['xyz']
        
        for li, layer_x_attn_weight in enumerate(attn_weight):
            ## sum in nhead
            layer_x_attn_weight = layer_x_attn_weight.sum(0)
            ## softmax
            layer_x_attn_weight = torch.nn.functional.softmax(layer_x_attn_weight * 100, dim=-1)
            ## argmax
            argmax_idx = torch.argmax(layer_x_attn_weight, dim=-1)
            
            ## pop corresponding xyz
            instrest_xyz = xyz[0, argmax_idx]
            ## create instrest radius ball 
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
            sphere.translate(instrest_xyz.numpy()) 
            sphere.paint_uniform_color(sphere_colors[token_idx])
            instrest_sphere_list.append(sphere)

    ## vis
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
    o3d.visualization.draw_geometries([vis_pcd, *axis_aligned_bounding_box_list, *instrest_sphere_list])
       
    
