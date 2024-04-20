import open3d as o3d
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


attn_dir = 'results/attn_vis_flex/encoder-openscene-maskformer-axis-align-w-sm-obj-wocausal-finetune-opt-1-3b/4epoch/qa'
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
    last_tokens = os.listdir(epi_p:=os.path.join(attn_dir, episode))[-2]
    # print(pred_seq[token_idx])
    attn_infos = torch.load(os.path.join(epi_p, last_tokens), map_location='cpu')
    
    scan_name = attn_infos['scan_name'][0]
    assert anno_info['scene_id'] == scan_name, f'{scene_id} != {scan_name}'
    
    ## [layers(24), nhead(32), scen_token(512)+text_token(?), scen_token(512)+text_token(?)]
    attn_weight = attn_infos['attn_weight']
    xyz = attn_infos['xyz']
    ## only perserve last k layers
    attn_weight = attn_weight[-16:, ...]
    ## only perserve scene token
    attn_weight = attn_weight[..., :512]
    ## only perserve text token as query
    attn_weight = attn_weight[:, :, 512:, :]
    ## mean in nhead
    ## [layers(16), text_token(?), scen_token(512)]
    avg_attn_weight = attn_weight.mean(1)
    ## mean in layer
    ## [text_token(?), scen_token(512)]
    avg_x_attn_weight = avg_attn_weight.mean(0)
    
    # 定义从蓝色到黄色的颜色映射
    map_colors = ["blue", "yellow", "red"]  # 蓝色到黄色
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("mycmap", map_colors)
    full_activations = np.zeros(point_clouds.shape[0])
    min_activate = 1e9
    for query_x_attn_weight in avg_x_attn_weight:
        ## 选出topk的激活值
        
        _, argmax_idx = query_x_attn_weight.float().topk(k=10)
        instrest_xyz = xyz[0, argmax_idx]
        
        ## 由于没有xyz在原始点云中的ind，所以concat在原始点云后面
        point_clouds = np.concatenate([point_clouds, instrest_xyz], axis=0)
        colors = np.concatenate([colors, np.full((instrest_xyz.shape[0], 3), [0.5, 0.5, 0.5])], axis=0)
        
        # full_activations = np.full(point_clouds.shape[0], query_x_attn_weight.min()*1000)
        full_activations = np.concatenate((full_activations, query_x_attn_weight[argmax_idx]*1000))
        if query_x_attn_weight[argmax_idx].min() < min_activate:
            min_activate = query_x_attn_weight[argmax_idx].min()
            
        from sklearn.neighbors import BallTree
        tree = BallTree(point_clouds)
        # 将每个激活点的激活值扩展到0.2米范围内的所有点
        radius = 0.2  # 0.2米
        for index, activation in zip(list(range(point_clouds.shape[0]-instrest_xyz.shape[0], point_clouds.shape[0])), query_x_attn_weight[argmax_idx]):
            # 找到在0.2米范围内的点
            ind = tree.query_radius(point_clouds[index:index+1], r=radius)[0]
            full_activations[ind] = activation.item()*1000  # 设置相同的激活值
        
    full_activations[full_activations == 0] = min_activate
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=0.3, algorithm='ball_tree').fit(point_clouds)
    distances, indices = nn.radius_neighbors(point_clouds)
    def gaussian_kernel(distance, sigma=0.05):
        return np.exp(-0.5 * (distance ** 2) / sigma ** 2)
    # 平滑激活值
    smoothed_activations = np.zeros(full_activations.shape[0])
    for i in range(full_activations.shape[0]):
        weights = gaussian_kernel(distances[i])
        weighted_activations = weights * full_activations[indices[i]]
        smoothed_activations[i] = np.sum(weighted_activations) / np.sum(weights)
        
    # 归一化激活值并映射到颜色
    normalized_activations = (smoothed_activations - np.min(smoothed_activations)) / (np.max(smoothed_activations) - np.min(smoothed_activations))
    activation_colors = cmap(normalized_activations)
    
    mixed_colors = 0.5 * colors + 0.5 * activation_colors[:,:3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(mixed_colors)
    o3d.visualization.draw_geometries([pcd, *axis_aligned_bounding_box_list])
    
        
        
        ## old visualization version
        # ## [layers(24), nhead(32), scen_token(512)+text_token(?)]
        # attn_weight = attn_infos['attn_weight']
        # ## only perserve last k layers
        # attn_weight = attn_weight[-16:, ...]
        # ## only perserve scene token
        # attn_weight = attn_weight[..., :512]
        
        # ## [bs(1), scen_token(512), 3]
        # xyz = attn_infos['xyz']
        
        # for li, layer_x_attn_weight in enumerate(attn_weight):
        #     ## sum in nhead
        #     layer_x_attn_weight = layer_x_attn_weight.sum(0)
        #     ## softmax
        #     layer_x_attn_weight = torch.nn.functional.softmax(layer_x_attn_weight.float() * 100, dim=-1)
        #     ## argmax
        #     argmax_idx = torch.argmax(layer_x_attn_weight, dim=-1)
            
        #     ## pop corresponding xyz
        #     instrest_xyz = xyz[0, argmax_idx]
        #     ## create instrest radius ball 
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
        #     sphere.translate(instrest_xyz.numpy()) 
        #     sphere.paint_uniform_color(sphere_colors[token_idx])
        #     instrest_sphere_list.append(sphere)

    ## vis
    # vis_pcd = o3d.geometry.PointCloud()
    # vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    # vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
    # o3d.visualization.draw_geometries([vis_pcd, *axis_aligned_bounding_box_list, *instrest_sphere_list])
       
    
