import os
import sys
from transformers import AutoConfig
from modeling_opt_flex import Shell_Model
from openscene_dense_pcd_fts_cache import OpenScene_Fts_Cache
import torch
import open3d as o3d
import numpy as np


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



num_finetune_hidden_layers = 0

config = AutoConfig.from_pretrained('src/flex_opt_config.json')
## 这里代码有bug 由于代码上的bug 第一层的flex self attn相当于self attn
config.num_finetune_hidden_layers = num_finetune_hidden_layers + 1
config.num_hidden_layers = 24 - num_finetune_hidden_layers - 1
print("acc_num_flex_hidden_layers: ", config.num_finetune_hidden_layers)
print("acc_num_hidden_layers: ", config.num_hidden_layers)
config.num_hidden_layers = config.num_finetune_hidden_layers + config.num_hidden_layers
config.num_finetune_hidden_layers = 0

model = Shell_Model(config=config)
model = model.cuda()
cache = OpenScene_Fts_Cache()

scene_list = os.listdir('/mnt/nfs/share/datasets/scannet/scans')

prompt = 'chair'
ckpt_path = 'ckpts/opt-1.3b/nipus_exp/encoder-openscene-maskformer-axis-align-w-sm-obj-wocausal/checkpoint_32k.pth'


msg = model.model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'], strict=False)
print(msg)

for scan_name in scene_list:

    scene_pcd_info = _get_scan_data(scan_name)
    point_clouds = scene_pcd_info["point_clouds"][:, :3]  
    colors = scene_pcd_info["point_clouds"][:, 3:6]


    openscene_info = cache.get_openscene_scan_datas(scan_name, preprocess=False)
    scene_tokens = openscene_info['scene_tokens']
    scene_xyz = openscene_info['scene_xyz']
    dense_region_tokens = openscene_info['dense_region_tokens']
    dense_region_xyz = openscene_info['dense_region_xyz']

    with torch.no_grad():
        scene_tokens = torch.from_numpy(scene_tokens).cuda().unsqueeze(0)
        scene_xyz = torch.from_numpy(scene_xyz).cuda().unsqueeze(0)
        dense_region_tokens = torch.from_numpy(dense_region_tokens).cuda().unsqueeze(0)
        dense_region_xyz = torch.from_numpy(dense_region_xyz).cuda().unsqueeze(0)

        ## extract scene tokens fts 
        scene_tokens_fts = model.model._run_mask_tranformer_encoder(scene_tokens, scene_xyz)
        
        ## extraxt text fts
        text_token = model.model.tokenizer(prompt, return_tensors="pt")
        embedding_layer = model.model.get_input_embeddings()
        text_fts = embedding_layer(text_token['input_ids'][: , 1].unsqueeze(0).cuda())

        ## calculate softmax similarity
        scene_tokens_fts = scene_tokens_fts / scene_tokens_fts.norm(dim=-1, keepdim=True)
        text_fts = text_fts / text_fts.norm(dim=-1, keepdim=True)
        similarity = text_fts.squeeze(0) @ scene_tokens_fts.squeeze(0).transpose(0, 1)
        
        ## softmax
        similarity = torch.nn.functional.softmax(similarity * 100, dim=-1)
        
        ## get topk
        similarity = similarity.cpu().numpy()
        
        ## get topk
        topk = 10
        topk_idx = np.array(similarity.argsort()[0][::-1][:topk])

        ## get topk scene xyz
        topk_scene_xyz = scene_xyz.cpu().numpy()[0, topk_idx, :]

        ## create instrest radius ball 
        instrest_sphere_list = []
        for ixyz in topk_scene_xyz:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
            sphere.translate(ixyz) 
            sphere.paint_uniform_color([0, 0, 1])
            instrest_sphere_list.append(sphere)
            
    ## vis
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
    o3d.visualization.draw_geometries([vis_pcd, *instrest_sphere_list])
