import open3d as o3d
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


attn_dir = 'results/attn_vis/qa'
exp_dir = 'results/toy_exp/scanqa/official/qa_pred_gt_val.json'


new_qa_pred_gt = {}
with open(exp_dir, 'r') as f:
    qa_pred_gt = json.load(f)
for k,v in qa_pred_gt.items():
    question_id = k.split('-')[0] + '-' + k.split('-')[1] + '-' + k.split('-')[2]
    new_qa_pred_gt[question_id] = v


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


## load annotation
anno = list(json.load(open('data/ScanQA/ScanQA_v1.0_val.json','r')))

attn_p_list = os.listdir(attn_dir)
random.shuffle(attn_p_list)

for qformer_x_attns_p in tqdm(attn_p_list):
    attn_infos = torch.load(os.path.join(attn_dir, qformer_x_attns_p), map_location='cpu')
    anno_info = anno[int(qformer_x_attns_p.split('.')[0])]
    scene_id = anno_info['scene_id']
    scan_name = attn_infos['scan_name'][0]
    assert anno_info['scene_id'] == scan_name, f'{scene_id} != {scan_name}'
    
    # question_answer = anno_info['question'].replace(' ', '_') + '-' + anno_info['answers'][0].replace(' ', '_')
    object_ids = anno_info['object_ids']
    object_names = anno_info["object_names"]
    
    ## get current qa score from exp dir
    pred_info = new_qa_pred_gt[anno_info['question_id']]
    uid = anno_info['question_id'] + '-' + anno_info['question'].replace(' ', '_') + '-' + anno_info['answers'][0].replace(' ', '_')
    score = pred_info['score']
    if not score['CiDEr'] < 0.1:
        continue
    
    print(anno_info['question'])
    print(anno_info['answers'])
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
        # cube = o3d.geometry.TriangleMesh.create_box(width=aabb.get_extent()[0],
        #                                     height=aabb.get_extent()[1],
        #                                     depth=aabb.get_extent()[2])
        # cube.translate(aabb.get_min_bound())  # 将立方体移动到边界盒的最小边界位置
        # cube.paint_uniform_color([0, 1, 0])  
        axis_aligned_bounding_box_list.append(aabb)
    
    ## [layers(3), bs(1), nhead(12), num_learnable_query(32), scen_token(1024)]
    x_attn_weight = attn_infos['x_attn_weight']
    ## [bs(1), scen_token(1024), 3]
    xyz = attn_infos['xyz']
    
    for li, layer_x_attn_weight in enumerate(x_attn_weight):
        ## remove batch
        layer_x_attn_weight = layer_x_attn_weight[0]
        ## sum in nhead
        layer_x_attn_weight = layer_x_attn_weight.sum(0)
        ## softmax
        layer_x_attn_weight = torch.nn.functional.softmax(layer_x_attn_weight * 100, dim=-1)
        ## argmax
        argmax_idx = torch.argmax(layer_x_attn_weight, dim=-1)
        assert len(argmax_idx) == x_attn_weight.shape[-2], f'{len(argmax_idx)} != {x_attn_weight.shape[-2]}'
        
        ## pop corresponding xyz
        instrest_xyz = xyz[0, argmax_idx]
        assert len(instrest_xyz) == x_attn_weight.shape[-2], f'{len(instrest_xyz)} != {x_attn_weight.shape[-2]}'
        
        ## create instrest radius ball 
        instrest_sphere_list = []
        for ixyz in instrest_xyz:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
            sphere.translate(ixyz) 
            sphere.paint_uniform_color([0, 0, 1])
            instrest_sphere_list.append(sphere)
            
        ## vis
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
        o3d.visualization.draw_geometries([vis_pcd, *axis_aligned_bounding_box_list, *instrest_sphere_list])

        # img_width, img_height = (1920, 1080)

        # mat = o3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'

        # renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        # renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
        # renderer_pc.scene.add_geometry("pcd", vis_pcd, mat)

        # # # Optionally set the camera field of view (to zoom in a bit)
        # vertical_field_of_view = 90  # between 5 and 90 degrees
        # aspect_ratio = img_width / img_height  # azimuth over elevation
        # near_plane = 0.1
        # far_plane = 50.0
        # fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        # renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        # center = [0, 0, 0]  # look_at target
        # eye = [0, 0, 2]  # camera position
        # up = [0, 1, 0]  # camera orientation
        # renderer_pc.scene.camera.look_at(center, eye, up)

        # # 捕获图像
        # image = renderer_pc.render_to_image()
        # o3d.io.write_image('rgb.png', image)