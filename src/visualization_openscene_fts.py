import os
import open3d as o3d
import torch 
import numpy as np
import clip

model_name="ViT-L/14@336px"
clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)

def get_clip_text_fts(text):
    text = clip.tokenize([text])
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

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

scan_name = 'scene0137_00'
scene_token_p = f'/mnt/nfs/share/Adaptive/openscene_scene_tokens_axis_align_w_sm_obj_0.2_r_0.25_10_0.05_500/{scan_name}/enc_features.pt'
scene_token_xyz_p = f'/mnt/nfs/share/Adaptive/openscene_scene_tokens_axis_align_w_sm_obj_0.2_r_0.25_10_0.05_500/{scan_name}/enc_xyz.pt'

# Load scene axis alignment matrix
meta_file = f'/mnt/nfs/share/datasets/scannet/scans/{scan_name}/{scan_name}.txt'
lines = open(meta_file).readlines()
axis_align_matrix = None
for line in lines:
    if 'axisAlignment' in line:
        axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))


scene_pcd_info = _get_scan_data(scan_name)
point_clouds = scene_pcd_info["point_clouds"][:, :3]  
## print bbox of pcd
# print(point_clouds[:, 0].max())
# print(point_clouds[:, 0].min())
# print(point_clouds[:, 1].max())
# print(point_clouds[:, 1].min())
# print(point_clouds[:, 2].max())
# print(point_clouds[:, 2].min())

colors = scene_pcd_info["point_clouds"][:, 3:6]

## [512,771]
scene_token = torch.load(scene_token_p, map_location='cuda')[0]
## [512,3]
scene_token_xyz = torch.load(scene_token_xyz_p, map_location='cpu')[0]

# pts = np.ones((scene_token_xyz.shape[0], 4))
# pts[:,0:3] = scene_token_xyz[:,0:3]
# scene_token_xyz = np.dot(pts, axis_align_matrix.transpose())[:, 0:3] # Nx4

## drop xyz
scene_token = scene_token[..., 3:]

prompt = 'telephone'
top_k = 5
print(prompt)
print(scan_name)

norm_text_fts = get_clip_text_fts(prompt)

## get similarity
scene_tokenv = scene_token / scene_token.norm(dim=-1, keepdim=True)
similarity = (norm_text_fts.float() @ scene_token.T.float()).squeeze()

## get top k results indices 
top_k_indices = torch.topk(similarity, top_k).indices

## get region of instrest
region_inds = [torch.load(f'/mnt/nfs/share/Adaptive/openscene_scene_tokens_axis_align_w_sm_obj_0.2_r_0.25_10_0.05_500/{scan_name}/region_inds_{i}.pt', map_location='cpu')[0] for i in top_k_indices]
region_xyzs = [torch.load(f'/mnt/nfs/share/Adaptive/openscene_scene_tokens_axis_align_w_sm_obj_0.2_r_0.25_10_0.05_500/{scan_name}/region_xyz_{i}.pt', map_location='cpu')[0] for i in top_k_indices]

instrest_xyz = scene_token_xyz[top_k_indices.cpu()]
instrest_region_xyz = []
for inds, xyzs in zip(region_inds, region_xyzs):
    for ind, xyz in zip(inds, xyzs):
        instrest_region_xyz.append(xyz)
instrest_region_xyz = torch.stack(instrest_region_xyz, dim=0)
# pts = np.ones((instrest_region_xyz.shape[0], 4))
# pts[:,0:3] = instrest_region_xyz[:,0:3]
# instrest_region_xyz = np.dot(pts, axis_align_matrix.transpose())[:, 0:3] # Nx4


## create instrest radius ball 
instrest_sphere_list = []
for ixyz in instrest_xyz:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
    sphere.translate(ixyz) 
    # print(ixyz)
    sphere.paint_uniform_color([0, 0, 1])
    instrest_sphere_list.append(sphere)
    
instrest_region_sphere_list = []
for ixyz in instrest_region_xyz:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # 设置球体的半径
    sphere.translate(ixyz) 
    # print(ixyz)
    sphere.paint_uniform_color([0, 1, 0])
    instrest_region_sphere_list.append(sphere)
    
## vis
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.colors = o3d.utility.Vector3dVector(colors)
vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
o3d.visualization.draw_geometries([vis_pcd, *instrest_region_sphere_list])



 