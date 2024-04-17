import os 
import json
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm

def _get_scan_pcd(scan_name, w_sm_obj):
    scan_name = scan_id.split('/')[1]
    source_dir = '/home/admin/Projects/LL3DA/data/scannet/scannet_data'
    source_path = os.path.join(f'{source_dir}_dense', scan_name) if not w_sm_obj else os.path.join(f'{source_dir}_w_sm_obj_dense', scan_name)
    mesh_vertices = np.load(source_path + "_aligned_vert.npy")
    instance_labels = np.load(
        source_path + "_ins_label.npy"
    )
    semantic_labels = np.load(
        source_path + "_sem_label.npy"
    )
    ## bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]
    instance_bboxes = np.load(source_path + "_aligned_bbox.npy")

    point_cloud = mesh_vertices[:, 0:6]
    MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
    point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
    pcl_color = point_cloud[:, 3:]
    normals = mesh_vertices[:,6:9]
    point_cloud = np.concatenate([point_cloud, normals], 1)
        
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
    
    return point_cloud

def _9dof_to_box(box):
    if isinstance(box, list):
        box = np.array(box)
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
        rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
    geo.color = [1,0,0]
    return geo

def point_inside_obb(point, obb):
    # 获取定向边界框的中心点、尺寸和旋转矩阵
    center = np.array(obb.center)
    extent = np.array(obb.extent)
    R = np.array(obb.R)

    # 将点变换到局部坐标系
    point_local = np.dot((point - center), np.linalg.inv(R))

    # 检查点是否在局部坐标系内的边界框内
    for i in range(3):
        if abs(point_local[i]) > extent[i]:
            return False
    return True

annotation = json.load(open("/home/admin/Projects/EmbodiedScan/data/small_size_object/train_small_than_1e-3_wdes_subset.json","r"))


for anno in tqdm(annotation):

    scan_id = anno['scan_id']
    tgt_bbox = anno['tgt_bbox']
    tgt_obj_name = anno['target']
    
    print(tgt_obj_name)
    
    # if tgt_obj_name == 'mouse':
    #     continue
    
    bbox = _9dof_to_box(tgt_bbox)
    pcd_w_sm_obj = _get_scan_pcd(scan_id, True)
    pcd = _get_scan_pcd(scan_id, False)
        
    vpcd = o3d.geometry.PointCloud()
    vpcd.points = o3d.utility.Vector3dVector(pcd[:,:3])
    vpcd.colors = o3d.utility.Vector3dVector(pcd[:,3:6])
    o3d.visualization.draw_geometries([vpcd, bbox])
    
    vpcd.points = o3d.utility.Vector3dVector(pcd_w_sm_obj[:,:3])
    vpcd.colors = o3d.utility.Vector3dVector(pcd_w_sm_obj[:,3:6])
    o3d.visualization.draw_geometries([vpcd, bbox])

    
