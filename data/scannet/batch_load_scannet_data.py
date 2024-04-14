""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py

Batch mode in loading Scannet scenes with vertices and ground truth labels for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""

import os
import sys
import datetime
import numpy as np
from load_scannet_data import export
import pdb

SCANNET_DIR = '/mnt/nfs/share/datasets/scannet/scans'   # TODO: change this
# SCANNET_DIR = '.../scans_test'   # HACK: If you wish to upload your results, remember to process the test set
SCAN_NAMES = os.listdir(SCANNET_DIR)
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_data_w_sm_obj_dense'

def export_one_scan(scan_name, output_filename_prefix):    
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
    mesh_vertices, aligned_vertices, semantic_labels, instance_labels, instance_bboxes, aligned_instance_bboxes = export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    aligned_vertices = aligned_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]
    
            
    ## add small obj pcd
    import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # print(aligned_vertices.shape)
    # pcd.points = o3d.utility.Vector3dVector(aligned_vertices[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(aligned_vertices[:, 3:6]/255)
    # # o3d.visualization.draw_geometries([pcd])
    small_obj_pcd_dir = f'/home/admin/Projects/EmbodiedScan/data/small_size_object/pcd/{scan_name}'
    if os.path.exists(small_obj_pcd_dir) and len(os.listdir(small_obj_pcd_dir)) > 0:
        small_obj_pcd = []
        for ply in os.listdir(small_obj_pcd_dir):
            ply_p = os.path.join(small_obj_pcd_dir, ply)
            point_cloud = o3d.io.read_point_cloud(ply_p)
            point = point_cloud.points
            color = point_cloud.colors
            normal = point_cloud.normals
            small_obj_pcd.extend(np.concatenate([point, color, normal], axis=1))
        # print(np.array(small_obj_pcd).shape)
        small_obj_pcd = np.array(small_obj_pcd)
        # print(small_obj_pcd.shape)
        small_obj_pcd[:, 3:6] = small_obj_pcd[:, 3:6]*255
        # spcd = o3d.geometry.PointCloud()
        # spcd.points = o3d.utility.Vector3dVector(small_obj_pcd[:, :3])
        # spcd.colors = o3d.utility.Vector3dVector(small_obj_pcd[:, 3:6]/255)
        
        # o3d.visualization.draw_geometries([spcd, pcd])
        # assert False
        aligned_vertices = np.concatenate([aligned_vertices, small_obj_pcd], axis=0)
    
    if instance_bboxes.shape[0] > 1:
        num_instances = len(np.unique(instance_labels))
        print('Num of instances: ', num_instances)

        # bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
        bbox_mask = np.in1d(instance_bboxes[:,-2], OBJ_CLASS_IDS) # match the mesh2cap
        instance_bboxes = instance_bboxes[bbox_mask,:]
        aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask,:]
        print('Num of care instances: ', instance_bboxes.shape[0])
    else:
        print("No semantic/instance annotation for test scenes")

    # N = mesh_vertices.shape[0]
    # if N > MAX_NUM_POINT:
    #     choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
    #     mesh_vertices = mesh_vertices[choices, :]
    #     aligned_vertices = aligned_vertices[choices, :]
    #     semantic_labels = semantic_labels[choices]
    #     instance_labels = instance_labels[choices]

    print("Shape of points: {}".format(mesh_vertices.shape))

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_aligned_vert.npy', aligned_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)
    np.save(output_filename_prefix+'_aligned_bbox.npy', aligned_instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    from tqdm import tqdm
    for scan_name in tqdm(SCAN_NAMES):
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        if os.path.exists(output_filename_prefix + '_vert.npy'): continue
        
        # print('-'*20+'begin')
        # print(datetime.datetime.now())
        # print(scan_name)
              
        export_one_scan(scan_name, output_filename_prefix)
             
        # print('-'*20+'done')

if __name__=='__main__':    
    batch_export()
