import numpy as np
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import ball_query, grouping_operation
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
import torch
import os


class OpenScene_Fts_Cache():
    def __init__(self, cache_dir='/mnt/nfs/share/Adaptive/openscene_dense_fts_distill_axis_align_w_sm_obj'):
        self.cache_dir = cache_dir
        self.dense_fts_cache = {}
        self.npoint = 40000
        self.pad_npoint = 200000
    
    def _get_dense_fts(self, scan_name):
        return np.load(os.path.join(self.cache_dir, scan_name + '_dense_fts.npy'))
    
    def _get_dense_pcd(self, scan_name):
        return torch.load(os.path.join(self.cache_dir, scan_name + '_xyz.pt'), map_location='cpu').numpy()
    
    def _get_reverse_inds(self, scan_name):
        return torch.load(os.path.join(self.cache_dir, scan_name + '_inds_reverse.pt'), map_location='cpu').numpy()
    
    def _get_gt_dense_instance_labels(self, scan_name):
        return np.load(os.path.join('data/scannet/scannet_data_w_sm_obj_dense', scan_name + '_ins_label.npy'))
    
    def get_openscene_scan_datas(self, scan_name, preprocess):
        if preprocess:
            dense_pcd = self._get_dense_pcd(scan_name)
            dense_fts = self._get_dense_fts(scan_name)
            reverse_inds = self._get_reverse_inds(scan_name)
            instance_labels = self._get_gt_dense_instance_labels(scan_name)
            
            inds = np.random.choice(len(dense_fts), self.npoint, replace=True)
            
            openscene_sparse_fts = dense_fts[inds, :]
            openscene_sparse_pcd = dense_pcd[inds, :]
            instance_labels = instance_labels[inds]
            
            valid_pcd_len = 0
            # if len(dense_pcd) <= self.pad_npoint:
            #     pad_pcd_array = np.ones((self.pad_npoint - len(dense_pcd), dense_pcd.shape[1]), dtype=dense_fts.dtype)*(1e3)
            #     pad_fts_array = np.zeros((self.pad_npoint - len(dense_fts), dense_fts.shape[1]), dtype=dense_fts.dtype)
                
            #     valid_pcd_len = len(dense_pcd)
            #     dense_pcd = np.concatenate([dense_pcd, pad_pcd_array], axis=0)
            #     dense_fts = np.concatenate([dense_fts, pad_fts_array], axis=0)
            # else:
            #     valid_pcd_len = self.pad_npoint
            #     save_inds = np.random.choice(len(dense_pcd), self.pad_npoint, replace=False)
            #     dense_pcd = dense_pcd[save_inds, :]
            #     dense_fts = dense_fts[save_inds, :]
            
            output_dict = {
                'openscene_dense_fts': dense_fts.astype(np.float32),
                'openscene_dense_pcd': dense_pcd.astype(np.float32),
                'openscene_sample_inds': inds,
                'openscene_sparse_fts': openscene_sparse_fts.astype(np.float32),
                'openscene_sparse_pcd': openscene_sparse_pcd.astype(np.float32),
                'valid_pcd_len' : valid_pcd_len ,
                'instance_labels': instance_labels.astype(np.int64),
            }
            
            return output_dict
        else:
            cache_dir = f'/mnt/nfs/share/Adaptive/openscene_scene_tokens_axis_align_w_sm_obj_0.2_r_0.25_10_0.05_500/{scan_name}'
            scene_tokens = torch.load(f'{cache_dir}/enc_features.pt', map_location='cpu').numpy().astype(np.float32),
            scene_tokens = scene_tokens[0][0]
            scene_xyz = torch.load(f'{cache_dir}/enc_xyz.pt', map_location='cpu').numpy().astype(np.float32)
            scene_xyz = scene_xyz[0]
            dense_region_tokens = [torch.load(f'{cache_dir}/region_features_{i}.pt', map_location='cpu') for i in range(scene_tokens.shape[0])]
            dense_region_tokens = torch.cat(dense_region_tokens, dim=0).numpy().astype(np.float32)
            dense_region_xyz = [torch.load(f'{cache_dir}/region_xyz_{i}.pt', map_location='cpu') for i in range(scene_tokens.shape[0])]
            dense_region_xyz = torch.cat(dense_region_xyz, dim=0).numpy().astype(np.float32)
            return {
                'scene_tokens': scene_tokens,
                'scene_xyz': scene_xyz,
                'dense_region_tokens': dense_region_tokens,
                'dense_region_xyz': dense_region_xyz
            }
    