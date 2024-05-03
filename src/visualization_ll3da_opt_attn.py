import open3d as o3d
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

attn_dir = 'results/ll3da_attn_op/official'
from transformers import AutoTokenizer


import numpy as np
import matplotlib.pyplot as plt

def plot_attention_map(attention_weights):
    """
    Plot attention map with activation values.
    
    Args:
    - attention_weights: 2D array containing the attention weights
    """
    # Normalize attention weights
    attention_weights = np.array(attention_weights)
    normalized_weights = attention_weights / np.max(attention_weights)
    
    # Get dimensions of the attention map
    rows, cols = attention_weights.shape
    
    # Plot attention map
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_weights, cmap='Reds', interpolation='nearest', aspect='auto')
    
    # Add activation values to each cell
    # for i in range(rows):
    #     for j in range(cols):
    #         plt.text(j, i, "{:.2f}".format(attention_weights[i, j]),
    #                  horizontalalignment='center', verticalalignment='center',
    #                  color='black', fontsize=10)
    
    plt.xlabel("learnable query")
    plt.ylabel("text token")
    # plt.title("Attention Map with Activation Values")
    plt.colorbar()
    plt.show()




tokenizer = AutoTokenizer.from_pretrained('ckpts/opt-model')

def _get_scan_data(scan_name,):
    data_path = 'data/scannet/scannet_data_dense'
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


attn_p_list = os.listdir(attn_dir)
attn_p_list.sort(key = lambda x: int(x))
from collections import Counter
scene_counter = Counter()

# random.shuffle(attn_p_list)

for qformer_x_attns_p in tqdm(attn_p_list[::5]):
    qformer_attn = torch.load(os.path.join(attn_dir, qformer_x_attns_p, 'qformer_attn.pt'), map_location='cpu')
    opt_attn = torch.load(os.path.join(attn_dir, qformer_x_attns_p, 'opt_attn.pt'), map_location='cpu')
    
    anno_info = opt_attn['anno']
    scene_id = anno_info['scene_id']
    
    scene_counter[scene_id] += 1
    if scene_counter[scene_id] > 3:
        continue
    
    object_ids = anno_info['object_ids']
    object_names = anno_info["object_names"]
    
    print(anno_info['question'])
    print(anno_info['answers'])
    print("pred", opt_attn['pred_answer'])
    print(scene_id)
    
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
    
    
    
    opt_attn_map = opt_attn['attn']
    
    ## mean in head
    opt_attn_map = opt_attn_map.mean(1)
    
    ## filter irrelevant layer
    opt_attn_map = opt_attn_map[..., :32]
    
    ## ### human: given the 3D scene, answer the question: " : 19
    ## " ### assistant:' : 11
    total_seq_len = opt_attn_map.shape[-1]
    output_len = len(tokenizer(opt_attn['pred_answer'])['input_ids'])
    
    answer_token_attn = opt_attn_map[:, -(len(opt_attn['output_ids'])-1): -(len(opt_attn['output_ids'])-output_len + 1) , :]
    prefix_len = len(tokenizer('### human: given the 3D scene, answer the question: "')['input_ids'])-1
    qs_token_len = len(tokenizer(anno_info['question'])['input_ids'])-1
    question_token_attn = opt_attn_map[:, 32+prefix_len:prefix_len+qs_token_len+32  :]
    
    opt_attn_map = torch.cat([question_token_attn, answer_token_attn], dim=1)
    
    
    # ## [layers(3), bs(1), nhead(12), num_learnable_query(32), scen_token(1024)]
    x_attn_weight = qformer_attn['x_attn_weight']
    ## [bs(1), scen_token(1024), 3]
    xyz = qformer_attn['xyz'].numpy()
    
    # average the layer attention weight
    avg_x_attn_weight = qformer_attn['x_attn_weight'].mean(0)[0]
    # average the head attention weight
    avg_x_attn_weight = avg_x_attn_weight.mean(0)


    # 定义从蓝色到黄色的颜色映射
    map_colors = ["blue", "yellow", "red"]  # 蓝色到黄色
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("mycmap", map_colors)
    full_activations = np.zeros(point_clouds.shape[0])
    min_activate = 1e9
    # plot_attention_map(avg_x_attn_weight)
    for query_x_attn_weight in avg_x_attn_weight:
        ## 选出topk的激活值
        
        _, argmax_idx = query_x_attn_weight.topk(k=10)
        instrest_xyz = xyz[0, argmax_idx]
        # print(instrest_xyz)
        # print(argmax_idx)
        
        ## 由于没有xyz在原始点云中的ind，所以concat在原始点云后面
        point_clouds = np.concatenate([point_clouds, instrest_xyz], axis=0)
        colors = np.concatenate([colors, np.full((instrest_xyz.shape[0], 3), [0.5, 0.5, 0.5])], axis=0)
        
        # full_activations = np.full(point_clouds.shape[0], query_x_attn_weight.min()*1000)
        full_activations = np.concatenate((full_activations, query_x_attn_weight[argmax_idx]*0))
        if query_x_attn_weight[argmax_idx].min() < min_activate:
            min_activate = query_x_attn_weight[argmax_idx].min()
        
        tree = BallTree(point_clouds)
        # 将每个激活点的激活值扩展到0.2米范围内的所有点
        radius = 0.2  # 0.2米
        inds = []
        activations = []
        for index, activation in zip(list(range(point_clouds.shape[0]-instrest_xyz.shape[0], point_clouds.shape[0])), query_x_attn_weight[argmax_idx]):
            # 找到在0.2米范围内的点
            ind = tree.query_radius(point_clouds[index:index+1], r=radius)[0]
            inds.append(ind)
            activations.append(activation.item())
        
        normalized_activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations))
        for ind, act in zip(inds, normalized_activations):
            full_activations[ind] = act  # 设置相同的激活值
        
    # full_activations[full_activations == 0] = min_activate
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
    # normalized_activations = (smoothed_activations - np.min(smoothed_activations)) / (np.max(smoothed_activations) - np.min(smoothed_activations))
    activation_colors = cmap(smoothed_activations)
    
    mixed_colors = 0.6 * colors + 0.4 * activation_colors[:,:3]
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_clouds)
    # pcd.colors = o3d.utility.Vector3dVector(mixed_colors)
    # o3d.visualization.draw_geometries([pcd, *axis_aligned_bounding_box_list])
    
    
    for visual_layer in range(8, 24):
        visual_layer = 20
        ks = [1, 2]
        plot_attention_map(opt_attn_map[visual_layer])
        for k in ks:
            layer_opt_attn_map = opt_attn_map[visual_layer]
            
            top_values, top_indices = torch.topk(layer_opt_attn_map.float(), k, dim=1, largest=True)
            print(top_values)
            print(top_indices)
            # top_indices = top_indices[:, k-1:]
            top_indices = np.unique(top_indices.numpy())
            print(top_indices)
            ## drop bias
            
        
            ## [layers(3), bs(1), nhead(12), num_learnable_query(32), scen_token(1024)]
            x_attn_weight = qformer_attn['x_attn_weight']
            ## [bs(1), scen_token(1024), 3]
            xyz = qformer_attn['xyz'].numpy()
            
            ## new visualization version
            # average the layer attention weight
            avg_x_attn_weight = qformer_attn['x_attn_weight'].mean(0)[0]
            # average the head attention weight
            avg_x_attn_weight = avg_x_attn_weight.mean(0)

        
            # 定义从蓝色到黄色的颜色映射
            map_colors = ["blue", "yellow", "red"]  # 蓝色到黄色
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("mycmap", map_colors)
            full_activations = np.zeros(point_clouds.shape[0])
            min_activate = 1e9
            for query_x_attn_weight in avg_x_attn_weight[top_indices]:
                ## 选出topk的激活值
                
                _, argmax_idx = query_x_attn_weight.topk(k=10)
                instrest_xyz = xyz[0, argmax_idx]
                print(instrest_xyz)
                
                ## 由于没有xyz在原始点云中的ind，所以concat在原始点云后面
                point_clouds = np.concatenate([point_clouds, instrest_xyz], axis=0)
                colors = np.concatenate([colors, np.full((instrest_xyz.shape[0], 3), [0.5, 0.5, 0.5])], axis=0)
                
                # full_activations = np.full(point_clouds.shape[0], query_x_attn_weight.min()*1000)
                full_activations = np.concatenate((full_activations, query_x_attn_weight[argmax_idx]*0))
                if query_x_attn_weight[argmax_idx].min() < min_activate:
                    min_activate = query_x_attn_weight[argmax_idx].min()
                
                tree = BallTree(point_clouds)
                # 将每个激活点的激活值扩展到0.2米范围内的所有点
                radius = 0.2  # 0.2米
                inds = []
                activations = []
                for index, activation in zip(list(range(point_clouds.shape[0]-instrest_xyz.shape[0], point_clouds.shape[0])), query_x_attn_weight[argmax_idx]):
                    # 找到在0.2米范围内的点
                    ind = tree.query_radius(point_clouds[index:index+1], r=radius)[0]
                    inds.append(ind)
                    activations.append(activation.item())
                
                normalized_activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations))
                for ind, act in zip(inds, normalized_activations):
                    full_activations[ind] = act  # 设置相同的激活值
                    
                    
                
            # full_activations[full_activations == 0] = min_activate
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
            # normalized_activations = (smoothed_activations - np.min(smoothed_activations)) / (np.max(smoothed_activations) - np.min(smoothed_activations))
            activation_colors = cmap(smoothed_activations)
            
            mixed_colors = 0.6 * colors + 0.4 * activation_colors[:,:3]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_clouds)
            pcd.colors = o3d.utility.Vector3dVector(mixed_colors)
            o3d.visualization.draw_geometries([pcd, *axis_aligned_bounding_box_list])

        break
    ## old visualization version
    # for li, layer_x_attn_weight in enumerate(x_attn_weight):
    #     ## remove batch
    #     layer_x_attn_weight = layer_x_attn_weight[0]
    #     ## sum in nhead
    #     layer_x_attn_weight = layer_x_attn_weight.sum(0)
    #     ## softmax
    #     layer_x_attn_weight = torch.nn.functional.softmax(layer_x_attn_weight * 100, dim=-1)
    #     ## argmax
    #     argmax_idx = torch.argmax(layer_x_attn_weight, dim=-1)
    #     assert len(argmax_idx) == x_attn_weight.shape[-2], f'{len(argmax_idx)} != {x_attn_weight.shape[-2]}'
        
    #     ## pop corresponding xyz
    #     instrest_xyz = xyz[0, argmax_idx]
    #     assert len(instrest_xyz) == x_attn_weight.shape[-2], f'{len(instrest_xyz)} != {x_attn_weight.shape[-2]}'
        
    #     ## create instrest radius ball 
    #     instrest_sphere_list = []
    #     for ixyz in instrest_xyz:
    #         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  # 设置球体的半径
    #         sphere.translate(ixyz) 
    #         sphere.paint_uniform_color([0, 0, 1])
    #         instrest_sphere_list.append(sphere)
            
    #     ## vis
    #     vis_pcd = o3d.geometry.PointCloud()
    #     vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    #     vis_pcd.points = o3d.utility.Vector3dVector(point_clouds)
    #     o3d.visualization.draw_geometries([vis_pcd, *axis_aligned_bounding_box_list, *instrest_sphere_list])

    