import os
import json
from tqdm import tqdm


with open('data/ScanRefer/ScanRefer_filtered_val.json', 'r') as f:
    scanrefer = json.load(f)

qa_format_dict = {}
global_annotation_id = 0

for anno in tqdm(scanrefer):
    key = '-'.join((anno['scene_id'], anno['object_id'], anno['object_name']))
    if not key in qa_format_dict.keys():
        qa_format_dict[key] = {
            'scene_id': anno['scene_id'],
            'object_id': anno['object_id'],
            "object_name": anno['object_name'],
            'global_ann_id': str(global_annotation_id),
            'answers': [anno['description']]
        }
        global_annotation_id += 1
    else:
        qa_format_dict[key]['answers'].append(anno['description'])
        
with open('data/ScanRefer/ScanRefer_filtered_val_qa_format.json', 'w') as f:
    json.dump(list(qa_format_dict.values()), f)
        