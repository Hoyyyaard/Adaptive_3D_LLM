import os
import json

result_file_p = 'results/toy_exp/scanqa/official/qa_pred_gt_val.json'
size_dataset_p = 'results/size_filter_scannet_qa_datasets/val_size_s_1e-2_num180.json'
result = json.load(open(result_file_p, 'r'))

filter_data = json.load(open(size_dataset_p, 'r'))
filter_key = [d['question_id'] for d in filter_data]

filter_sta = {
    "bleu-1": [],
    "bleu-2": [],
    "bleu-3": [],
    "bleu-4": [],
    "CiDEr": [],
    "rouge": [],
    "meteor": []
}
out_of_filter_sta = {
    "bleu-1": [],
    "bleu-2": [],
    "bleu-3": [],
    "bleu-4": [],
    "CiDEr": [],
    "rouge": [],
    "meteor": []
}

for k,v in result.items():
    scene_id = k.split("-")[0:3]
    scene_id = "-".join(scene_id)
    score = v['score']
    if scene_id in filter_key:
        for sk in filter_sta.keys():
            filter_sta[sk].append(score[sk])
    else:
        for sk in out_of_filter_sta.keys():
            out_of_filter_sta[sk].append(score[sk])

print("--------------filter sta--------------")
for k,v in filter_sta.items():
    print(k, sum(v)/len(v))
print("--------------out_of_filter sta--------------")
for k,v in out_of_filter_sta.items():
    print(k, sum(v)/len(v))
            

    
    