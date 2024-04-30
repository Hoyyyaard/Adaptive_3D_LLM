import os, sys, time, math, json, importlib
import torch
import datetime
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.ap_calculator import APCalculator
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    init_distributed, 
    is_distributed, 
    is_primary, 
    get_rank,
    barrier,
    all_reduce_average,
    all_gather_dict
)

def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric

@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    
    # prepare ground truth caption labels
    print("preparing corpus...")
    
    annotations = dataset_loader.dataset.annotations
    corpus = {
        '-'.join((anno['question_id'], anno['question'])): anno['answers'] \
            for anno in annotations
    }
    candidates = {}
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    net_dtype = next(model.parameters()).dtype
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    barrier()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    from tqdm import tqdm
    pbar = tqdm(total=num_batches, desc=f"Evaluate {epoch_str}")
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        pbar.update(1)
        curr_time = time.time()
        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            #     if batch_data_label[key].dtype == torch.float32:
            #         batch_data_label[key] = batch_data_label[key].to(net_dtype)
            # else:
            #     batch_data_label[key] = batch_data_label[key]
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask': batch_data_label['qformer_attention_mask'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
            'scan_idx' : batch_data_label['scan_idx'],
            'scan_name' : batch_data_label['scan_name']
        }
        if os.getenv("adaptive_pcd_input", False) == "True":
            model_input['sample_prob'] = batch_data_label['sample_prob']
            
        instruction = tokenizer.batch_decode(batch_data_label['instruction'],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False
                                            )
        if args.finetune_flex_opt:
            outputs = model(batch_data_label, is_eval=True, task_name='qa')
        else:
            outputs = model(batch_data_label, is_eval=True, task_name='qa')
        
        attentions=outputs["attentions"]
        
        outputs = dict(
            output_ids=outputs["output_ids"],
        )
        
        outputs = all_gather_dict(outputs)
        batch_data_label = all_gather_dict(batch_data_label)
        
        output_ids = outputs["output_ids"]  # batch x max_length
        answers = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        sample_index = batch_data_label['scan_idx'].cpu().tolist()
        gt_answers = [annotations[sample_index[idx]]['answers'] for idx in range(output_ids.shape[0])]
        print(f"GT: {gt_answers}")
        print(f"Pred: {answers}")
        for idx in range(output_ids.shape[0]):
            anno = annotations[sample_index[idx]]
            key = '-'.join((anno['question_id'], anno['question']))
            answer = answers[idx]
            answer = ' '.join(filter(lambda w: w, answer.split(' ')))
            candidates[key] = [answer]
        
        ll3da_opt_attn_output = os.getenv("ll3da_opt_attn_output", 'False')
        if ll3da_opt_attn_output == 'True':
            opt_attn_op_dir = os.environ['LL3DA_ATTN_OP_DIR']
            for idx in range(output_ids.shape[0]):
                anno = annotations[sample_index[idx]]
                answer = answers[idx]
                attn = torch.cat(attentions[idx], dim=0)
                instr = batch_data_label['instruction'][idx][batch_data_label['instruction_mask'][idx].bool()]
                opt_attn_dict = {
                    'anno' : anno,
                    'pred_answer' : answer,
                    'attn' : attn,
                    'instr' : instr,
                    'output_ids': output_ids[idx]
                }
                idx = batch_data_label['scan_idx'][idx].item()
                os.makedirs(os.path.join(opt_attn_op_dir, str(idx)), exist_ok=True)
                torch.save(opt_attn_dict, os.path.join(opt_attn_op_dir, str(idx), 'opt_attn.pt'))
            

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)
        
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logout(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                f"Evaluating on iter: {curr_train_iter}; "
                f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
        barrier()
    
    # end of forward pass traversion
    score_per_caption, message, eval_metric = score_captions(
        OrderedDict([(key, corpus[key]) for key in candidates]), candidates
    )
    
    if is_primary():
        logout("\n----------------------Evaluation-----------------------\n")
        logout(message)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        
        with open(os.path.join(args.log_dir, "qa_scores.json"), "w") as f: 
            json.dump(message, f)
        
        with open(os.path.join(args.log_dir, "qa_corpus_val.json"), "w") as f: 
            json.dump(corpus, f, indent=4)
        
        with open(os.path.join(args.log_dir, "qa_pred_val.json"), "w") as f:
            json.dump(candidates, f, indent=4)
        
        with open(os.path.join(args.log_dir, "qa_pred_gt_val.json"), "w") as f:
            pred_gt_val = {}
            for scene_object_id, scene_object_id_key in enumerate(candidates):
                pred_gt_val[scene_object_id_key] = {
                    'pred': candidates[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                    'score': {
                        'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                        'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                        'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                        'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                        'CiDEr': score_per_caption['cider'][scene_object_id],
                        'rouge': score_per_caption['rouge'][scene_object_id],
                        'meteor': score_per_caption['meteor'][scene_object_id]
                    }
                }
            json.dump(pred_gt_val, f, indent=4)
            
    return eval_metric