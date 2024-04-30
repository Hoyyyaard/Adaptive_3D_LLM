import os, sys, time, math, json
import torch
import datetime
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor
import torch.distributed as dist
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
from tqdm import tqdm

class Logger:
    def __init__(self, args):
        exp_name = os.path.split(args.checkpoint_dir)[-1]
        self.logger = open(os.path.join(args.checkpoint_dir, f'{exp_name}-logger.log'), 'a')
    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr

def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr

def do_preprocess(
    args,
    model,
    model_no_ddp,
    optimizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    
    logout = Logger(args)
    
    if is_primary():
        logout(f"call with args: {args}")

    barrier()

    pbar = tqdm(total=len(dataloaders['train']))
    for batch_idx, batch_data_label in enumerate(dataloaders['train']):
        pbar.update(1)
        continue
    
def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    
    logout = Logger(args)
    
    if is_primary():
        logout(f"call with args: {args}")
        # logout(f"{model}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()
    
    max_tolerant_nan = 4
    curr_nan_times = 0
    
    if args.preprocess_dense_token:
        import copy
        tokenizer = copy.deepcopy(model_no_ddp.detector.tokenizer)
        sample_point = int(model_no_ddp.captioner.transformer.model.decoder.dense_token_selection._preenc_npoints * \
                            model_no_ddp.captioner.transformer.model.decoder.dense_token_selection._query_topk * \
                            model_no_ddp.captioner.transformer.model.decoder.dense_token_selection._scene_token_topk)
        tokenizer.npoints = sample_point * 2
        encoder = copy.deepcopy(model_no_ddp.detector.encoder)
        encoder.interim_downsampling.npoint = sample_point

    for curr_epoch in tqdm(range(args.start_epoch, args.max_epoch)):
        
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(curr_epoch)
        
        pbar = tqdm(total=len(dataloaders['train']))
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            pbar.update(1)
    
            curr_time = time.time()
            
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            for key in batch_data_label:
                if not isinstance(batch_data_label[key], list):
                    batch_data_label[key] = batch_data_label[key].to(net_device)
            
            # max_len = 1e-9
            # for attn_mask in batch_data_label['attention_mask']:
            #     if attn_mask.sum() > max_len:
            #         max_len = attn_mask.sum()
            # if is_distributed():
            #     seq_len_list = [torch.empty_like(max_len) for _ in range(args.ngpus)]
            #     dist.all_gather(seq_len_list, max_len)
            #     max_len = max(seq_len_list).item()
            # else:
            #     max_len = max_len.item()
            # batch_data_label['input_ids'] = batch_data_label['input_ids'][:, :int(max_len)]
            # batch_data_label['attention_mask'] = batch_data_label['attention_mask'][:, :int(max_len)]
            # batch_data_label['gradient_mask'] = batch_data_label['gradient_mask'][:, :int(max_len)]
            
    
            # Forward pass
            optimizer.zero_grad()
    
            if args.preprocess_dense_token:
                def _break_up_pc(pc):
                    # pc may contain color/normals.
                    xyz = pc[..., 0:3].contiguous()
                    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
                    return xyz, features
                
                def _run_encoder(point_clouds, inds=None):
                    xyz, features = _break_up_pc(point_clouds)
                    
                    ## pointcloud tokenization
                    # xyz: batch x npoints x 3
                    # features: batch x channel x npoints
                    # inds: batch x npoints
                    pre_enc_xyz, pre_enc_features, pre_enc_inds = tokenizer(xyz, features, inds)

                    # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
                    pre_enc_features = pre_enc_features.permute(2, 0, 1)

                    # xyz points are in batch x npointx channel order
                    enc_xyz, enc_features, enc_inds = encoder(
                        pre_enc_features, xyz=pre_enc_xyz
                    )
                    if enc_inds is None:
                        # encoder does not perform any downsampling
                        enc_inds = pre_enc_inds
                    else:
                        # use gather here to ensure that it works for both FPS and random sampling
                        enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
                    return enc_features

                base_cache_dir = 'results/process_datasets/ll3da_flex_gt_dense_token'
                ## STEP1: CROP出GT相关的点云
                for bsz in range(batch_data_label['point_clouds'].shape[0]):
                    scan_name = batch_data_label['scan_name'][bsz].split('_')[0]
                    task_name = batch_data_label['task_name'][bsz]
                    ## 获得稠密的点云和标签
                    pointclouds = model_no_ddp.captioner.transformer.model.decoder.dense_token_selection.pcd_dict[scan_name][0].cpu().numpy()
                    pcd = pointclouds[:,:3]
                    instance_label = model_no_ddp.captioner.transformer.model.decoder.dense_token_selection.pcd_dict[scan_name][2]
                    tgt_related_pcd = pointclouds[instance_label == batch_data_label['tgt_obj_id'][bsz].item()+1]

                    ## 采样TOKEN
                    tgt_related_pcd = torch.from_numpy(tgt_related_pcd).float().cuda()
                    tgt_related_pcd = tgt_related_pcd.unsqueeze(0)

                    scan_idx = batch_data_label['scan_idx'][bsz].item()

                    ## STEP2：采样TOKEN
                    ## STEP3: 生成TOKEN的特征
                    reg_features = _run_encoder(tgt_related_pcd).permute(1, 0, 2)
                    save_p = os.path.join(base_cache_dir, 'val', task_name, scan_name, f'{scan_idx}.pt')
                    os.makedirs(os.path.dirname(save_p), exist_ok=True)
                    torch.save(reg_features, save_p)

                continue
    
            outputs = model(batch_data_label, is_eval=False)
            

            loss = outputs['loss']
            loss = all_reduce_average(loss)
            
            if not math.isfinite(loss.item()):
                if curr_nan_times < max_tolerant_nan:
                    logout("Loss in not finite. Skip this training step.")
                    curr_nan_times += 1
                    continue
                else:
                    logout("Loss in not finite. Terminate training.")
                    exit(-1)
            curr_nan_times = 0
            
            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
                
            # for n, p in model_no_ddp.named_parameters():
            #     assert not torch.isnan(p).any()
            
            optimizer.step()
            
            # for n, p in model_no_ddp.named_parameters():
            #     assert not torch.isnan(p).any()
    
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
    
            # logging
            if is_primary() and curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
            
            barrier()
            # save ckpt
            if is_primary() and (curr_iter + 1) % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    curr_epoch,
                    args,
                    best_val_metrics,
                    filename=f"checkpoint_{(curr_iter + 1) // 1000}k.pth",
                )
            
            # eval
            if (curr_iter + 1) % args.eval_every_iteration == 0 \
                and (curr_iter + 1) > args.start_eval_after:
                
                eval_metrics = {}
                model.eval()
                for test_loader in dataloaders['test']:
                    task_metrics = test_loader.dataset.eval_func(
                        args,
                        curr_epoch,
                        model,
                        dataset_config,
                        test_loader,
                        logout,
                        curr_train_iter=curr_iter
                    )
                    eval_metrics.update(task_metrics)
                model.train()
                
                if not best_val_metrics or (
                    best_val_metrics[args.criterion] < eval_metrics[args.criterion]
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        model_no_ddp,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    if is_primary():
                        logout(
                            f"Epoch [{curr_epoch}/{args.max_epoch}] "
                            f"saved current best val checkpoint at {filename}; "
                            f"{args.criterion} {eval_metrics[args.criterion]}"
                        )
            # end of an iteration
        
          
        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename=f"checkpoint_{(curr_epoch + 1)}epoch.pth",
        )
        
        eval_metrics = {}
        model.eval()
        for test_loader in dataloaders['test']:
            task_metrics = test_loader.dataset.eval_func(
                args,
                curr_epoch,
                model,
                dataset_config,
                test_loader,
                logout,
                curr_train_iter=curr_iter
            )
            eval_metrics.update(task_metrics)
            if is_primary():
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}] "
                    f"{args.criterion} {eval_metrics[args.criterion]}"
                )
        model.train()
        
        if not best_val_metrics or (
            best_val_metrics[args.criterion] < eval_metrics[args.criterion]
        ):
            best_val_metrics = eval_metrics
            filename = "checkpoint_best.pth"
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                curr_epoch,
                args,
                best_val_metrics,
                filename="checkpoint_best.pth",
            )
            if is_primary():
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}] "
                    f"{args.criterion} {eval_metrics[args.criterion]}"
                )
    
    # end of training
    eval_metrics = {}
    model.eval()
    for test_loader in dataloaders['test']:
        task_metrics = test_loader.dataset.eval_func(
            args,
            curr_epoch,
            model,
            dataset_config,
            test_loader,
            logout,
            curr_train_iter=curr_iter
        )
        eval_metrics.update(task_metrics)
    return 
    
def do_flex_opt_finetune(
    args,
    model,
    model_no_ddp,
    optimizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    
    logout = Logger(args)
    
    if is_primary():
        logout(f"call with args: {args}")
        logout(f"{model_no_ddp.model.config}")
    
    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device
    net_dtype = next(model.parameters()).dtype

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)


    barrier()
    
    max_tolerant_nan = 4
    curr_nan_times = 0
    
    for curr_epoch in tqdm(range(args.start_epoch, args.max_epoch)):
        
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(curr_epoch)
        
        pbar = tqdm(total=len(dataloaders['train']))
        for batch_idx, batch_data_label in enumerate(dataloaders['train']):
            pbar.update(1)

            curr_time = time.time()
              
            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            for key in batch_data_label:
                if not isinstance(batch_data_label[key], list):
                    batch_data_label[key] = batch_data_label[key].to(net_device)
                    if not args.token_preprocess:
                        if batch_data_label[key].dtype == torch.float32:
                            batch_data_label[key] = batch_data_label[key].to(net_dtype)
                else:
                    batch_data_label[key] = batch_data_label[key]
            
            # Forward pass
            optimizer.zero_grad()
            if args.token_preprocess:
                model_no_ddp.model.forward_preprocess_scene_token(batch_data_label)
                continue
            else:
                outputs = model(batch_data_label, is_eval=False)
            loss = outputs['loss']
            loss = all_reduce_average(loss)
            
            if not math.isfinite(loss.item()):
                if curr_nan_times < max_tolerant_nan:
                    logout("Loss in not finite. Skip this training step.")
                    curr_nan_times += 1
                    continue
                else:
                    logout("Loss in not finite. Terminate training.")
                    exit(-1)
            curr_nan_times = 0
            
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model_no_ddp.model.parameters(), args.clip_gradient)
            
            # for n, p in model_no_ddp.model.named_parameters():
            #     assert not torch.isnan(p).any()
                
            optimizer.step()
            
            # for p in model_no_ddp.model.parameters():
            #     assert not torch.isnan(p).any()
    
            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())
    
            # logging
            if is_primary() and curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
            
            barrier()
            # save ckpt
            if is_primary() and (curr_iter + 1) % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp.model,
                    optimizer,
                    curr_epoch,
                    args,
                    best_val_metrics,
                    filename=f"checkpoint_{(curr_iter + 1) // 1000}k.pth",
                )
            
            # eval
            # if (curr_iter + 1) % args.eval_every_iteration == 0 \
            #     and (curr_iter + 1) > args.start_eval_after:
                
            #     eval_metrics = {}
            #     model.eval()
            #     for test_loader in dataloaders['test']:
            #         task_metrics = test_loader.dataset.eval_func(
            #             args,
            #             curr_epoch,
            #             model,
            #             dataset_config,
            #             test_loader,
            #             logout,
            #             curr_train_iter=curr_iter
            #         )
            #         eval_metrics.update(task_metrics)
            #     model.train()
                
            #     if not best_val_metrics or (
            #         best_val_metrics[args.criterion] < eval_metrics[args.criterion]
            #     ):
            #         best_val_metrics = eval_metrics
            #         filename = "checkpoint_best.pth"
            #         save_checkpoint(
            #             args.checkpoint_dir,
            #             model_no_ddp,
            #             optimizer,
            #             curr_epoch,
            #             args,
            #             best_val_metrics,
            #             filename="checkpoint_best.pth",
            #         )
            #         if is_primary():
            #             logout(
            #                 f"Epoch [{curr_epoch}/{args.max_epoch}] "
            #                 f"saved current best val checkpoint at {filename}; "
            #                 f"{args.criterion} {eval_metrics[args.criterion]}"
            #             )
            # end of an iteration
        
          
        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp.model,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )
    
    # # end of training
    # eval_metrics = {}
    # model.eval()
    # for test_loader in dataloaders['test']:
    #     task_metrics = test_loader.dataset.eval_func(
    #         args,
    #         curr_epoch,
    #         model,
    #         dataset_config,
    #         test_loader,
    #         logout,
    #         curr_train_iter=curr_iter
    #     )
    #     eval_metrics.update(task_metrics)
    return 