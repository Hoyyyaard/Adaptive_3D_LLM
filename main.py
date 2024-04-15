import os, argparse, importlib
import numpy as np
import torch

from collections import OrderedDict
import copy

from engine import do_train, do_preprocess, do_flex_opt_finetune
from models.model_general import CaptionNet
from datasets.scannet_base_dataset import DatasetConfig
from torch.multiprocessing import set_start_method
from transformers import AutoConfig
from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier


def make_args_parser():
    parser = argparse.ArgumentParser("LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    # DISABLE warmup learning rate during dense caption training
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    # only ACTIVATE during dense caption training
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    parser.add_argument("--pretrained_weights", default=None, type=str)
    
    
    ##### Model #####
    # input based parameters
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    parser.add_argument(
        "--detector", default="detector_Vote2Cap_DETR", 
        help="folder of the detector"
    )
    parser.add_argument(
        "--captioner", default=None, type=str, help="folder of the captioner"
    )
    # training strategy
    parser.add_argument(
        "--freeze_detector", default=False, action='store_true', 
        help="freeze all parameters other than the caption head"
    )
    parser.add_argument(
        "--freeze_llm", default=False, action='store_true', 
        help="freeze the llm for caption generation"
    )
    # caption related hyper parameters
    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument(
        "--max_des_len", default=128, type=int, 
        help="maximum length of object descriptions."
    )
    parser.add_argument(
        "--max_gen_len", default=32, type=int, 
        help="maximum length of object descriptions."
    )
    
    ##### Dataset #####
    parser.add_argument("--max_prompts", default=16, type=int, help="number of visual interactions")
    parser.add_argument("--dataset", default='scannet', help="dataset list split by ','")
    parser.add_argument("--grid_size_3d", default=255, type=int, help="grid size of the 3D scene")    
    parser.add_argument('--vocab', default="llama-hf/7B", type=str, help="The LLM backend")
    parser.add_argument('--qformer_vocab', default="bert-base-uncased", type=str, help="The QFormer backend")
    
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)
    parser.add_argument("--eval_every_iteration", default=4000, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument(
        "--criterion", default='CiDEr', type=str,
        help='metrics for saving the best model'
    )
    parser.add_argument("--test_ckpt", default="", type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--save_every", default=4000, type=int)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--filter_name", default='captioner.transformer.', type=str)
    
    ##### Distributed #####
    parser.add_argument("--ngpus", default=1, type=int, help='number of gpus')
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)
    
    parser.add_argument("--log_dir", default='results/debug', type=str)
    parser.add_argument("--special_dataset", default=None, type=str)
    
    ##### USer #####
    parser.add_argument("--adaptive_pcd_input", default=False, action='store_true')
    parser.add_argument("--caption_box_query", default=False, action='store_true')
    parser.add_argument("--adaptive_pcd_num", default=10000)
    parser.add_argument("--adaptive_pcd_scale", default=1)
    parser.add_argument("--preprocess_pcd", action='store_true')
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--vis_detection", action='store_true')
    parser.add_argument("--train_encoder", action='store_true')
    parser.add_argument("--only_sample_tgt", action='store_true')
    ## Use less vote to model dense region
    parser.add_argument("--local_config", action='store_true')
    ## Use to train det like official to random sample Pseed
    parser.add_argument("--no_sample_prob", action='store_true')
    
    parser.add_argument("--finetune_flex_opt", action='store_true')
    parser.add_argument("--token_preprocess", action='store_true')
    parser.add_argument("--freeze_flex_llm", action='store_true')
    parser.add_argument("--load_pretrain_encoder", action='store_true')
    parser.add_argument("--gradient_checkpoint", action='store_true')
    parser.add_argument("--num_flex_hidden_layers", required=True, type=int)
    
    args = parser.parse_args()
    args.use_height = not args.no_height
    
    os.environ['adaptive_pcd_input'] = str(args.adaptive_pcd_input)
    os.environ['adaptive_pcd_num'] = str(args.adaptive_pcd_num)
    os.environ['only_sample_tgt'] = str(args.only_sample_tgt)
    os.environ['no_sample_prob'] = str(args.no_sample_prob)
    print(f'adaptive_pcd_input: {args.adaptive_pcd_input}')
    print(f'caption_box_query: {args.caption_box_query}')
    print(f'cache dir: {args.cache_dir}')
    print(f'adaptive_pcd_num: {args.adaptive_pcd_num}')
    print(f'train_encoder: {args.train_encoder}')
    print(f'local_config: {args.local_config}')
    print(f'no_sample_prob: {args.no_sample_prob}')
    print('============== fintune flex opt =====================')
    print(f'finetune_flex_opt: ', args.finetune_flex_opt)
    print(f'token_preprocess: ', args.token_preprocess)
    print(f'freeze_flex_llm: ', args.freeze_flex_llm)
    print(f'load_pretrain_encoder: ', args.load_pretrain_encoder)
    print(f'gradient_checkpoint: ', args.gradient_checkpoint)
    print(f'num flex layers: ', args.num_flex_hidden_layers)
    
    return args


def build_dataloader_func(args, dataset, split):
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset, 
            shuffle=(split=='train')
        )
    else:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )
    return sampler, dataloader

def build_dataset(args):
    
    dataset_config = DatasetConfig()
    datasets = {'train': None, 'test': []}
    
    train_datasets = []
    for dataset in args.dataset.split(','):
        dataset_module = importlib.import_module(f'datasets.{dataset}')
        train_datasets.append(
            dataset_module.Dataset(
                args,
                dataset_config, 
                split_set="train", 
                use_color=args.use_color,
                use_normal=args.use_normal,
                use_multiview=args.use_multiview,
                use_height=args.use_height,
                augment=True
            )
        )
        datasets['test'].append(
            dataset_module.Dataset(
                args,
                dataset_config, 
                split_set="val", 
                use_color=args.use_color,
                use_normal=args.use_normal,
                use_multiview=args.use_multiview,
                use_height=args.use_height,
                augment=False
            )
        )
    
    ## USer : tmp implement for adapative pcd input trainning
    # print("Use object caption task for test")
    # dataset_module = importlib.import_module(f'datasets.unified_object_caption')
    # datasets['test'] = [ 
    #         dataset_module.Dataset(
    #             args,
    #             dataset_config, 
    #             split_set="val", 
    #             use_color=args.use_color,
    #             use_normal=args.use_normal,
    #             use_multiview=args.use_multiview,
    #             use_height=args.use_height,
    #             augment=False
    #         )]
    
    datasets['train'] = torch.utils.data.ConcatDataset(train_datasets)
    
    train_sampler, train_loader = build_dataloader_func(args, datasets['train'], split='train')
    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)
    
    return dataset_config, datasets, dataloaders    
    
def main(local_rank, args):
    
    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )
    
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + get_rank())
    
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!'
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = CaptionNet(args, dataset_config, datasets['train'])
    
    # testing phase
    if args.test_only:

        # try:
        checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"], strict=False)
        # except:
        #     print('test the model from scratch...')
        
        model_no_ddp = model.cuda()
        model = model.cuda(local_rank)
        
        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        
        for test_loader in dataloaders['test']:
            test_loader.dataset.eval_func(
                args,
                -1,
                model,
                dataset_config,
                test_loader
            )
        
    # training phase
    else:
        
        assert (
            args.checkpoint_dir is not None
        ), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        ### whether or not use pretrained weights
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            model.load_state_dict(checkpoint['model'], strict=False)
            
            print('====                                          ====')
            print('==== loading following pre-trained parameters ====')
            print('====                                          ====')
            for name, param in checkpoint['model'].items():
                print('\t', name, param.shape)
        
        if not args.preprocess_pcd:
            model_no_ddp = model.cuda()
            model = model.cuda(local_rank)
            
            if is_distributed():
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank],
                )
                
            if args.optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(
                    filter(lambda params: params.requires_grad, model_no_ddp.parameters()), 
                    lr=args.base_lr, 
                    weight_decay=args.weight_decay
                )
            elif args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(
                    filter(lambda params: params.requires_grad, model_no_ddp.parameters()), 
                    lr=args.base_lr, 
                    weight_decay=args.weight_decay
                )
            else:
                raise NotImplementedError
            
            print('====                                          ====')
            print('====  Only training the following parameters  ====')
            print('====                                          ====')
            for name, param in model_no_ddp.named_parameters():
                if param.requires_grad is True:
                    print('\t', name, param.shape)
            
            loaded_epoch, best_val_metrics = resume_if_possible(
                args.checkpoint_dir, model_no_ddp, optimizer
            )
            args.start_epoch = loaded_epoch + 1
            do_train(
                args,
                model,
                model_no_ddp,
                optimizer,
                dataset_config,
                dataloaders,
                best_val_metrics,
            )
        else:
            do_preprocess(
                args,
                None,
                None,
                None,
                None,
                dataloaders,
                None,
            )
            
def finetune_flex_opt_main(local_rank, args):
    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )
    
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + get_rank())
    
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!'
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    from src.modeling_opt_flex import FlexOPTForCausalLM, Shell_Model
    config = AutoConfig.from_pretrained('src/flex_opt_config.json')
    config.num_flex_hidden_layers = args.num_flex_hidden_layers + 1
    config.num_hidden_layers = 24 - args.num_flex_hidden_layers - 1
    print("acc_num_flex_hidden_layers: ", config.num_flex_hidden_layers)
    print("acc_num_hidden_layers: ", config.num_hidden_layers)
    if args.freeze_flex_llm:
        config.num_hidden_layers = config.num_flex_hidden_layers + config.num_hidden_layers
        config.num_flex_hidden_layers = 0
        print('============================freeze llm====================================')
    model = Shell_Model(config=config)
    if not args.freeze_flex_llm:
        ## 由于代码上的bug 第一层的flex self attn相当于self attn
        del model.model.model.decoder.flex_layers[0].self_attn.k_hr_proj
        del model.model.model.decoder.flex_layers[0].self_attn.v_hr_proj
    # testing phase
    if args.test_only:

        # try:
        checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
        if args.test_ckpt.split('/')[-1] == 'pytorch_model.bin':
            replace_keys = [f'model.decoder.layers.{li+config.num_hidden_layers}' for li in range(config.num_flex_hidden_layers)]
            flex_checkpoint = copy.deepcopy(checkpoint)
            for k,v in checkpoint.items():
                find_key = None
                for rk in replace_keys:
                    if k.find(rk) != -1:
                        find_key = rk
                        break
                if not find_key is None:
                    layer_idx = int(find_key.split('.')[-1])-config.num_hidden_layers
                    flex_checkpoint[k.replace(find_key, f'model.decoder.flex_layers.{layer_idx}')] = v
                    flex_checkpoint.pop(k)
            checkpoint = flex_checkpoint
        if args.test_ckpt == 'ckpts/opt-model/pytorch_model.bin':
            msg = model.model.load_state_dict(checkpoint, strict=False)
        elif args.load_pretrain_encoder:
            msg = model.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            msg = model.model.load_state_dict(checkpoint['model'], strict=True)
        # except:
        #     print('test the model from scratch...')
        print(msg)
    
        
        # model_no_ddp = model.cuda()
        model = model.cuda(local_rank)
        
        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        
        with torch.no_grad():
            for test_loader in dataloaders['test']:
                test_loader.dataset.eval_func(
                    args,
                    -1,
                    model,
                    dataset_config,
                    test_loader
                )
        
    # training phase
    else:
        
        if args.gradient_checkpoint:
            print('Training use gradient checkpointing...')
            # from functools import partial
            # notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=True)
            # torch.utils.checkpoint.checkpoint = notfailing_checkpoint
            model.model.gradient_checkpointing_enable()
            model.model.model.decoder.gradient_checkpointing = True
        assert (
            args.checkpoint_dir is not None
        ), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        ### whether or not use pretrained weights
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            ## 记载官方模型时：使用原来的self attn的参数初始化flex attn的参数 这里需要改网络名字
            if args.pretrained_weights.split('/')[-1] == 'pytorch_model.bin':
                replace_keys = [f'model.decoder.layers.{li+config.num_hidden_layers}' for li in range(config.num_flex_hidden_layers)]
                flex_checkpoint = copy.deepcopy(checkpoint)
                for k,v in checkpoint.items():
                    find_key = None
                    for rk in replace_keys:
                        if k.find(rk) != -1:
                            find_key = rk
                            break
                    if not find_key is None:
                        layer_idx = int(find_key.split('.')[-1])-config.num_hidden_layers
                        flex_checkpoint[k.replace(find_key, f'model.decoder.flex_layers.{layer_idx}')] = v
                        flex_checkpoint.pop(k)
                checkpoint = flex_checkpoint
            elif args.load_pretrain_encoder:
                checkpoint = checkpoint['model']
                replace_keys = [f'model.decoder.layers.{li+config.num_hidden_layers}' for li in range(config.num_flex_hidden_layers)]
                flex_checkpoint = copy.deepcopy(checkpoint)
                for k,v in checkpoint.items():
                    find_key = None
                    for rk in replace_keys:
                        if k.find(rk) != -1:
                            find_key = rk
                            break
                    if not find_key is None:
                        layer_idx = int(find_key.split('.')[-1])-config.num_hidden_layers
                        flex_checkpoint[k.replace(find_key, f'model.decoder.flex_layers.{layer_idx}')] = v
                        flex_checkpoint.pop(k)
                checkpoint = flex_checkpoint
            else:
                checkpoint = checkpoint['model']
            
            msg = model.model.load_state_dict(checkpoint, strict=False)
            
            print('====                                          ====')
            print('==== loading following pre-trained parameters ====')
            print('====                                          ====')
            # for name, param in checkpoint.items():
            #     print('\t', name, param.shape)
            print(msg)
                    
        model_no_ddp = model.cuda()
        
        
        for param in model_no_ddp.model.parameters():
            param.requires_grad = True
        model_no_ddp.model.train()
        
        ## Only train the linear layers
        if args.freeze_flex_llm:
            assert config.num_flex_hidden_layers == 0
            for name, param in model_no_ddp.model.named_parameters():
                if name in checkpoint.keys():
                    param.requires_grad_(False)
            
            print("---------------------Trainable parameters when freeze llm: ----------------------")
            for name, param in model_no_ddp.model.named_parameters():
                if param.requires_grad:
                    print(name)
        else:
            model_no_ddp.model.scene_token_in_head.weight.requires_grad_(False)
            print("---------------------freeze vision encoder----------------------")
        
        ## Only for test flex performance
        # for name, param in model_no_ddp.model.named_parameters():
        #      if name.find('hr_proj') != -1:
        #         param.requires_grad_(True)
        #      else:
        #         param.requires_grad_(False)
        
        
        if is_primary():
            print('====                                          ====')
            print('====  Only training the following parameters  ====')
            print('====                                          ====')
            for name, param in model_no_ddp.model.named_parameters():
                if param.requires_grad is True:
                    print('\t', name, param.shape)
        
        optimizer_param = filter(lambda params: params.requires_grad, model_no_ddp.model.parameters())
       
        
        model = model.cuda(local_rank)
        if is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank] , #find_unused_parameters=True
            )
            
        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                optimizer_param, 
                lr=args.base_lr, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                optimizer_param,
                lr=args.base_lr, 
                weight_decay=args.weight_decay
            )
        else:
            raise NotImplementedError
        
        # print("Parameters to be updated by the optimizer:")
        # for param_group in optimizer.param_groups:
        #     print(param_group['params'])
    
        
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp.model, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_flex_opt_finetune(
            args,
            model,
            model_no_ddp,
            optimizer,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    main_func = main if not args.finetune_flex_opt else finetune_flex_opt_main
    if world_size == 1:
        main_func(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main_func, nprocs=world_size, args=(args,))

if __name__ == "__main__":
    args = make_args_parser()
    
    os.environ['PYTHONWARNINGS']='ignore:semaphore_tracker:UserWarning'

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)