export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 

# python -m torch.distributed.launch --nproc_per_node=7 \
#            --nnodes=2 --node_rank=0 --master_addr="172.16.0.3" \
#            --master_port=12445 \
python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_scanrefer,unified_densecap_nr3d,unified_3dllm_embodied_dialogue,unified_3dllm_embodied_planning,unified_3dllm_scene_description \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights ckpts/opt-model/pytorch_model.bin \
    --checkpoint_dir ckpts/opt-1.3b/nipus_exp/0423-EncoderMLP-WoVisualPrpmpt-ST128WoXYZ-OPT1_3bmWoCausalMask\
    --max_epoch 32 \
    --max_des_len 512 \
    --dataset_num_workers 4 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12445 \
    --save_every 10000 \
    --batchsize_per_gpu 4 --ngpus 8 --base_lr 1e-3 --final_lr 1e-5 \
    --cache_dir results/debug \
    --finetune_opt1_3b \
    --num_finetune_hidden_layers 0 \
    --openscene_cache_dir /mnt/nfs/share/Adaptive/0423_openscene_scene_tokens_axis_align_w_pcd_info_w_token_instance_label_s_128_0.4_512 \
    --gradient_checkpoint \
