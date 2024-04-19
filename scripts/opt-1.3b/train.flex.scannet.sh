export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=2,3,5,6,4,0,7,1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_scanrefer,unified_densecap_nr3d,unified_3dllm_embodied_dialogue,unified_3dllm_embodied_planning,unified_3dllm_scene_description \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights ckpts/opt-1.3b/nipus_exp/encoder-openscene-maskformer-axis-align-w-sm-obj-wocausal/checkpoint_32k.pth \
    --checkpoint_dir ckpts/opt-1.3b/nipus_exp/finetune_model/encoder-openscene-maskformer-axis-align-w-sm-obj-wocausal/4layer/finetune_flex_self_attn_encoder \
    --max_epoch 8 \
    --dataset_num_workers 4 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12445 \
    --save_every 2000 \
    --batchsize_per_gpu 1 --ngpus 7 --base_lr 1e-5 --final_lr 1e-6 \
    --cache_dir results/debug \
    --finetune_flex_self_attn \
    --load_pretrain_encoder \
    --num_finetune_hidden_layers 4 \
    # --gradient_checkpoint \
