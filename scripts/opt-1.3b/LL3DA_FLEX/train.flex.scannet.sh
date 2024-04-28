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
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --checkpoint_dir ckpts/opt-1.3b/nipus_exp/LL3DA_FLEX/test \
    --max_epoch 32 \
    --freeze_llm \
    --freeze_detector \
    --criterion "CiDEr" \
    --dataset_num_workers 4 \
    --eval_every_iteration 1000000 \
    --dist_url tcp://localhost:12445 \
    --save_every 2000 \
    --batchsize_per_gpu 2 --ngpus 8 --base_lr 5e-3 --final_lr 5e-5 \
    --cache_dir results/debug \
    --finetune_flex_self_attn \
    --num_finetune_hidden_layers 8 \
    --use_flex_attn --max_des_len 128 \
    --gradient_checkpoint \
