export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
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
    --pretrained_weights ckpts/opt-model/pytorch_model.bin \
    --checkpoint_dir ckpts/opt-1.3b/flex/model/encoder-openscene-maskformer-axis-align-w-sm-obj/4layer/only_finetune_self_attn\
    --max_epoch 8 \
    --dataset_num_workers 4 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12445 \
    --save_every 2000 \
    --batchsize_per_gpu 3 --ngpus 6 --base_lr 1e-5 --final_lr 1e-6 \
    --cache_dir results/debug \
    --only_finetune_self_attn \
    --num_finetune_hidden_layers 4 \
    --load_pretrain_encoder \
    # --gradient_checkpoint \
