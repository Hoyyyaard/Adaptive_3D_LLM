export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=1,2,3,0,5,6,4,7
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
    --checkpoint_dir ckpts/opt-1.3b/flex/pretrain\
    --max_epoch 32 \
    --dataset_num_workers 8 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12245 \
    --save_every 2000 \
    --batchsize_per_gpu 8 --ngpus 8 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
    --freeze_flex_llm
