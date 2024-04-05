export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights ckpts/opt-model/pytorch_model.bin \
    --checkpoint_dir ckpts/opt-1.3b/flex/scene_token_512-dense_token_1wto128_r1-flex_layer_from8\
    --max_epoch 32 \
    --dataset_num_workers 1 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12245 \
    --save_every 5000 \
    --batchsize_per_gpu 4 --ngpus 7 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
