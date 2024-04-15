export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scene_list \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights ckpts/opt-model/pytorch_model.bin \
    --checkpoint_dir result/debug\
    --max_epoch 1 \
    --dataset_num_workers 8 \
    --finetune_flex_opt \
    --dist_url tcp://localhost:12245 \
    --save_every 500000000 \
    --batchsize_per_gpu 1 --ngpus 8 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
    --token_preprocess \
    --num_flex_hidden_layers 0
