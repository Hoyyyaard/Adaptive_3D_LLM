export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=2,3,5,6,4,0,7,1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

ckpt_dir=ckpts/opt-1.3b/nipus_exp/LL3DA_FLEX/FLEXATTN-8LAYER-GTTOKEN-SCANNET
mkdir -p ${ckpt_dir}
python -u main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_scanrefer,unified_densecap_nr3d \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --checkpoint_dir ${ckpt_dir} \
    --max_epoch 32 \
    --freeze_llm \
    --freeze_detector \
    --criterion "CiDEr" \
    --dataset_num_workers 4 \
    --eval_every_iteration 1000000 \
    --dist_url tcp://localhost:12445 \
    --save_every 1000000 \
    --batchsize_per_gpu 4 --ngpus 8 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
    --finetune_flex_self_attn \
    --num_finetune_hidden_layers 8 \
    --use_flex_attn --max_des_len 128 \
    --filter_name 'none' \
    --use_gt_dense_token   | tee ${ckpt_dir}/log.log
