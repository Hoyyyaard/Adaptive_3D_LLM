export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
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
    --checkpoint_dir ckpts/opt-1.3b/gpu003/region-encoder\
    --max_epoch 32 \
    --dist_url tcp://localhost:19345 \
    --eval_every_iteration 1000000 \
    --start_eval_after 100 \
    --save_every 10000 \
    --criterion 'CiDEr' \
    --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 7 --base_lr 1e-4 --final_lr 1e-6 \
    --max_des_len 512 \
    --max_prompt 1 --use_beam_search \
    --adaptive_pcd_input \
    --cache_dir results/process_datasets/adaptive_pcds_adapt_scale_4w_wvote \
    --train_encoder \
    --adaptive_pcd_num 40000 \
