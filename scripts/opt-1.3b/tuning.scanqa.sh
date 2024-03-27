export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 0 \
    --dataset unified_scanqa \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/opt-1.3b/ll3da-scanqa-tuned \
    --max_epoch 5 \
    --dist_url tcp://localhost:12345 \
    --eval_every_iteration 4000000 \
    --start_eval_after -1 \
    --save_every 1000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 16 --ngpus 4 --base_lr 1e-6 --final_lr 1e-6 \
    --max_des_len 224 \
    --max_prompt 1 --use_beam_search \
    --pretrained_weights ./ckpts/opt-1.3b/ll3da-scannet/checkpoint_best.pth \
    # --adaptive_pcd_input 
