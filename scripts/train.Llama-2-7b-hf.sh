export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --pretrained_weights ./pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
    --vocab meta-llama/Llama-2-7b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/Llama-2-7b/ll3da-generalist \
    --max_epoch 32 \
    --dist_url tcp://localhost:12345 \
    --eval_every_iteration 10000 \
    --start_eval_after 19999 \
    --save_every 10000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 8 --base_lr 1e-4 --final_lr 1e-6 \
    --max_des_len 128 \
    --max_prompt 1 --use_beam_search \
    --num_finetune_hidden_layers 0 \
    --cache_dir results/debug
