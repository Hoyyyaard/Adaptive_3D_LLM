export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir ckpts/opt-1.3b/ll3da-scannet \
    --test_ckpt ckpts/opt-1.3b/ll3da-scannet/checkpoint_best.pth \
    --dataset unified_densecap_scanrefer \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:11222 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 24 --ngpus 8 \
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only \
    --log_dir results/toy_exp/scanrefer/size_large_than_1 \
    --special_dataset results/size_filter_scanrefer_datasets/val_size_l_1_num271.json