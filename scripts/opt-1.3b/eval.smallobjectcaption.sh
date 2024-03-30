export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir results \
    --test_ckpt ckpts/opt-1.3b/adaptive/new-encoder-1w/checkpoint_10k.pth \
    --dataset unified_small_object_caption_embodiedscan \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:12165 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 16 --ngpus 1\
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only \
    --log_dir results/debug \
    --adaptive_pcd_input \
    --cache_dir None \
    --adaptive_pcd_num 10000 \
    --caption_box_query

