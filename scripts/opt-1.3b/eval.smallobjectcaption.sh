export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=6
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir results \
    --test_ckpt ckpts/opt-1.3b/adaptive/adaptive-region-encoder-tuned-scannet/checkpoint.pth \
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
    --log_dir results/toy_exp/small_obj_caption/click/adaptive-region-encoder-tuned-scannet \
    --adaptive_pcd_input \
    --cache_dir results/process_datasets/adaptive_pcds_4w/ \
    --adaptive_pcd_num 40000 \
    --no_sample_prob \

