export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_object_caption \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --test_ckpt ckpts/opt-1.3b/flex/test-flex-8layer-freeze-llm/checkpoint.pth \
    --checkpoint_dir results\
    --finetune_flex_opt \
    --dist_url tcp://localhost:12245 \
    --batchsize_per_gpu 1 --ngpus 1 \
    --cache_dir results/debug \
    --test_only \
    --use_beam_search \
    --log_dir results/top_exp/object_caption/flex/flex-8layer-freeze-llm-2epoch
    # --gradient_checkpoint 