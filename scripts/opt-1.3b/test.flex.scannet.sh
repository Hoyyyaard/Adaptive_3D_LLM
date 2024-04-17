export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --test_ckpt ckpts/opt-1.3b/flex/encoder-openscene-maskformer-axis-align-w-sm-obj/checkpoint_38k.pth \
    --checkpoint_dir results\
    --dist_url tcp://localhost:12245 \
    --batchsize_per_gpu 1 --ngpus 1 \
    --cache_dir results/debug \
    --test_only \
    --use_beam_search \
    --log_dir results/toy_exp/unified_small_object_scanqa/flex/encoder-openscene-maskformer-axis-align-w-sm-obj/38k \
    --freeze_flex_llm \
    --load_pretrain_encoder \
    --finetune_flex_opt \
    --num_finetune_hidden_layers 0
    # --gradient_checkpoint 