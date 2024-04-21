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
    --dataset unified_scanqa \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --test_ckpt ckpts/opt-1.3b/nipus_exp/encoder-openscene-maskformer-axis-align-wocausal-finetune-opt-1-3b/checkpoint.pth \
    --checkpoint_dir results\
    --dist_url tcp://localhost:12245 \
    --batchsize_per_gpu 1 --ngpus 1 \
    --cache_dir results/debug \
    --test_only \
    --use_beam_search \
    --log_dir results/toy_exp/nipus_exp/unified_scanqa/encoder-openscene-maskformer-axis-align-wocausal-finetune-opt-1-3b/5epoch \
    --finetune_opt1_3b \
    --finetune_flex_opt \
    --num_finetune_hidden_layers 0 \
    --openscene_cache_dir /mnt/nfs/share/Adaptive/0420_openscene_scene_tokens_axis_align_w_pcd_info_s_512_0.2_128 \
    # --gradient_checkpoint 