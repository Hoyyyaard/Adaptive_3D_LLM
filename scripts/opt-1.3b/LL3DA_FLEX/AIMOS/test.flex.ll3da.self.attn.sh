export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM
    ~/scratch/miniconda3-ppc64le/envs/ll3da/bin/python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa \
    --vocab ckpts/opt-model \
    --qformer_vocab bert-base-embedding \
    --test_ckpt ckpts/opt-1.3b/nipus_exp/LL3DA_FLEX/SELFATTN-8LAYER/checkpoint_20k.pth \
    --checkpoint_dir results\
    --dist_url tcp://localhost:12245 \
    --batchsize_per_gpu 8 --ngpus 6 \
    --cache_dir results/debug \
    --test_only \
    --use_beam_search \
    --log_dir results/nipus_exp/unified_scanqa/LL3DA_FLEX/SELFATTN-8LAYER/20k \
    --only_finetune_self_attn \
    --use_flex_attn \
    --num_finetune_hidden_layers 8 \
    # --gradient_checkpoint 