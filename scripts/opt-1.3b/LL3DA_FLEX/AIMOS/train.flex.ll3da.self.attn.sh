export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# export CUDA_VISIBLE_DEVICES=2,3,5,6,4,0,7,1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
NODE_RANK=${SLURM_PROCID}

# if ']' is the last character of the node list
SLURM=${SLURM_NODELIST:0:3}

if [ "${SLURM_NODELIST: -1}" == "]" ]; then
    if [ $SLURM == "npl" ]; then
        # NPL
        ip=${SLURM}${SLURM_NODELIST:4:2}
    else
        # DCS
        ip=${SLURM}${SLURM_NODELIST:4:3}
    fi
    FLAG=1
else
    ip=$SLURM_NODELIST
    FLAG=0
fi

NUM_GPUS_PER_NODE=$1

echo "ip: $ip"
echo "FLAG: $FLAG"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "NODE_RANK: $NODE_RANK"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"

if [ $FLAG -eq 1 ]; then
    NUM_NODES=${2:-1}
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK"
else
    CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

ckpt_dir=ckpts/opt-1.3b/nipus_exp/LL3DA_FLEX/SELFATTN-8LAYER
mkdir -p /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM/${ckpt_dir}
cd /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM
    $CMD main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_scanrefer,unified_densecap_nr3d,unified_3dllm_embodied_dialogue,unified_3dllm_embodied_planning,unified_3dllm_scene_description \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --pretrained_weights pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --checkpoint_dir ${ckpt_dir} \
    --max_epoch 32 \
    --freeze_llm \
    --freeze_detector \
    --criterion "CiDEr" \
    --dataset_num_workers 4 \
    --eval_every_iteration 100000000 \
    --dist_url tcp://localhost:12445 \
    --save_every 100000000 \
    --batchsize_per_gpu 8 --ngpus 4 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
    --only_finetune_self_attn \
    --num_finetune_hidden_layers 8 \
    --use_flex_attn --max_des_len 128 \
    --slurm_run    \
    --filter_name 'none'  >> /gpfs/u/home/LMCG/LMCGljnn/scratch/zhy/Adaptive_3D_LLM/${ckpt_dir}/log.log