export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=2,3,5,6,4,0,1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

NCCL_SOCKET_IFNAME=eno1
NCCL_IB_DISABLE=1 
NCCL_P2P_DISABLE=1 
NCCL_DEBUG=INFO 

RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
NODE_RANK=0
ip=192.168.1.49
NUM_GPUS_PER_NODE=7

echo "ip: $ip"
echo "NODE_RANK: $NODE_RANK"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"


CMD="/gpfs/u/home/LMCG/LMCGljnn/scratch/miniconda3-ppc64le/envs/ll3da/bin/python -u -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK"


ckpt_dir=ckpts/opt-1.3b/nipus_exp/LL3DA_FLEX/FLEXATTN-8LAYER-SCANNET-EVALWOGT
mkdir -p ${ckpt_dir}
    $CMD main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --warm_lr_epochs 1 \
    --dataset unified_scanqa,unified_densecap_scanrefer,unified_densecap_nr3d \
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
    --batchsize_per_gpu 2 --ngpus 4 --base_lr 1e-4 --final_lr 1e-6 \
    --cache_dir results/debug \
    --finetune_flex_self_attn \
    --num_finetune_hidden_layers 8 \
    --use_beam_search \
    --use_flex_attn --max_des_len 128 \
    --slurm_run    \
    --filter_name 'none'  | tee ${ckpt_dir}/log.log