export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=5,6,7,4,0,1,2,3
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir ckpts/opt-1.3b/gpu003/ll3da-scannet-adaptive-traindet-wodetloss \
    --test_ckpt ckpts/opt-1.3b/ll3da-scannet/checkpoint_best.pth \
    --dataset unified_object_caption \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:12165 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 24 --ngpus 4 \
    --max_des_len 256 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only \
    --log_dir results/toy_exp/scanqa/base-model \
    --adaptive_pcd_input 


