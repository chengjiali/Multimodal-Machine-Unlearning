#!/bin/bash

set -e

cd /data/jiali/vlul

export WANDB_MODE=offline

export M='retrain'
export TASK='snli_ve'
export BACKBONE='blip'


export DF=1000
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node=2 unlearn.py \
                --unlearn_method ${M} \
                --backbone ${BACKBONE} \
                --task ${TASK} \
                --df_size ${DF} \
                --cfg-path configs/original/${BACKBONE}/${TASK}.yaml
