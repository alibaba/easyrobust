#!/usr/bin/env bash

set -x

EXP_DIR=exps/ddetr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size=1 \
    --no_encoder --no_input_proj \
    --no_dec_def_attn --num_feature_levels=1 \
    --enable_iouaware --enable_tokenlabel \
    --pvt=ddetr \
    ${PY_ARGS}