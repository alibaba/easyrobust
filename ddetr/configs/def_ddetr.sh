#!/usr/bin/env bash

set -x

EXP_DIR=exps/ddetr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size=1 \
    --no_encoder --no_input_proj \
    --with_box_refine --two_stage \
    --enable_centerness --enable_iouaware --enable_tokenlabel \
    --pvt=def-ddetr \
    ${PY_ARGS}