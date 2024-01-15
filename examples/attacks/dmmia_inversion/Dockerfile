# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG wandb_key=""

WORKDIR /workspace

RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

RUN if [[ "$wandb_key" = "" ]] ; then echo WandB API key not provided ; else wandb login $wandb_key; fi

ENTRYPOINT ["/entry.sh"]
