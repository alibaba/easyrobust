export PYTHONIOENCODING=UTF-8

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
export  LANG="en_US.UTF-8"

set -e
set -u
#set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean"

#asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
#lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

asr_config=exp/asr_train_asr_conformer5_raw_bpe5000_frontend_confn_fft400_frontend_confhop_length160_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_sp/config.yaml
lm_config=exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/config.yaml
lm_config=conf/tuning/train_lm_adam.yaml

asr_config=exp_2/exp/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp/config.yaml
lm_config=exp_2/exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/config.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"


