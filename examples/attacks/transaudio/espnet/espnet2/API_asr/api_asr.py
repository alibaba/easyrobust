# -*- coding: utf8 -*-
import json
import numpy
import numpy as np
import os
import sys
sys.path.append("..")
import time
import threading
from datetime import datetime
from espnet2.API_tools.tencent.common import credential
from espnet2.API_tools.tencent.asr import flash_recognizer
from espnet2.API_tools.aliyun import fileTrans
from espnet2.API_tools.huawei import MyCallback, wav2pcm
from espnet2.API_tools.zhibo import API_cls_
from huaweicloud_sis.client.rasr_client import RasrClient
from huaweicloud_sis.bean.rasr_request import RasrRequest
from huaweicloud_sis.bean.callback import RasrCallBack
from huaweicloud_sis.bean.sis_config import SisConfig 
from aip import AipSpeech
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import soundfile
import torchaudio
import pdb
import unittest
from hsfpy import *
from zhon.hanzi import punctuation
from espnet2.attack.tools_attack import CER, edit_distance, numpy_SNR, SCORE, upload_oss_file
from espnet2.attack.tools_attack import Speech2Text
class LocalAsr():
    def __init__(self):
        asr_train_config  = 'exp/asr_train_asr_transformer2_raw_zh_char_batch_bins20000000_ctc_confignore_nan_gradtrue_sp/config.yaml'
        asr_model_file = 'exp/asr_train_asr_transformer2_raw_zh_char_batch_bins20000000_ctc_confignore_nan_gradtrue_sp/valid.acc.ave_10best.pth'
        lm_train_config = 'exp/lm_train_lm_transformer_zh_char_optim_conflr0.001_scheduler_confwarmup_steps25000_batch_bins3000000_accum_grad1/config.yaml'
        lm_file = 'exp/lm_train_lm_transformer_zh_char_optim_conflr0.001_scheduler_confwarmup_steps25000_batch_bins3000000_accum_grad1/valid.loss.ave_10best.pth'
        self.speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        device='cuda',
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=20,
        ctc_weight=0.3,
        lm_weight=0.5,
        penalty=0.0,
        nbest=1
    )
    def forward(self, args):
        score_function = SCORE()
        for a_type, sub_type in args.attack_type.items():
            for sub_a_type in sub_type:
                data_dir = os.path.join('generated', a_type, sub_a_type)
                text_dir = os.path.join(data_dir, 'text_target.txt')
                pre_text_list = []
                with open(text_dir, 'r', encoding="utf-8") as file_to_read:
                    while True:
                        lines = file_to_read.readline()
                        if not lines:
                            break
                        p_tmp = lines.split()[0]
                        wav_name = p_tmp.split('/')[-1].split('.')[0]
                        audio_dir = os.path.join(data_dir, wav_name + '.wav')
                        speech, rate = torchaudio.load(audio_dir)
                        nbests = self.speech2text(speech)
                        pre_text, *_ = nbests[0]
                        pre_text_list.append(pre_text)
                if sub_a_type == 'insert':
                    args = score_function.score_insert(args, data_dir, pre_text_list)
                if sub_a_type == 'delet':
                    args = score_function.score_delet(args, data_dir, pre_text_list)
                if sub_a_type  == 'sub':
                    args = score_function.score_sub(args, data_dir, pre_text_list)
                if a_type == 'general':
                    args = score_function.score_general(args, data_dir, pre_text_list, sub_a_type)
        return args
