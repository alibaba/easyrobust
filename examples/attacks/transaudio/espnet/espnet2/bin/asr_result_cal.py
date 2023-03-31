# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import os
import sys
import os
sys.path.append('../../../')
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import json
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
import numpy as np
import torch
import soundfile
import torchaudio
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List
from espnet2.tasks.asr import ASRTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils import config_argparse
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.utils.types import str2bool
from espnet2.torch_utils.device_funcs import to_device
from espnet.utils.cli_utils import get_commandline_args
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

from collections import Counter

def inference(
    args,
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    streaming: bool,
    results: dict,
    attack_type: dict,
    target_api: dict,
    nj : dict,
):
    output_dir = output_dir.split('/')[:-1]
    output_dir = '/'.join(output_dir) + '/output.' 
    if 'dev' in output_dir:
        return args.results
    for i in range(int(nj)):    
        save_path_results = os.path.join(output_dir + str(i+1), 'results.json')
        with open(save_path_results, 'r') as fh:
            data = json.load(fh)
            for t1 in ['clean', 'white', 'query', 'general']:
                if t1 == 'clean':
                    print('8888check')
                    for t2 in ['cer', 'med']:
                        args.results[t1][t2] += data[t1][t2]
                if t1 == 'white':
                    for t2 in ['fgsm', 'mfgsm', 'pgd', 'spa_fgsm', 'spa_mfgsm', 'spa_pgd', 'commandsong']:
                        for t3 in ['cer', 'sr', 'times', 'snr']:
                            args.results[t1][t2][t3] += data[t1][t2][t3]
                if t1 == 'query':
                    for t2 in ['sr', 'times', 'cer', 'snr', 'med']:
                        args.results[t1][t2] += data[t1][t2]
                if t1 == 'transfer':
                    for t2 in ['insert', 'delet', 'sub']:
                        for t3 in ['cer', 'sr', 'med']:
                            args.results[t1][t2][t3] += data[t1][t2][t3]
                if t1 == 'general':
                    for t2 in ['speed', 'tdi', 'gaussian', 'kitchen', 'field', 'traffic', 'tmetro', 'style']:
                        for t3 in ['cer', 'med']:
                            args.results[t1][t2][t3] += data[t1][t2][t3]
            
    for t1 in ['clean', 'white', 'query', 'general']:
                if t1 == 'clean':
                    for t2 in ['cer', 'med']:
                        args.results[t1][t2] /= 100.0
                if t1 == 'white':
                    for t2 in ['fgsm', 'mfgsm', 'pgd', 'spa_fgsm', 'spa_mfgsm', 'spa_pgd', 'commandsong']:
                        for t3 in ['cer', 'sr', 'times', 'snr']:
                            args.results[t1][t2][t3] /= 100.0
                if t1 == 'query':
                    for t2 in ['sr', 'times', 'cer', 'snr', 'med']:
                        args.results[t1][t2] /= 100.0
                if t1 == 'transfer':
                    for t2 in ['insert', 'delet', 'sub']:
                        for t3 in ['cer', 'sr', 'med']:
                            args.results[t1][t2][t3] /= 100.0
                if t1 == 'general':
                    for t2 in ['speed', 'tdi', 'gaussian', 'kitchen', 'field', 'traffic', 'tmetro', 'style']:
                        for t3 in ['cer', 'med']:
                            args.results[t1][t2][t3] /= 100.0 
    
    ave_path_results = os.path.join('generated', 'results.json')
    with open(save_path_results, 'w') as fh:
        json.dump(args.results, fh)
    return args    
    

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument("--lm_train_config", type=str)
    group.add_argument("--lm_file", type=str)
    group.add_argument("--word_lm_train_config", type=str)
    group.add_argument("--word_lm_file", type=str)
    group.add_argument("--ngram_file", type=str)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    parser.add_argument(
        "--nj",
        default=0,
        help="The number of output",
    )


    return parser

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    args.results = {'clean':{'cer':0, 'med':0}, 'white':{'fgsm':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'mfgsm':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'pgd':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'spa_fgsm':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'spa_mfgsm':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'spa_pgd':{'cer':0, 'sr':0, 'times':0, 'snr':0}, 'commandsong':{'cer':0, 'sr':0, 'times':0, 'snr':0} }, 'query':{'sr':0, 'times':0, 'cer':0, 'snr':0, 'med':0}, 'transfer':{'insert':{'cer':0, 'med':0, 'sr':0}, 'delet':{'cer':0, 'med':0, 'sr':0}, 'sub':{'cer':0, 'med':0, 'sr':0}}, 'general':{'speed':{'cer':0, 'med':0}, 'tdi':{'cer':0, 'med':0}, 'gaussian':{'cer':0, 'med':0}, 'kitchen':{'cer':0, 'med':0}, 'field':{'cer':0, 'med':0}, 'traffic':{'cer':0, 'med':0}, 'tmetro':{'cer':0, 'med':0}, 'style':{'cer':0, 'med':0}}}
            
    #args.attack_type = {'clean':1, 'white':['fgsm','mfgsm','pgd','spa_fgsm','spa_mfgsm','spa_pgd','commandsong'], 'query':0}
    #args.attack_type = {'transfer':['delet']}
    args.attack_type = {'clean':1}
    #args.attack_type = {'clean':1, 'general': ['speed', 'tdi', 'gaussian', 'kitchen', 'field', 'straffic', 'tmetro', 'beijing']}
    #args.attack_type = {'general':['traffic']}
    #args.target_api = {'inside':['zhibo']}
    args.target_api = {'outside':['aliyun']}
    
    kwargs = vars(args)
    kwargs.pop("config", None)
    args = inference(args, **kwargs)

if __name__ == "__main__":
    main()



