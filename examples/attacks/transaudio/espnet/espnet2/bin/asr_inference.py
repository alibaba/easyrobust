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


from esp_zoo.espnet_model_zoo.espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

#Attack Set
from espnet2.attack.black_attack import TransferInsert, TransferDelet, TransferSub
from espnet2.API_asr.api_asr import LocalAsr


def eval_clean(args, batch, speech2text):
    if args.ngpu >= 1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    kwargs = vars(args)
    kwargs.pop("config", None)
    clean = Clean(**kwargs)
    args = clean(args, batch, speech2text)
    return args
def eval_white(args, batch, speech2text):
    if args.ngpu >= 1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    kwargs = vars(args)
    kwargs.pop("config", None)
    whiteattack = WhiteAttack(**kwargs)
    args = whiteattack(args, batch, speech2text)
    return args
def eval_query(args, batch):
    if args.ngpu >= 1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    kwargs = vars(args)
    kwargs.pop("config", None)
    speech2text = Speech2Text_query(**kwargs)
    queryattack = Genetic(speech2text)
    args = queryattack.attack(args, batch)
    return args
def eval_transfer(args, batch, speech2text, key):
    if args.ngpu >= 1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    args.epsilon = 0.0002
    args.alpha = 0.08
    args.iteration = 100
    kwargs = vars(args)
    kwargs.pop("config", None)

    for a_type in args.attack_type['transfer']:
        save_path = os.path.join('generated', 'transfer', a_type)
        os.makedirs(save_path, exist_ok=True)
        args.epsilon = 0.0002
        args.alpha = 0.08
        args.iteration = 100
        if 'insert' in a_type:
            insert_attack = TransferInsert(**kwargs)
            adv, target_text  = insert_attack(args, batch, speech2text)
            adv_save_path = os.path.join(save_path, key + '.wav') 
            torchaudio.save(adv_save_path, adv, 16000)
            filename = os.path.join(save_path, 'text_target.txt') 
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(key + ' ' + target_text +'\n')
        if 'delet' in a_type:
            save_path = os.path.join('generated', 'transfer', a_type)
            os.makedirs(save_path, exist_ok=True)
            delet_attack_1 = TransferDelet(**kwargs)
            args.asr_train_config = 'exp/exp/exp/asr_train_asr_streaming_transformer_raw_zh_char_sp/config.yaml'
            args.asr_model_file = 'exp/exp/exp/asr_train_asr_streaming_transformer_raw_zh_char_sp/valid.acc.ave_10best.pth'
            args.lm_train_config = 'exp/exp/exp/lm_train_lm_zh_char/config.yaml'
            args.lm_file = 'exp/exp/exp/lm_train_lm_zh_char/20epoch.pth'
            kwargs = vars(args)
            kwargs.pop("config", None)
            delet_attack_2 = TransferDelet(**kwargs)
            adv, target_text, del_text  = delet_attack(args, batch, speech2text, delet_attack_1, delet_attack_2)
            adv_save_path = os.path.join(save_path, key + '.wav')
            torchaudio.save(adv_save_path, adv, 16000)
            filename = os.path.join(save_path, 'text_target.txt')
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(key + ' ' + target_text + ' ' + del_text + '\n')
        if 'sub' in a_type:
            save_path = os.path.join('generated', 'transfer', a_type)
            os.makedirs(save_path, exist_ok=True)
            sub_attack_1 = TransferSub(**kwargs)
            args.asr_train_config = 'exp/exp/exp/asr_train_asr_streaming_transformer_raw_zh_char_sp/config.yaml'
            args.asr_model_file = 'exp/exp/exp/asr_train_asr_streaming_transformer_raw_zh_char_sp/valid.acc.ave_10best.pth'
            args.lm_train_config = 'exp/exp/exp/lm_train_lm_zh_char/config.yaml'
            args.lm_file = 'exp/exp/exp/lm_train_lm_zh_char/20epoch.pth'
            kwargs = vars(args)
            kwargs.pop("config", None)
            sub_attack_2 = TransferSub(**kwargs)
            args.asr_train_config = 'exp/asr_train_asr_transformer2_raw_zh_char_batch_bins20000000_ctc_confignore_nan_gradtrue_sp/config.yaml'
            args.asr_model_file = 'exp/asr_train_asr_transformer2_raw_zh_char_batch_bins20000000_ctc_confignore_nan_gradtrue_sp/valid.acc.ave_10best.pth'
            args.lm_train_config = 'exp/lm_train_lm_transformer_zh_char_optim_conflr0.001_scheduler_confwarmup_steps25000_batch_bins3000000_accum_grad1/config.yaml'
            args.lm_file = 'exp/lm_train_lm_transformer_zh_char_optim_conflr0.001_scheduler_confwarmup_steps25000_batch_bins3000000_accum_grad1/valid.loss.ave_10best.pth'
            kwargs = vars(args)
            kwargs.pop("config", None)
            sub_attack_3 = TransferSub(**kwargs)
            adv, target_text, sub_text  = sub_attack(args, batch, sub_attack_1, speech2text, sub_attack_2, sub_attack_3)
            adv_save_path = os.path.join(save_path, key + '.wav')
            torchaudio.save(adv_save_path, adv, 16000)
            filename = os.path.join(save_path, 'text_target.txt')
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(key + ' ' + target_text  + ' ' + sub_text + '\n')
    return args

def eval_general(args, batch, key):
    save_path = os.path.join('generated', 'general')
    kwargs = vars(args)
    kwargs.pop("config", None)
    gau_attack = GeneralAttack(**kwargs)
    for a_type in args.attack_type['general']:
        if a_type == 'gaussian':
            adv, text = gau_attack.gaussian(batch)
        if a_type == 'speed':
            adv, text = gau_attack.speed(batch)
        if a_type == 'tdi':
            adv, text = gau_attack.tdi(batch)
        if a_type == 'kitchen':
            adv, text = gau_attack.demand(batch, 'DKITCHEN', 0.5, 0.5)
        if a_type == 'field':
            adv, text = gau_attack.demand(batch, 'NFIELD', 0.5, 0.5)
        if a_type == 'traffic':
            adv, text = gau_attack.demand(batch, 'STRAFFIC', 0.75, 0.15)
        if a_type == 'metro':
            adv, text = gau_attack.demand(batch, 'TMETRO', 0.75, 0.15)
        if a_type == 'style':
            adv, text = gau_attack.style(batch)
        save_path_ = os.path.join(save_path, a_type)
        os.makedirs(save_path_, exist_ok=True)
        adv_save_path = os.path.join(save_path_, key + '.wav')
        torchaudio.save(adv_save_path, torch.tensor(adv).unsqueeze(0), 16000)
        filename = os.path.join(save_path_, 'text_target.txt')
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(key + ' ' + text  + '\n')
    return args


def eval_score(args):
    for target_API in  args.target_api['outside']:
        API = LocalAsr()
        args = API.forward(args)
    return args

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
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"
    # 1. Set random-seed & Lp & iters
    set_all_random_seed(seed)
    attack_type = "fgsm"
    epsilon = 0.06
    alpha = 0.002
    iteration = 50
    speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
    )
    # 2. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )
    keys = []
    text_trgeted = []
    with DatadirWriter(output_dir) as writer:
        for  batch in loader:
            try:
                batch = to_device(batch, device=device)
                for key,value in  args.attack_type.items():
                    if value:
                        if 'transfer' in key:
                            key_i = batch[0][0]
                            args = eval_transfer(args, batch, speech2text, key_i)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest
        if 'general' in args.attack_type:
            args = eval_score(args)
        save_path_results = os.path.join(output_dir, 'results.json')
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

    return parser

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)

    args.results = {'transfer':{'insert':{'cer':0, 'med':0, 'sr':0}, 'delet':{'cer':0, 'med':0, 'sr':0}, 'sub':{'cer':0, 'med':0, 'sr':0}}}
    args.attack_type = {'transfer':['sub']}
    kwargs = vars(args)
    kwargs.pop("config", None)
    args = inference(args, **kwargs)

if __name__ == "__main__":
    main()



