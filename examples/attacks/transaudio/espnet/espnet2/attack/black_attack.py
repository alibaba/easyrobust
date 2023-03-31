#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ngram import NgramFullScorer
from espnet.nets.scorers.ngram import NgramPartScorer
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.attack.style_trans.test import CycleGANTraining
from esp_zoo.espnet_model_zoo.espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from torch.autograd import Variable

import os
import soundfile
import torchaudio
from scipy.signal import butter, lfilter
import scipy.stats as st
import jieba
import jieba.posseg as pseg
import librosa
import random
from snownlp import sentiment
from snownlp import SnowNLP
import cv2 as cv
from espnet2.attack.tools_attack import TDIAttack, awgn
token_list = "./data/zh_token_list/char/tokens.txt"
with open(token_list, encoding="utf-8") as f:
    token_list = [line.rstrip() for line in f]
    # Overwriting token_list to keep it as "portable".
    token_list = list(token_list)
tokenizer = build_tokenizer(
                token_type="char",
            )
token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol="<unk>",
            )
import scipy.signal as sig
def highpass_filter_1(wav_data, sample_rate, crit_freq = 1000 ):

    N = 10                            # order of the filter
    nyq_freq = float(sample_rate)/2.  # nyquist frequency
    crit_freq_norm = crit_freq/nyq_freq    # critical frequency (normalized)
    rp = 1                            # passband ripple (dB)
    rs = 80                           # stopband min attenuation (dB)
    btype = 'highpass'
    ftype = 'ellip'
    b, a = sig.iirfilter(N, crit_freq_norm, rp, rs, btype, ftype)

    # apply filter and return filtered data
    return  sig.lfilter(b, a, wav_data)

class TransferInsert:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        iteration: int = 1000,
        log_level: Union[int, str] = None,
        output_dir: str = None,
        ngpu: int = None,
        seed: int = None,
        num_workers: int = None,
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        results: dict = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        word_lm_file: Optional[str] = None,
        allow_variable_data_keys: bool = None,
        attack_type: dict = None,
        target_api: dict = None,
    ):
        assert check_argument_types()

        # 1. Build ASR model

        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        # TODO(karita): make all scorers batchfied

        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(asr_train_config)
                    logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                else:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")
        
        self.iner_attack_type = 'mfgsm'
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.attack_type = attack_type
        self.epsilon = 0.0002
        self.alpha = 0.08
        self.iteration = 100
        self.momentum = 0.5
        self.grad = 0
        self.flag = 0

    def fgsm_attack(self, sound, epsilon, data_grad):

        # find direction of gradient
        #import scipy
        #data_grad = scipy.signal.savgol_filter(data_grad,80,10)
        sign_data_grad = data_grad.sign()

        # add noise "epilon * direction" to the ori sound
        tmp = epsilon * sign_data_grad
        perturbed_sound = sound + tmp

        return perturbed_sound
    def mi_fgsm_attack(self, sound, epsilon, data_grad):
        if self.flag == 0:
            self.grad = data_grad
            self.flag = 1
        self.grad = self.momentum * self.grad +  data_grad / torch.sum(torch.abs(data_grad))
        tmp = epsilon * self.grad.sign()
        perturbed_sound = sound - tmp
        return perturbed_sound
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :

        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound
    #@torch.no_grad()
    def __call__(
            self, args, batch, speech2text) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        #d = ModelDownloader()
        #text2speech = Text2Speech(**d.download_and_unpack("./exp/tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best.zip"))
        tts_path = './exp/tts/exp/tts_train_tacotron2_raw_phn_pypinyin_g2p_phone/'
        text2speech = Text2Speech(tts_path + "config.yaml", tts_path + "199epoch.pth")
        batch = batch[1]
        target_text_pre = batch["text"][0]

        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)

        seg_list = pseg.cut(target_text_pre, use_paddle=True)
        taget_text_remade = []
        bian = 0
        position = 0
        p_flag = 0
        for word, flag in seg_list:
            if flag == 'd' and not bian:
                word_ = '不'
                word = word_ + word
                bian = 1
                p_flag = 1

            if flag == 'v' and not bian:
                word_ = '不'
                word = word_ + word
                bian = 1
                p_flag = 1
            if flag == 'a' and not bian:
                word_ = '不'
                word = word_ + word
                bian = 1
                p_flag = 1
            if bian == 0 :
                position += len(word)
            taget_text_remade.append(word)
        if bian == 0:
            word_ = '不'
            bian = 1
            p_flag = 1
            taget_text_remade.insert(0,word_)
            position = 0
        target_text_remade = "".join(taget_text_remade)
        speech_tts = text2speech(word_)[0]
        tmp = 1.0 * (position+1) / len(target_text_pre)
        position_s = int(len(batch["speech"][0])*tmp)
        speech_tts = torch.tensor(highpass_filter_1(speech_tts, 16000, 1000)).to(torch.float32)
        speech_len = torch.tensor(speech_tts.shape)
        tokens = tokenizer.text2tokens(word_)
        text_ints = token_id_converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([p_flag])
        adv_1 = speech_tts.clone().detach().requires_grad_(True).to(self.device).unsqueeze(0)

        self.grad = 0
        i = 0
        p = 0.5
        #adv = adv_1 + 0.0001 * torch.randn(adv_1.shape).requires_grad_(True).to(self.device)
        adv = adv_1
        adv = adv[:,:4877]

        speech_len = torch.tensor([4877])
        import julius
        for i in range(self.iteration):
            _adv = adv.clone().detach().requires_grad_(True)
            t = torch.rand(())
            retval = self.asr_model(julius.highpass_filter(_adv, 0.1).to(torch.float32).cuda() if 0 else _adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)

            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss, stats, weight = retval
            self.asr_model.zero_grad()
            stats = {k: v for k, v in stats.items() if v is not None}
            loss = 0.1 * loss + 10*stats['cer']  + 10*stats['cer_ctc']
            loss.backward()
            data_grad = _adv.grad.data

            #adv_= torch.tensor(highpass_filter_1(_adv[0].cpu().clone().detach().numpy(), 16000, 50)).to(torch.float32)
            #t = torch.rand(())
            #if t<0.3:
                #_adv =  0.9 * _adv.detach() + 0.1 * adv_.unsqueeze(0).cuda()
            if self.iner_attack_type == "mfgsm":
                self.epsilon = self.alpha/self.iteration
                adv = self.mi_fgsm_attack(_adv, self.epsilon, data_grad)
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text
        #adv = adv[0][:4877]
        adv = adv[0]
        hanming = torch.tensor(np.hamming(4877)).to(torch.float32).cuda()
        #hanming = torch.tensor(np.kaiser(4877,8)).to(torch.float32).cuda()
        adv = adv.mul(hanming)
        speech_input = torch.cat((batch["speech"][0][:position_s-500].cuda(), adv.cuda(),  batch["speech"][0][position_s-500:].cuda()),0)
        speech_input = speech_input.unsqueeze(0)
        ################################################################
        self.grad = 0
        adv = speech_input.clone().detach().requires_grad_(True).to(self.device).cuda()
        speech_len = torch.tensor([adv.shape[1]])
        tokens = tokenizer.text2tokens(target_text_remade)
        text_ints = token_id_converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        print(target_text_remade)
        self.iteration = 2
        for i in range(self.iteration//50):
            _adv = adv.clone().detach().cuda().requires_grad_(True)
            t = torch.rand(())
            retval = self.asr_model(julius.highpass_filter(_adv, 0.1).to(torch.float32).cuda() if 0 else _adv, speech_len.cuda(), text_input.cuda(), text_len.cuda())
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss, stats, weight = retval
            self.asr_model.zero_grad()
            stats = {k: v for k, v in stats.items() if v is not None}

            loss = 0.1 * loss + 10*stats['cer'] + stats['wer'] + 10*stats['cer_ctc']
            loss.backward(retain_graph=True)
            data_grad = _adv.grad.data
            if self.iner_attack_type == "mfgsm":
                self.epsilon =  self.alpha/self.iteration
                adv = self.mi_fgsm_attack(_adv, self.epsilon, data_grad)
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)
            adv = adv.clone().detach().requires_grad_(True)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text

                print(target_text_remade)
                print(best_text)
        speech_input = adv.detach().cpu()

        return speech_input, target_text_remade

class TransferDelet:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        iteration: int = 1000,
        log_level: Union[int, str] = None,
        output_dir: str = None,
        ngpu: int = None,
        seed: int = None,
        num_workers: int = None,
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        results: dict = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        word_lm_file: Optional[str] = None,
        allow_variable_data_keys: bool = None,
        attack_type: dict = None,
        target_api: dict = None,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram
        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(asr_train_config)
                    logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                else:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()

        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.iteration = iteration
        self.momentum = 0.5
        self.grad = 0
        self.flag = 0
        self.token_list = token_list
        self.token_type = token_type
    def gkern(self, kernlen = 3, nsig = 1):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(-2,x)
        kern2d = st.norm.pdf(-2,x)
        kernel_raw = np.outer(kern1d, kern2d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    def blur(self, tensor_image, epsilon, stack_kerne):
        min_batch=tensor_image.shape[0]
        channels=tensor_image.shape[1]
        out_channel=channels
        kernel=torch.FloatTensor(stack_kerne).cuda()
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        x = torch.ones([1, 3, tensor_image.shape[1], tensor_image.shape[2]])
        x[:,:2,:,:] = tensor_image
        data_grad = torch.nn.functional.conv2d(x.cuda(),weight.cuda(),bias=None,stride=1,padding=(1,0), dilation=1)

        return data_grad[0,:2,:,:]

    def fgsm_attack(self, sound, epsilon, data_grad):

        # find direction of gradient
        sign_data_grad = data_grad.sign()

        # add noise "epilon * direction" to the ori sound
        tmp = epsilon * sign_data_grad
        perturbed_sound = sound + tmp

        return perturbed_sound
    def mi_fgsm_attack(self, sound, epsilon, data_grad):
        if self.flag == 0:
            self.grad = data_grad
            self.flag = 1
        self.grad = self.momentum * self.grad +  data_grad / torch.sum(torch.abs(data_grad))
        tmp = epsilon * self.grad.sign()
        perturbed_sound = sound - tmp.cuda()
        return perturbed_sound
    #@torch.no_grad()
    def __call__(
            self, batch, speech2text, local_flag, target_text_pre_input) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        
        tts_path = './exp/tts/exp/tts_train_tacotron2_raw_phn_pypinyin_g2p_phone/'
        text2speech = Text2Speech(tts_path + "config.yaml", tts_path + "199epoch.pth")
        
        batch_o = batch
        batch = batch[1]
        t = random.randint(30,40)
        if local_flag:
            target_text_pre = target_text_pre_input
        else:
            target_text_pre = batch["text"][0]
            token = self.converter.ids2tokens(target_text_pre)
            target_text_pre = self.tokenizer.tokens2text(token)

        seg_list = pseg.cut(target_text_pre, use_paddle=True)
        taget_text_remade = []
        bian = 0
        position = 0
        p_flag = 0
        del_word = None
        ########################################## Del
        for word, flag in seg_list:
            if flag == 'a' and len(word) < 3 and not bian:
                del_word = word
                bian = 1
                p_flag = len(word)
                continue
            if flag == 'n' and len(word) < 3  and not bian:
                del_word = word
                bian = 1
                p_flag = len(word)
                continue
            if bian == 0 :
                position += len(word)
            taget_text_remade.append(word)
        if bian == 0:
            del_word = target_text_pre[-1]
            bian = 1
            p_flag = 1
            position -= 1
            taget_text_remade.pop()
        target_text_remade = "".join(taget_text_remade)
        print(target_text_remade,'target_text_remade')
        ######### mid_target
        tmp = 1.0 * (position+1) / len(target_text_pre)
        tmp_2 = 1.0 * (position+1+p_flag) / len(target_text_pre)
        position_s = int(len(batch["speech"][0])*tmp)
        self.grad = 0
        
        if not local_flag:

            local_speech = batch["speech"][0][position_s-1000 : position_s-1000 + 5000 * p_flag]
            speech_len = torch.tensor([local_speech.shape[0]])
            tokens = tokenizer.text2tokens('。。')
            text_ints = token_id_converter.tokens2ids(tokens)
            text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
            text_len = torch.tensor([text_input.shape[1]])
            adv_1 = local_speech.clone().detach().requires_grad_(True).to(self.device).unsqueeze(0)
            self.grad = 0
            i = 0
            p = 0.5
            adv = adv_1
        gaijin = 0
        for i in range(self.iteration//2):
            if local_flag:
                break
            _adv = adv.clone().detach().requires_grad_(True)
            retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss, stats, weight = retval
            self.asr_model.zero_grad()
            stats = {k: v for k, v in stats.items() if v is not None}
            #loss = 0.1 * loss + 10*stats['cer']  + 10*stats['cer_ctc']
            loss.backward()
            data_grad = _adv.grad.data
            len_tmp = data_grad.shape[1]
            magnitude = torch.stft(data_grad[0].cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320)).permute(2, 1, 0)
            kernel = self.gkern(3, 1).astype(np.float32)
            stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            stack_kernel = np.expand_dims(stack_kernel, 3)
            dst = self.blur(magnitude, 1, stack_kernel)

            dst = torch.tensor(dst).permute(2,1,0)
            dst = torch.istft(dst.cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320), length = len_tmp)
            t = torch.rand(())
            if 0 :
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)
            if 1:
                self.epsilon = self.alpha/self.iteration
                adv = self.mi_fgsm_attack(_adv, self.epsilon, data_grad)
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text
        if local_flag:
            speech_input = batch['speech']
        else:
            adv_ = adv[0]
            hanming = torch.tensor(np.hamming(adv_.shape[0])).to(torch.float32).cuda()
            adv_ = adv_.mul(hanming).clone().detach().requires_grad_(True).to(self.device)
            adv = batch["speech"][0].cuda()

            adv_left = adv[:position_s-1000]
            hanming = torch.tensor(np.hamming(adv_left.shape[0])).to(torch.float32).cuda()
            adv_left = adv_left.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv_right = adv[position_s-1000 + adv_.shape[0]:]
            hanming = torch.tensor(np.hamming(adv_right.shape[0])).to(torch.float32).cuda()
            adv_right = adv_right.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv = batch["speech"][0].cuda()
            adv[:position_s-1000] = adv_left
            adv[position_s-1000 + adv_.shape[0]:] = adv_right
            adv[position_s-1000 : position_s-1000 + adv_.shape[0]] =  adv_.cuda()
            speech_input = adv.unsqueeze(0)
        ################################################################
        self.grad = 0
        adv = speech_input.clone().detach().requires_grad_(True).to(self.device).cuda()
        speech_len = torch.tensor([adv.shape[1]])
        tokens = tokenizer.text2tokens(target_text_remade)
        text_ints = token_id_converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        print(target_text_remade, '------------target_text_remade--------')
        for i in range(2):
            _adv = adv.clone().detach().cuda().requires_grad_(True)
            t = torch.rand(())
            retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss, stats, weight = retval
            self.asr_model.zero_grad()
            stats = {k: v for k, v in stats.items() if v is not None}

            #loss = 0.1 * loss + 10*stats['cer'] + stats['wer'] + 10*stats['cer_ctc']
            loss.backward(retain_graph=True)
            data_grad = _adv.grad.data
            len_tmp = data_grad.shape[1]
            magnitude = torch.stft(data_grad[0].cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320)).permute(2, 1, 0)
            kernel = self.gkern(3, 1).astype(np.float32)
            stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            stack_kernel = np.expand_dims(stack_kernel, 3)
            dst = self.blur(magnitude, 1, stack_kernel)

            dst = torch.tensor(dst).permute(2,1,0)
            dst = torch.istft(dst.cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320), length = len_tmp)
            t = torch.rand(())
            if t<0.5 :
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)


            print('grad', data_grad)
            if 1: 
                self.epsilon =  self.alpha/self.iteration
                adv = self.mi_fgsm_attack(_adv.cuda(), self.epsilon, data_grad.cuda())
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)

            adv = adv.clone().detach().requires_grad_(True)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text
        speech_input = adv.cuda()
        batch_o[1]["speech"] = speech_input
        return  batch_o, target_text_pre, target_text_remade, del_word
class TransferSub:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        iteration: int = 1000,
        log_level: Union[int, str] = None,
        output_dir: str = None,
        ngpu: int = None,
        seed: int = None,
        num_workers: int = None,
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        results: dict = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        word_lm_file: Optional[str] = None,
        allow_variable_data_keys: bool = None,
        attack_type: dict = None,
        target_api: dict = None,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )
        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(asr_train_config)
                    logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                else:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.iteration = iteration
        self.momentum = 0.9
        self.grad = 0
        self.flag = 0
        self.token_list = token_list
        self.token_type = token_type
    def gkern(self, kernlen = 3, nsig = 1):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(-2,x)
        kern2d = st.norm.pdf(-2,x)
        kernel_raw = np.outer(kern1d, kern2d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def blur(self, tensor_image, epsilon, stack_kerne):
        min_batch=tensor_image.shape[0]
        channels=tensor_image.shape[1]
        out_channel=channels
        kernel=torch.FloatTensor(stack_kerne).cuda()
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        x = torch.ones([1, 3, tensor_image.shape[1], tensor_image.shape[2]])
        x[:,:2,:,:] = tensor_image
        data_grad = torch.nn.functional.conv2d(x.cuda(),weight.cuda(),bias=None,stride=1,padding=(1,0), dilation=1)

        return data_grad[0,:2,:,:]
    def fgsm_attack(self, sound, epsilon, data_grad):

        # find direction of gradient
        #import scipy
        #data_grad = scipy.signal.savgol_filter(data_grad,80,10)
        sign_data_grad = data_grad.sign()

        # add noise "epilon * direction" to the ori sound
        tmp = epsilon * sign_data_grad
        perturbed_sound = sound + tmp

        return perturbed_sound
    def mi_fgsm_attack(self, sound, epsilon, data_grad):
        if self.flag == 0:
            self.grad = data_grad
            self.flag = 1
        self.grad = self.momentum * self.grad +  data_grad / torch.sum(torch.abs(data_grad))
        tmp = epsilon * self.grad.sign()
        perturbed_sound = sound - tmp.cuda()
        return perturbed_sound
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :

        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound
    #@torch.no_grad()
    def __call__(
            self,  batch, speech2text, speech2text_2, speech2text_3) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        tts_path = './exp/tts/exp/tts_train_tacotron2_raw_phn_pypinyin_g2p_phone/'
        text2speech = Text2Speech(tts_path + "config.yaml", tts_path + "199epoch.pth")
        batch_o = batch
        batch = batch[1]
        t = random.randint(30,40)
        


        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)
        seg_list = pseg.cut(target_text_pre, use_paddle=True)
        print(seg_list)
        taget_text_remade = []
        bian = 0
        position = 0
        p_flag = 0
        sub = '张坤'
        for word, flag in seg_list:
            if len(word) == 2  and not bian:
                word = sub
                bian = 1
                p_flag = 2
            if bian == 0 :
                position += len(word)
            taget_text_remade.append(word)
        if bian == 0:
            word = sub
            bian = 1
            p_flag = 2
            taget_text_remade.append(word)
        target_text_remade = "".join(taget_text_remade)
        speech_tts = text2speech(sub)[0]
        #speech_tts = librosa.resample(speech_tts.numpy(), 24000, 16000)
        #soundfile.write('./tmp.wav', speech_tts, 16000)
        #speech_tts, _ = soundfile.read('./tmp.wav', 16000)
        #speech_tts = librosa.resample(speech_tts.numpy(), 28000, 16000)
        tmp = 1.0 * (position+1) / len(target_text_pre)
        position_s = int(len(batch["speech"][0])*tmp)
        speech_tts = highpass_filter_1(speech_tts, 16000, 100)
        #speech_tts = torch.tensor(lowpass_filter_1(speech_tts, 16000, 50)).to(torch.float32)
        speech_tts = torch.tensor(speech_tts[:14400]).to(torch.float32)

        if 1:

            #tmp_local = 0.5 * batch["speech"][0][position_s-1000 : position_s-1000 + 10000] + 0.5*speech_tts
            tmp_local = speech_tts
            #hanming = torch.tensor(np.hamming(tmp_local.shape[0])).to(torch.float32)
            #tmp_local = tmp_local.mul(hanming).clone().detach().requires_grad_(True).to(self.device)
            tmp_local = torch.tensor(tmp_local).clone().detach().requires_grad_(True).to(self.device)
            local_speech = tmp_local
        else:
            local_speech = batch["speech"][0][position_s-1000 : position_s-1000 + 4877*2].cuda()
        speech_len = torch.tensor(local_speech.shape)

        tokens = tokenizer.text2tokens(sub)
        text_ints = token_id_converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([p_flag])
        adv_1 = local_speech.clone().detach().requires_grad_(True).to(self.device).unsqueeze(0)

        self.grad = 0
        i = 0
        p = 0.5
        adv = adv_1

        for i in range(self.iteration//2):
            print('-----------------i-------------', i)
            _adv = adv.clone().detach().requires_grad_(True)
            retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss_1 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_1, stats, weight = retval
            self.asr_model.zero_grad()
            ##########
            retval = speech2text_2.asr_model(_adv, speech_len, text_input, text_len, 0, None, 0)
            if isinstance(retval, dict):
                loss_2 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_2, stats, weight = retval
            speech2text_2.asr_model.zero_grad()

            ##########
            retval = speech2text_3.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss_3 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_3, stats, weight = retval
            speech2text_3.asr_model.zero_grad()

            loss = loss_1 #+ loss_2 + loss_3
            loss.backward()
            data_grad = _adv.grad.data
            print(data_grad,'data_grad')
            len_tmp = data_grad.shape[1]
            magnitude = torch.stft(data_grad[0].cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320)).permute(2, 1, 0)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 80))
            dst = cv.dilate(magnitude.detach().cpu().numpy(), kernel)
            dst = torch.tensor(dst).permute(2,1,0)
            dst = torch.istft(dst, n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320), length = len_tmp)
            t = torch.rand(())
            if 0:
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)
            if 1: 
                self.epsilon = self.alpha/self.iteration * 5
                adv = self.mi_fgsm_attack(_adv, self.epsilon, data_grad)
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text
        adv_ = adv[0]
        if 0:
            speech_input = batch['speech']
        else:
            adv = batch["speech"][0].cuda()
            hanming = torch.tensor(np.hamming(adv_.shape[0])).to(torch.float32).cuda()
            adv_ = adv_.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv_left = adv[:position_s-1000]
            hanming = torch.tensor(np.bartlett(adv_left.shape[0])).to(torch.float32).cuda()
            adv_left = adv_left.mul(hanming).clone().detach()

            adv_right = adv[position_s-1000 + adv_.shape[0]:]
            hanming = torch.tensor(np.hamming(adv_right.shape[0])).to(torch.float32).cuda()
            adv_right = adv_right.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv = batch["speech"][0].cuda()
            adv[:position_s-1000].data = adv_left
            adv[position_s-1000 + adv_.shape[0]:].data = adv_right.cuda()
            adv[position_s-1000 : position_s-1000 + adv_.shape[0]].data =  adv_.cuda()
            speech_input = adv.unsqueeze(0)

        ################################################################
        self.grad = 0
        adv = speech_input.clone().detach().requires_grad_(True).to(self.device).cuda()
        speech_len = torch.tensor([adv.shape[1]])
        tokens = tokenizer.text2tokens(target_text_remade)
        text_ints = token_id_converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        print(target_text_remade, '------------target_text_remade--------')
        for i in range(self.iteration//4):
            _adv = adv.clone().detach().cuda().requires_grad_(True)
            t = torch.rand(())
            retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss_1 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_1, stats, weight = retval
            self.asr_model.zero_grad()

            retval = speech2text_2.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss_2 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_2, stats, weight = retval
            speech2text_2.asr_model.zero_grad()

            retval = speech2text_3.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, None, 0)
            if isinstance(retval, dict):
                loss_3 = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss_3, stats, weight = retval
            speech2text_3.asr_model.zero_grad()

            loss = loss_1 + loss_2 + loss_3

            loss.backward(retain_graph=True)
            data_grad = _adv.grad.data

            len_tmp = data_grad.shape[1]
            magnitude = torch.stft(data_grad[0].cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320)).permute(2, 1, 0)
            #kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 160))
            #dst = cv.dilate(magnitude.detach().cpu().numpy(), kernel)
            kernel = self.gkern(3, 1).astype(np.float32)
            stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            stack_kernel = np.expand_dims(stack_kernel, 3)
            dst = self.blur(magnitude, 1, stack_kernel)

            dst = torch.tensor(dst).permute(2,1,0)
            dst = torch.istft(dst.cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320), length = len_tmp)
            t = torch.rand(())
            if t<0.5:
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)


            print('grad', data_grad)
            if 1: 
                self.epsilon =  self.alpha/self.iteration
                adv = self.mi_fgsm_attack(_adv.cuda(), self.epsilon, data_grad.cuda())
            else:
                random_noise = torch.FloatTensor(_adv.shape).uniform_(-self.alpha, self.alpha).to(self.device)
                _adv = Variable(adv.data + random_noise, requires_grad=True)
                adv = self.pgd_attack(_adv, speech_raw, self.epsilon, self.alpha, data_grad)

            adv = adv.clone().detach().requires_grad_(True)
            batch_test = {}
            batch_test["speech"] = adv.squeeze(0)
            result = speech2text(**batch_test)
            for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                best_text_token = torch.tensor(token_int)
                best_text = text
                print(target_text_remade)
                print(best_text)
        speech_input = adv.cuda()
        if 0:
        #if t<35:
            tmp = TDIAttack(speech_input[0].cpu().detach().numpy(), t)
            tmp = torch.tensor(tmp).cuda()
        batch_o[1]["speech"] = speech_input
        #noise = batch["speech"][0].cpu() - speech_input[0].cpu()
        #noise = batch["speech"][0].cpu() - tmp.cpu()
        #batch_o[1]['speech'][0] -= 0.1 * noise
        return  batch_o, target_text_remade, sub

class GeneralAttack:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        streaming: bool = False,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        iteration: int = 1000,
        log_level: Union[int, str] = None,
        output_dir: str = None,
        ngpu: int = None,
        seed: int = None,
        num_workers: int = None,
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        results: dict = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        word_lm_file: Optional[str] = None,
        allow_variable_data_keys: bool = None,
        attack_type: dict = None,
        target_api: dict = None,
    ):
        assert check_argument_types()
        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        decoder = asr_model.decoder
        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                if streaming:
                    beam_search.__class__ = BatchBeamSearchOnlineSim
                    beam_search.set_streaming_config(asr_train_config)
                    logging.info("BatchBeamSearchOnlineSim implementation is selected.")
                else:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"Beam_search: {beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.iteration = iteration
        self.momentum = 0.5
        self.grad = 0
        self.flag = 0
        self.token_list = token_list
        self.token_type = token_type
    def gkern(self, kernlen = 3, nsig = 1):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(-2,x)
        kern2d = st.norm.pdf(-2,x)
        kernel_raw = np.outer(kern1d, kern2d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def blur(self, tensor_image, epsilon, stack_kerne):
        min_batch=tensor_image.shape[0]
        channels=tensor_image.shape[1]
        out_channel=channels
        kernel=torch.FloatTensor(stack_kerne).cuda()
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        x = torch.ones([1, 3, tensor_image.shape[1], tensor_image.shape[2]])
        x[:,:2,:,:] = tensor_image
        data_grad = torch.nn.functional.conv2d(x.cuda(),weight.cuda(),bias=None,stride=1,padding=(1,0), dilation=1)

        return data_grad[0,:2,:,:]

    def gaussian(self, batch):
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)
        data = np.array(batch["speech"][0].cpu())
        snr = np.random.uniform(20)
        tmp = awgn(data, snr, out='signal', method='vectorized', axis=0)
        res = torch.tensor(tmp)
        return res, target_text_pre
    def speed(self, batch):
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)
        data = np.array(batch["speech"][0].cpu())
        speed_rate = np.random.uniform(0.3)
        wav_speed_tune = cv.resize(data, (1, int(len(data) * speed_rate))).squeeze()
        tmp = wav_speed_tune
        res = torch.tensor(tmp)
        return res, target_text_pre
    def tdi(self, batch):
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)

        t = 3
        tmp = TDIAttack(batch["speech"][0].cpu().detach().numpy(), t)
        res = torch.tensor(tmp)
        return res, target_text_pre
    def demand(self, batch, g_type, a1, a2):
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)

        data = batch['speech'][0]
        t = np.random.randint(1, 16)
        if t<10:
            t = 'ch0'+ str(t)
        else:
            t = 'ch' + str(t)
        path = os.path.join('data/demand', str(g_type), t + '.wav')
        noise, rate = torchaudio.load(path)
        noise = noise[0]
        times = len(data) // len(noise) + 1
        noise = np.repeat(noise, times, 0)
        noise = noise[:len(data)]
        adv = a1 * torch.tensor(noise) + a2 * data.cpu()
        return adv, target_text_pre
    def style(self, batch):
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = self.converter.ids2tokens(target_text_pre)
        target_text_pre = self.tokenizer.tokens2text(token)

        data = batch['speech'][0]
        path_t = '../../../espnet2/attack/style_trans'
        logf0s_normalization = path_t + '/cache/logf0s_normalization.npz'
        mcep_normalization = path_t + '/cache/mcep_normalization.npz'
        coded_sps_A_norm = path_t + '/cache/coded_sps_A_norm.pickle'
        coded_sps_B_norm = path_t + '/cache/coded_sps_B_norm.pickle'
        model_checkpoint = path_t + '/checkpoint/'
        validation_A_dir = path_t + '/data/S0913/'
        output_A_dir = path_t + '/converted_sound/S0913/' 
        validation_B_dir = path_t + '/data/gaoxiaosong/'
        output_B_dir = path_t + '/converted_sound/gaoxiaosong/'
        resume_training_at = path_t + '/checkpoint/_CycleGAN_CheckPoint'
        cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_A_norm=coded_sps_A_norm,
                                coded_sps_B_norm=coded_sps_B_norm,
                                model_checkpoint=model_checkpoint,
                                validation_A_dir=validation_A_dir,
                                output_A_dir=output_A_dir,
                                validation_B_dir=validation_B_dir,
                                output_B_dir=output_B_dir,
                                restart_training_at=resume_training_at)
        adv = cycleGAN.validation_for_A_dir_(data)
        return adv, target_text_pre 
