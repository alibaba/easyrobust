#!/usr/bin/env python3 
import argparse
import random
import logging
from pathlib import Path
import sys
sys.path.append('../../..')
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
import numpy as np
import torch
import torchaudio
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
from espnet2.tasks.asr_ import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none

from torch.autograd import Variable
import scipy.stats as st
import os
import soundfile
from scipy.signal import butter, lfilter
import random
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import jieba
import jieba.posseg as pseg
import nltk
import librosa
from snownlp import sentiment
from snownlp import SnowNLP
from esp_zoo.espnet_model_zoo.espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
import cv2 as cv

from espnet2.attack.nlp_fluency.models import MaskedBert, MaskedAlbert
from transformers import AutoTokenizer, AutoModel
import librosa.display
import matplotlib.pyplot as plt
def plot_spectrogram(mag, save=''):
        librosa.display.specshow(mag, x_axis='off', cmap='viridis')
        plt.title('spectrogram')
        if save != '':
            plt.savefig(save, format='jpg')
            plt.close()
        else:
            plt.show()

#from espnet2.bin.asr_inference import Speech2Text
token_list = "./data/en_token_list/bpe_unigram5000/tokens.txt"
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
#Time Domain Interval Attack (#1 in paper)
def TDIAttack(data, windowSize):
    n = int(len(data)/windowSize)
    #Breaks array into buckets of elements
    #Each bucket has 'windowSize' amount of elements
    def createBuckets(arr, n):
        length = len(arr)
        return [ arr[i*length // n: (i+1)*length // n] 
                 for i in range(n) ]

    #Load audio file
    arr = np.copy(data)
    
    #Store split array into variable
    splitArray = createBuckets(arr,n)

    l = list()

    for x in splitArray[:n]:
        l.extend(np.fliplr([x])[0])
    
    #Stores the modified array and casts it as int
    data2 = np.asanyarray(l)
    data2= np.asarray(data2)

    return data2

def highpass_filter(data, cutoff=1000, fs=25000, order=10): #7000
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)
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
def lowpass_filter_1(wav_data, sample_rate, crit_freq = 1000 ):

    N = 10                            # order of the filter
    nyq_freq = float(sample_rate)/2.  # nyquist frequency
    crit_freq_norm = crit_freq/nyq_freq    # critical frequency (normalized)
    rp = 1                            # passband ripple (dB)
    rs = 80                           # stopband min attenuation (dB)
    btype = 'lowpass'
    ftype = 'ellip'
    b, a = sig.iirfilter(N, crit_freq_norm, rp, rs, btype, ftype)

    # apply filter and return filtered data
    return  sig.lfilter(b, a, wav_data)
class Speech2Text_attack:
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
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        alpha: float = 0.1,
        iteration: int = 1000,
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
        perturbed_sound = sound - tmp
        
        return perturbed_sound
    def mi_fgsm_attack(self, sound, epsilon, data_grad):
        '''
        if self.flag == 0:
            self.grad = data_grad
            self.flag = 1
        '''
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
    def __call__o(
            self, batch, speech2text, text2speech) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
    
        batch_o = batch
        batch = batch[1]
        
        target_text_pre = batch["text"][0]
        token = speech2text.converter.ids2tokens(target_text_pre)
        target_text_pre = speech2text.tokenizer.tokens2text(token)
        print(target_text_pre)
        

        from nltk import word_tokenize
        sent_tokenizer = nltk.data.load('./punkt/english.pickle')
        seg_list = word_tokenize(target_text_pre)
        #seg_list = nltk.pos_tag(seg_list) 
        len_seg_list = len(seg_list)
        #position_w = random.randint(1,len_seg_list)
        taget_text_remade = []
        position_c = 0
        position_p = 0
        #del_word = seg_list[position_w-1]
        #p_flag = len(del_word)
        bian = 0
        ########################################## Del
        t = random.randint(0,len_seg_list-2)
        for i in range(len(seg_list)):
            if i > t and len(seg_list[i]) > 5 and not bian:
                del_word = seg_list[i]
                p_flag = len(del_word)
                bian = 1
                continue
            else:
                taget_text_remade.append(seg_list[i])
                if not bian:
                    position_c += 1
                    position_c += len(seg_list[i])
                    position_p += len(seg_list[i])
        if bian == 0 :
            del_word = seg_list[0]
            p_flag = len(del_word)
            bian = 1
            taget_text_remade.pop(0)
            position_c = 0
            position_p = 0

        target_text_remade = " ".join(taget_text_remade) 
        position_c_left = position_c-1 
        position_c_right = position_c+p_flag+1
        mid_word = ',' * p_flag
        target_text_remade_engry = " ".join(taget_text_remade[:position_c_left])  + mid_word + " ".join(taget_text_remade[position_c_right:])
        ######### mid_target
        tmp = 1.0 * (position_p+t-2) / len(target_text_pre)
        tmp_2 = 1.0 * (position_p+p_flag+t-2) / len(target_text_pre)
        position_left = int(len(batch["speech"][0])*tmp)
        position_right = int(len(batch["speech"][0])*tmp_2)
        ''' 
        print(target_text_pre[position_c-2: position_c+p_flag+2])
        print('ssss')
        print(target_text_pre[position_c+p_flag+2])
        print(target_text_pre)
        print(taget_text_remade)
        print(del_word)
        tem_speech = batch["speech"][0][position_left:position_right]
        tem_speech_1 = batch["speech"][0][:position_left]
        torchaudio.save('./zem.wav', tem_speech.cpu(), 16000)
        torchaudio.save('./zem_1.wav', tem_speech_1.cpu(), 16000)
        exit()
        '''
        self.grad = 0
        local_flag = 0
        if not local_flag:
            adv_o = batch["speech"][0]
            adv = batch["speech"][0]
            adv = adv[position_left : position_right]
            speech_len = torch.tensor([adv.shape[0]])
            self.grad = 0
            i = 0
            p = 0.5
            adv = adv.clone().detach().requires_grad_(True).to(self.device).unsqueeze(0)

            #tokens = tokenizer.text2tokens(target_text_remade_engry)
            tokens = speech2text.tokenizer.text2tokens(mid_word)
            text_ints = speech2text.converter.tokens2ids(tokens)
            text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
            text_len = torch.tensor([text_input.shape[1]])
        gaijin = 0
        for i in range(self.iteration//2):
            _adv = adv.clone().detach().requires_grad_(True)
            if gaijin:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, 0, enc, m1, m2, position, position+p_flag )

            else:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, 0, 0, 0, 0, 0, 0)
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
            if 0 :
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)

            if self.attack_type == "fgsm":
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
            
            adv_left = adv[:position_left]
            hanming = torch.tensor(np.hamming(adv_left.shape[0])).to(torch.float32).cuda()
            adv_left = adv_left.mul(hanming).clone().detach().requires_grad_(True).to(self.device)
            
            adv_right = adv[position_left + adv_.shape[0]:]
            hanming = torch.tensor(np.hamming(adv_right.shape[0])).to(torch.float32).cuda()
            adv_right = adv_right.mul(hanming).clone().detach().requires_grad_(True).to(self.device)
            
            adv = batch["speech"][0].cuda()
            adv[:position_left] = adv_left
            adv[position_left + adv_.shape[0]:] = adv_right
            adv[position_left : position_left + adv_.shape[0]] =  adv_.cuda()
            speech_input = adv.unsqueeze(0)
        self.grad = 0
        adv = speech_input.clone().detach().requires_grad_(True).to(self.device).cuda()
        speech_len = torch.tensor([adv.shape[1]])
        tokens =  speech2text.tokenizer.text2tokens(target_text_remade)
        text_ints = speech2text.converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        print(target_text_remade, '------------target_text_remade--------')
        for i in range(self.iteration*2):
            _adv = adv.clone().detach().cuda().requires_grad_(True)
            t = torch.rand(())
            if gaijin:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), position_s-1000, position_s-1000 + 5000 * p_flag, 0, 0, 0, 0, 0)
            else:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), tmp, tmp_2, 0, 0, 0, 0, 0)
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                cer = retval["cer"][0]
            else:
                loss, stats, weight = retval
            ppl = 0
            if ppl:
                batch_test = {}
                batch_test["speech"] = _adv.squeeze(0)
                result = speech2text(**batch_test)
                for n, (text, token, token_int, hyp) in zip(range(1, 1 + 1), result):
                    best_text_token = torch.tensor(token_int)
                    best_text = text
                sentences = [best_text,target_text_remade ]
                ppl_1 = self.ppl_model.perplexity(
                x=" ".join(best_text),
                verbose=False,
                temperature=1.0,
                batch_size=50,
                )
                loss += torch.abs(torch.tensor(ppl_1-ppl_target))
            
            
            self.asr_model.zero_grad()
            stats = {k: v for k, v in stats.items() if v is not None}
            
            #loss = 0.1 * loss + 10*stats['cer'] + stats['wer'] + 10*stats['cer_ctc']
            loss.backward(retain_graph=True)
            data_grad = _adv.grad.data
            
            #plot_spectrogram(data_grad.cpu().numpy(), './tem_spec/ctc_att.wav')
            #exit()
            
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
            if self.attack_type == "fgsm":
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
                '''
                seg_list_1 = word_tokenize(target_text_remade)
                seg_list_2 = word_tokenize(best_text)
                if seg_list_1 == seg_list_2:
                    print('white attack success')
                else:
                    print(seg_list_1)
                    print(seg_list_2)
                    print('unsuccess')
                '''
        speech_input = adv.cuda()
        batch_o[1]["speech"] = speech_input
        return  batch_o, target_text_pre, target_text_remade, del_word
    #@torch.no_grad()
    def __call__(
            self, batch, speech2text, text2speech) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        file_del_time = './utt2dur'
        utt2dur_time = {}
        with open(file_del_time, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                key, time_left, time_right, _ = lines.strip().split(' ')
                utt2dur_time[key] = [time_left, time_right]
        
        file_del = './utt_text_100'
        utt2dur = {}
        with open(file_del, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                key, del_word = lines.strip().split(' ')
                utt2dur[key] = del_word
                
        batch_o = batch
        key = batch[0][0]
        del_word = utt2dur[key]
        p_flag = len(del_word)
        batch = batch[1]
        target_text_pre = batch["text"][0]
        token = speech2text.converter.ids2tokens(target_text_pre)
        target_text_pre = speech2text.tokenizer.tokens2text(token)
        target_text_pre_tem = target_text_pre.split(' ')
        del_position = 0
        for i in range(len(target_text_pre_tem)):
            if target_text_pre_tem[i] == del_word:
                del_position = i
                break
        target_text_remade = ' '.join(target_text_pre_tem[:del_position]+target_text_pre_tem[del_position+1:])
        mid_word = ',' * p_flag
        target_text_remade_engry = " ".join(target_text_pre_tem[:del_position])  + ' ' + mid_word + ' ' + " ".join(target_text_pre_tem[del_position+1:])
        
        time_left, time_right = utt2dur_time[key][0], utt2dur_time[key][1]
        position_left = int((float(time_left)+0.1)* 16000)
        position_right = int(float(time_right) * 16000)
        
        adv = batch["speech"]
        speech_len = torch.tensor([adv.shape[1]])
        tokens =  speech2text.tokenizer.text2tokens('HELLO '*20)
        text_ints = speech2text.converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        '''
        print(target_text_pre)
        print(target_text_remade)
        print(target_text_remade_engry)
        print(del_word)
        print(adv.shape)
        tem_speech = batch["speech"][0][position_left:position_right]
        tem_speech_1 = batch["speech"][0][:position_left]
        torchaudio.save('./cut_wav/' + key + '.wav', tem_speech.cpu(), 16000)
        torchaudio.save('./cut_wav/' + key + '1.wav', tem_speech_1.cpu(), 16000)
        '''

        self.grad = 0
        local_flag = 0
        if not local_flag:
            adv_o = batch["speech"][0]
            adv = batch["speech"][0]
            adv = adv[position_left: position_right]
            #adv = torch.tensor(highpass_filter_1(adv.clone().data, 16000, 500)).to(torch.float32)
            speech_len = torch.tensor([adv.shape[0]])
            self.grad = 0
            i = 0
            p = 0.5
            adv = adv.clone().detach().requires_grad_(True).to(self.device).unsqueeze(0)

            #tokens = tokenizer.text2tokens(target_text_remade_engry)
            tokens = speech2text.tokenizer.text2tokens(mid_word)
            text_ints = speech2text.converter.tokens2ids(tokens)
            text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
            text_len = torch.tensor([text_input.shape[1]])

        gaijin = 0
        for i in range(self.iteration//2):
            _adv = adv.clone().detach().requires_grad_(True)
            if gaijin:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, 0, enc, m1, m2, position, position+p_flag )

            else:
                print(_adv.shape, speech_len, text_input.shape, text_len)
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, 0, 0, 0, 0, 0, 0)
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

            loss.backward(retain_graph=True)
            data_grad = _adv.grad.data
            print('grad', data_grad)
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

            if self.attack_type == "fgsm":
                self.epsilon = self.alpha/self.iteration
                adv = self.fgsm_attack(_adv, self.epsilon, data_grad)

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
        if 1:
            print(adv_o, '777777777')
            adv_ = adv[0]
            hanming = torch.tensor(np.hamming(adv_.shape[0])).to(torch.float32).cuda()
            adv_ = adv_.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv_left = adv_o[:position_left].cuda()
            hanming = torch.tensor(np.hamming(adv_left.shape[0])).to(torch.float32).cuda()
            adv_left = adv_left.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv_right = adv_o[position_left + adv_.shape[0]:].cuda()
            hanming = torch.tensor(np.hamming(adv_right.shape[0])).to(torch.float32).cuda()
            adv_right = adv_right.mul(hanming).clone().detach().requires_grad_(True).to(self.device)

            adv = adv_o.cuda()
            adv[:position_left] = adv_left
            adv[position_left + adv_.shape[0]:] = adv_right
            adv[position_left : position_left + adv_.shape[0]] =  adv_.cuda()
            speech_input = adv.unsqueeze(0)
        
        self.grad = 0
        adv = speech_input.clone().detach().requires_grad_(True).to(self.device).cuda()
        speech_len = torch.tensor([adv.shape[1]])
        torch.cuda.empty_cache()
        tokens =  speech2text.tokenizer.text2tokens(target_text_remade)
        text_ints = speech2text.converter.tokens2ids(tokens)
        text_input = torch.tensor(np.array(text_ints, dtype=np.int64)).unsqueeze(0)
        text_len = torch.tensor([text_input.shape[1]])
        for i in range(self.iteration*2):
            _adv = adv.clone().detach().cuda().requires_grad_(True)
            t = torch.rand(())
            if gaijin:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), position_s-1000, position_s-1000 + 5000 * p_flag, 0, 0, 0, 0, 0)
            else:
                retval = self.asr_model(_adv, speech_len.cuda(), text_input.cuda(), text_len.cuda(), 0, 0, 0, 0, 0, 0, 0)
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

            #plot_spectrogram(data_grad.cpu().numpy(), './tem_spec/ctc_att.wav')
            #exit()
            '''
            len_tmp = data_grad.shape[1]
            magnitude = torch.stft(data_grad[0].cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320)).permute(2, 1, 0)
            kernel = self.gkern(3, 1).astype(np.float32)
            stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            stack_kernel = np.expand_dims(stack_kernel, 3)
            dst = self.blur(magnitude, 1, stack_kernel)

            dst = torch.tensor(dst).permute(2,1,0)
            dst = torch.istft(dst.cpu(), n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320), length = len_tmp)
            t = torch.rand(())
            if 0:
            #if t<0.5 :
                data_grad = dst.unsqueeze(0).clone().detach().cuda().requires_grad_(True)

            '''
            print('grad', data_grad)
            if self.attack_type == "fgsm":
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
                print('            __________________________________________________ _______________________               ')
                print(target_text_remade)
                print(best_text)
            speech_input = adv.cuda()
        batch_o[1]["speech"] = speech_input
        
        return  speech_input,  target_text_remade, del_word

class Speech2Text:
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
        self.token_list = token_list
        self.token_type = token_type

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[Optional[str], List[str], List[int], Hypothesis]]:
        """Inference
 
        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _ = self.asr_model.encode(**batch)
        assert len(enc) == 1, len(enc)

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=enc[0], maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
        )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, Hypothesis), type(hyp)

            # remove sos/eos and get results
            token_int = hyp.yseq[1:-1].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        assert check_return_type(results)
        return results


def ensemble_attack(batch, speech2text_attack_1, speech2text_attack_2,  speech2text, text2speech):
    target_text_pre = None
    del_word = None
    for i in range(50):
        batch, target_text_pre, target_text_remade, del_word = speech2text_attack_1(batch, speech2text, text2speech, i, target_text_pre)
        torch.cuda.empty_cache()
        #batch, target_text_pre, target_text_remade, del_word = speech2text_attack_2(batch, speech2text, text2speech, i+1, target_text_pre, del_word)
        #batch, target_text_pre, target_text_remade, del_word = speech2text_attack_3(batch, speech2text, text2speech, i+1, target_text_pre)
    
    return i, batch[1]["speech"], target_text_remade, del_word


def inference(
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
):
    '''
    asr_train_config_1  = './exp_1/exp/asr_train_asr_transformer_e18_raw_bpe_sp/config.yaml' 
    asr_model_file_1 = './exp_1/exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth'
    lm_train_config_1 = './exp_1/exp/lm_train_lm_adam_bpe/config.yaml'
    lm_file_1 = './exp_1/exp/lm_train_lm_adam_bpe/17epoch.pth'
    asr_train_config_2 = './exp_2/exp/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp/config.yaml'
    asr_model_file_2 = './exp_2/exp/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp/valid.acc.ave_10best.pth'
    lm_train_config_2 = './exp_2/exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/config.yaml'
    lm_file_2 = './exp_2/exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/valid.loss.ave_10best.pth'
    #lm_train_config = 'exp2/kamo-naoyuki/aishell_conformer/lm/config.yaml'
    #lm_file = 'exp2/kamo-naoyuki/aishell_conformer/lm/valid.loss.ave_10best.pth'
    '''
    asr_train_config_1 ='./exp_2/exp/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp/config.yaml'
    asr_model_file_1 = './exp_2/exp/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp/valid.acc.ave_10best.pth'
    lm_train_config_1 = './exp_2/exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/config.yaml'
    lm_file_1 = './exp_2/exp/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/valid.loss.ave_10best.pth'


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
    print('***********************************!!!!!!!!')
    # 1. Set random-seed
    set_all_random_seed(seed)
    attack_type = "fgsm"
    epsilon = 0.0001
    alpha = 0.02
    iteration = 50
    device = "cuda"
    #asr_train_config = 'exp/kamo-naoyuki/aishell_conformer/train_asr_transformer.yaml'
    
    #asr_train_config = './conf/' 
    # 2. Build speech2text_attack
    speech2text_attack_1 = Speech2Text_attack(
        asr_train_config=asr_train_config_1,
        asr_model_file=asr_model_file_1,
        lm_train_config=lm_train_config_1,
        lm_file=lm_file_1,
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
        attack_type=attack_type,
        epsilon=epsilon,
        alpha=alpha,
        iteration=iteration,
    )
   

    # 2.1 Build speech2text
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
    d = ModelDownloader()
    text2speech = None
    #text2speech = Text2Speech(**d.download_and_unpack("./tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave.zip")) 
    #text2speech = Text2Speech('./tts/exp/tts_train_xvector_transformer_raw_phn_tacotron_g2p_en_no_space/config.yaml', './tts/exp/tts_train_xvector_transformer_raw_phn_tacotron_g2p_en_no_space/train.loss.ave_5best.pth')
    #text2speech = Text2Speech('./tts_train_fastspeech_raw_phn_pypinyin_g2p_phone_train.loss.best.zip')
    text2speech = Text2Speech(**d.download_and_unpack('./tts_train_fastspeech_raw_phn_pypinyin_g2p_phone_train.loss.best.zip'))
    # 3. Build data-iterator
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
    
    perturbed_data = torch.tensor(0)
    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        SUCCESS = []
        text_trgeted = []
        keys = []
        del_words = []
        for  batch in loader:
            #_bs = len(next(iter(batch.values())))
            #iassert len(keys) == _bs, f"{len(keys)} != {_bs}"
            
            #batch = {v[0]v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            
            # N-best list of (text, token, token_int, hyp_object)
            batch_2 = batch
            #success_1, adv_data_1, target_text_remade_1, del_word = ensemble_attack(batch, speech2text_attack_1, speech2text_attack_2, speech2text, text2speech)
            adv_data_1, target_text_remade_1, del_word = speech2text_attack_1(batch, speech2text, text2speech)
            
            text_trgeted.append(target_text_remade_1)
            del_words.append(del_word)
            adv_data = torch.tensor(adv_data_1)
            # process of batch_test

            batch_test = {}
            batch_test["speech"] = batch[1]["speech"][0]

            #batch_test = to_device(batch_test, device=device)
            if device == "cuda":
                batch_test["speech"] =  adv_data.detach().squeeze(0).cuda()
            else:
                batch_test["speech"] =  adv_data.detach().squeeze(0)
            adv_data = adv_data.detach()
            torch.cuda.empty_cache()

            results = speech2text(**batch_test)

            # Only supporting batch_size==1
            key = batch[0][0]
            keys.append(key)
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    ibest_writer["text"][key] = text
            if not os.path.isdir(output_dir+'/'+'output_adv'):
                os.mkdir(output_dir+'/'+'output_adv')
            print(text,'target_text_oriangel')
            #torchaudio.save('./exp/asr_config_raw_en_bpe5000_sp/decode_asr_lm_lm_train_lm_adam_en_bpe5000_valid.loss.ave_asr_model_valid.acc.best/test_clean/logdir/output.1/output_adv/'+ key + '_adv.wav', adv_data.cpu(), 16000)
            #torchaudio.save(output_dir+'/'+'output_adv'+'/'+key+'_adv.wav', adv_data[0].cpu(), 16000)
        print('------target_asr-------',torch.sum(torch.tensor(SUCCESS))/len(SUCCESS))
        filename_label = output_dir+'/'+'output_adv'+'/'+'text_target.txt'
        with open(filename_label, 'w') as f:
            j = 0
            for i in text_trgeted:
                f.write(keys[j] + ' ' + i + ' ' + del_words[j] + '\n')
                j +=1

                


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
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
