import os
import sys
from sys import path

os.chdir('./espnet/egs2/aishell/asr1')
os.system('CUDA_VISIBLE_DEVICES=1 ./run.sh --skip_data_prep true --skip_train true ') 
