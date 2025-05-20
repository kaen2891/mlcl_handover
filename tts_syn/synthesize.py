from glob import glob
import os
import pandas as pd
from tqdm import tqdm

import torch


import argparse

parser = argparse.ArgumentParser('argument for TTS generation')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='xtts', choices=['xtts', 'your_tts', 'vits', 'tacotron2'])

args = parser.parse_args()

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

annotation = './dataset/train-clean-100.csv'

if args.model == 'xtts':
    args.model_name = 'xtts_v2'
    args.model_dir = 'tts_models/multilingual/multi-dataset/xtts_v2'
elif args.model == 'your_tts':
    args.model_name = args.model
    args.model_dir = 'tts_models/multilingual/multi-dataset/your_tts'
elif args.model == 'vits':
    args.model_name = args.model
    args.model_dir = 'tts_models/en/ljspeech/vits'
elif args.model == 'tacotron2':
    args.model_name = 'tacotron2-DDC'
    args.model_dir = 'tts_models/en/ljspeech/tacotron2-DDC'

wav_dir = './dataset/'
save_dir = './dataset/synthetic_0522/{}'.format(args.model_name)

#os.makedirs(save_dir, exist_ok=True)

print(annotation)

df = pd.read_csv(annotation)
files = df['file_path'].values.tolist()
texts = df['text'].values.tolist()

from TTS.api import TTS
tts = TTS(args.model_dir).to(device)

print('tts', tts)

for i, (file, text) in enumerate(tqdm(zip(files, texts), total=len(files), desc="synthesizing")):
    _dir, filename = os.path.split(file)
    save_wav_dir = os.path.join(save_dir, _dir)
    os.makedirs(save_wav_dir, exist_ok=True)
    
    if os.path.isfile(os.path.join(save_wav_dir, filename.replace('.flac', '.wav'))):
        continue
    
    
    if args.model in ['xtts', 'your_tts']:
        tts.tts_to_file(text=text, speaker_wav=os.path.join(wav_dir, file), language="en", file_path=os.path.join(save_wav_dir, filename.replace('.flac', '.wav')))
    else:
        tts.tts_to_file(text=text, speaker_wav=os.path.join(wav_dir, file), file_path=os.path.join(save_wav_dir, filename.replace('.flac', '.wav')))


