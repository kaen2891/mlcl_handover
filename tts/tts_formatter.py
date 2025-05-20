import os
from glob import glob
import torchaudio

def libri_formatter(root_path, meta_file_train=None, **kwargs):
    samples = []
    for speaker_dir in glob(os.path.join(root_path, "*", "*")):
        trans_files = [f for f in os.listdir(speaker_dir) if f.endswith(".trans.txt")]
        for trans_file in trans_files:
            trans_path = os.path.join(speaker_dir, trans_file)
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", maxsplit=1)
                    if len(parts) != 2:
                        continue
                    file_id, text = parts
                    audio_path = os.path.join(speaker_dir, file_id + ".flac")
                    if not os.path.exists(audio_path):
                        continue
                    samples.append({
                        "text": text,
                        "audio_file": audio_path,
                        "speaker_name": file_id.split("-")[0],
                        "root_path": root_path,
                        "language": "en"
                    })
    return samples
    

from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig

dataset_config = BaseDatasetConfig(
    formatter="custom",
    meta_file_train="",  
    path="/home2/jw/workspace/asr/mlcl_handover/asr/dataset/LibriSpeech/LibriSpeech/train-clean-100/",
    language="en"
)

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=libri_formatter)


# if you want to save this?, remove comments
import pandas as pd

train_df = pd.DataFrame(train_samples)
eval_df = pd.DataFrame(eval_samples)

train_df.to_csv("./train_samples.csv", index=False, encoding='utf-8')
eval_df.to_csv("./eval_samples.csv", index=False, encoding='utf-8')