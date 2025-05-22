import os
import pandas as pd
import torchaudio



def get_librispeech_data(librispeech_path):
    audio_files = []
    for root, _, files in os.walk(librispeech_path):
        transcriptions = {}
        for file in files:
            if file.endswith(".trans.txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcriptions[parts[0]] = parts[1]

        for file in files:
            if file.endswith(".flac"):
                _dir, file_num = os.path.split(root)
                _dir_dir, speak_num = os.path.split(_dir)
                _dir_dir_dir, set_name = os.path.split(_dir_dir)
                _dir_dir_dir_dir, libri = os.path.split(_dir_dir_dir)
                _dir_dir_dir_dir_dir, libri2 = os.path.split(_dir_dir_dir_dir)
                file_path = os.path.join(root, file)
                save_dir = os.path.join(libri2, libri, set_name, speak_num, file_num, file)               
                
                waveform, sample_rate = torchaudio.load(file_path)
                duration = waveform.shape[1] / sample_rate
                
                file_id = os.path.splitext(file)[0]
                text = transcriptions.get(file_id, "") 
                
                audio_files.append((save_dir, duration, text))
    
    df = pd.DataFrame(audio_files, columns=["file_path", "duration", "text"])
    return df

LIBRISPEECH_PATH = "../dataset/LibriSpeech/LibriSpeech/dev-clean"
df = get_librispeech_data(LIBRISPEECH_PATH)
df.to_csv('../dataset/dev-clean.csv', index=False)

LIBRISPEECH_PATH = "../dataset/LibriSpeech/LibriSpeech/dev-other"
df = get_librispeech_data(LIBRISPEECH_PATH)
df.to_csv('../dataset/dev-other.csv', index=False)

LIBRISPEECH_PATH = "../dataset/LibriSpeech/LibriSpeech/test-clean"
df = get_librispeech_data(LIBRISPEECH_PATH)
df.to_csv('../dataset/test-clean.csv', index=False)

LIBRISPEECH_PATH = "../dataset/LibriSpeech/LibriSpeech/test-other"
df = get_librispeech_data(LIBRISPEECH_PATH)
df.to_csv('../dataset/test-other.csv', index=False)