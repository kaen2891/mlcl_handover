import os
import random
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class ASRDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        
        self.args = args
        self.sample_rate = args.sample_rate
        self.processor = self.args.processor
        self.train_flag = train_flag
        
        if self.train_flag == 'train':
            annotation_file = os.path.join(args.data_folder, args.train_annotation_file)
            print(annotation_file)
            self.data_inputs = self.get_text_audio(annotation_file)           
        
        elif self.train_flag == 'dev_clean':
            annotation_file = os.path.join(args.data_folder, args.dev_clean_annotation_file)
            self.data_inputs = self.get_text_audio(annotation_file)
        
        elif self.train_flag == 'dev_other':
            annotation_file = os.path.join(args.data_folder, args.dev_other_annotation_file)
            self.data_inputs = self.get_text_audio(annotation_file)           
            
        elif self.train_flag == 'test_clean':
            annotation_file = os.path.join(args.data_folder, args.test_clean_annotation_file)
            self.data_inputs = self.get_text_audio(annotation_file)
        
        elif self.train_flag == 'test_other':
            annotation_file = os.path.join(args.data_folder, args.test_other_annotation_file)
            self.data_inputs = self.get_text_audio(annotation_file)
                
    def get_text_audio(self, annotation_file):
        df = pd.read_csv(annotation_file)
        data_inputs = []        
        files = df['file_path'].values.tolist()
        texts = df['text'].values.tolist()
                        
        data_inputs = list(zip(files, texts))
        return data_inputs

    def __getitem__(self, index):
        audio, text = self.data_inputs[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.args.data_folder, audio))
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        input_features = self.processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=self.sample_rate).input_features.squeeze(0)
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        
        return input_features, labels, text
    
    def __len__(self):
        return len(self.data_inputs)