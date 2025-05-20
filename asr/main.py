from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.asr_dataset import ASRDataset
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save') 
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='librispeech')
    parser.add_argument('--data_folder', type=str, default='./dataset/')    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--train_annotation_file', type=str, default='original_annotation/real_set.csv')
    parser.add_argument('--dev_clean_annotation_file', type=str, default='./data/')
    parser.add_argument('--dev_other_annotation_file', type=str, default='./data/')
    parser.add_argument('--test_clean_annotation_file', type=str, default='./data/')
    parser.add_argument('--test_other_annotation_file', type=str, default='./data/')
    

    # model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--framework', type=str, default='transformers', 
                        help='using pretrained speech models from s3prl or huggingface', choices=['transformers'])
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    parser.add_argument('--method', type=str, default='ce')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    return args


def collate_fn(batch):
    input_features, labels, texts = zip(*batch)
    input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_features_padded, labels_padded, texts

def set_loader(args):
    train_dataset = ASRDataset(train_flag='train', transform=None, args=args, print_flag=True)
    val_clean_dataset = ASRDataset(train_flag='dev_clean', transform=None, args=args, print_flag=True)
    val_other_dataset = ASRDataset(train_flag='dev_other', transform=None, args=args, print_flag=True)
    
    test_clean_dataset = ASRDataset(train_flag='test_clean', transform=None, args=args, print_flag=True)
    test_other_dataset = ASRDataset(train_flag='test_other', transform=None, args=args, print_flag=True)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    val_clean_loader = torch.utils.data.DataLoader(val_clean_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_other_loader = torch.utils.data.DataLoader(val_other_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    test_clean_loader = torch.utils.data.DataLoader(test_clean_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_other_loader = torch.utils.data.DataLoader(test_other_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, val_clean_loader, val_other_loader, test_clean_loader, test_other_loader, args
    


def set_model(args):
    args.processor = WhisperProcessor.from_pretrained("openai/{}".format(args.model)) # openai/whisper-tiny    
    model = WhisperForConditionalGeneration.from_pretrained("openai/{}".format(args.model)) # openai/whisper-tiny
        
    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)    
    criterion = nn.CrossEntropyLoss()    
    criterion = [criterion.cuda()]
            
    model.cuda()    
    optim_params = model.parameters()
    optimizer = set_optimizer(args, optim_params)
    
    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    
    avg_pool = nn.AdaptiveAvgPool1d(1)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #wers = AverageMeter()
    end = time.time()
    
    for idx, (input_features, labels, texts) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        input_features = input_features.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        bsz = labels.shape[0]
                
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict())]
                alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            outputs = model(input_features, labels=labels)
            loss = outputs.loss
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg



def validate(val_loader, model, criterion, args, best_wer, best_model=None):
    save_bool = False
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    references = []
    hypothesis = []
    
    with torch.no_grad():
        end = time.time()
    
        #for idx, (input_features, labels, texts) in enumerate(val_loader):
        for idx, (input_features, labels, texts) in enumerate(val_loader):
            # data load
            data_time.update(time.time() - end)
            input_features = input_features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            
            with torch.cuda.amp.autocast():
                outputs = model(input_features, labels=labels)
                loss = outputs.loss
                generated_ids = model.generate(input_features)
            
            losses.update(loss.item(), bsz)
            transcription = args.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            references.extend(texts)
            hypothesis.extend(transcription)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses))
        
    wer_score = wer(references, hypothesis)
    
    if wer_score < best_wer[0]:
        save_bool = True
        best_wer = [wer_score]
        #best_model = [deepcopy(model.state_dict())]
        best_model = [deepcopy(model.state_dict())]

    print(' * WER: {} (Best WER: {})'.format(wer_score, best_wer[0]))
    return best_wer, best_model, save_bool, losses.avg, wer_score

def plot_metrics(losses, name, args):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    
    plt.savefig(os.path.join(args.save_folder, name))
    plt.cla()
    plt.clf()


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    #torch.autograd.set_detect_anomaly(True)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_dev_clean_model = None
    best_dev_other_model = None
    best_test_clean_model = None
    best_test_other_model = None
    
    best_dev_clean = [100]
    best_dev_other = [100]
    best_test_clean = [100]
    best_test_other = [100]
    
    model, criterion, optimizer = set_model(args)
    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    train_loader, dev_clean_loader, dev_other_loader, test_clean_loader, test_other_loader, args = set_loader(args)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    
    total_loss = 0.0
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        train_losses = []
        dev_clean_losses = []
        dev_other_losses = []
        dev_clean_wers = []
        dev_other_wers = []
        
        test_clean_losses = []
        test_other_losses = []
        test_clean_wers = []
        test_other_wers = []
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)
            
            # train for one epoch
            time1 = time.time()
            #loss, wer = train(train_loader, model, criterion, optimizer, epoch, args, total_loss, scaler)            
            loss = train(train_loader, model, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, Loss {:.2f}'.format(epoch, time2-time1, loss))
            
            # eval for one epoch
            
            best_dev_clean, best_dev_clean_model, dev_clean_save_bool, dev_clean_loss, dev_clean_wer = validate(dev_clean_loader, model, criterion, args, best_dev_clean, best_dev_clean_model)
            print('Dev-Clean WER = {}'.format(best_dev_clean[0]))            
            best_dev_other, _, dev_other_save_bool, dev_other_loss, dev_other_wer = validate(dev_other_loader, model, criterion, args, best_dev_other, best_dev_other_model)
            print('Dev-Other WER = {}'.format(best_dev_other[0]))            
            best_test_clean, best_test_clean_model, test_clean_save_bool, test_clean_loss, test_clean_wer = validate(test_clean_loader, model, criterion, args, best_test_clean, best_test_clean_model)
            print('Test-Clean WER = {}'.format(best_test_clean[0]))            
            best_test_other, _, test_other_save_bool, test_other_loss, test_other_wer = validate(test_other_loader, model, criterion, args, best_test_other, best_test_other_model)
            print('Test-Other WER = {}'.format(best_test_other[0]))
                        
            train_losses.append(loss)
            dev_clean_losses.append(dev_clean_wer)
            dev_other_losses.append(dev_other_loss)
            test_clean_losses.append(test_clean_loss)
            test_other_losses.append(test_other_loss)
            
            dev_clean_wers.append(dev_clean_wer)
            dev_other_wers.append(dev_other_wer)
            test_clean_wers.append(test_clean_wer)
            test_other_wers.append(test_other_wer)
            
            
        plot_metrics(train_losses, 'train.png', args)
        np.save(os.path.join(args.save_folder, 'train_losses'), train_losses)
        
        plot_metrics(dev_clean_losses, 'dev_clean.png', args)
        np.save(os.path.join(args.save_folder, 'dev_clean_losses'), dev_clean_losses)
        plot_metrics(dev_clean_wers, 'dev_clean_wer.png', args)
        np.save(os.path.join(args.save_folder, 'dev_clean_wers'), dev_clean_wers)
        
        plot_metrics(dev_other_losses, 'dev_other.png', args)
        np.save(os.path.join(args.save_folder, 'dev_other_losses'), dev_other_losses)
        plot_metrics(dev_other_wers, 'dev_other_wer.png', args)
        np.save(os.path.join(args.save_folder, 'dev_other_wers'), dev_other_wers)
        
        plot_metrics(test_clean_losses, 'test_clean.png', args)
        np.save(os.path.join(args.save_folder, 'test_clean_losses'), test_clean_losses)
        plot_metrics(test_clean_wers, 'test_clean_wer.png', args)
        np.save(os.path.join(args.save_folder, 'test_clean_wers'), test_clean_wers)
        
        plot_metrics(test_other_losses, 'test_other.png', args)
        np.save(os.path.join(args.save_folder, 'test_other_losses'), test_other_losses)
        plot_metrics(test_other_wers, 'test_other_wer.png', args)
        np.save(os.path.join(args.save_folder, 'test_other_wers'), test_other_wers)
            

        # save a checkpoint of ASR model with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best_test_clean.pth')
        model.load_state_dict(best_test_clean_model[0])
        save_model(model, optimizer, args, epoch, save_file)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_wer, _, _  = validate(test_clean_loader, model, criterion, args, best_test_clean)
    
    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_test_clean, path=os.path.join(args.save_dir, 'results_test_clean.json'))
    
if __name__ == '__main__':
    main()
