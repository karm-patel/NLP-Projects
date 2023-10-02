import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io
from nltk.tokenize import word_tokenize
import re

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset.dataloader import DateDataset
from models.enc_dec import EncoderRNN, DecoderRNN
from train_utils import asMinutes, timeSince, train_epoch, val_epoch

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
gpu_device, cpu_device
n_epochs = 200
LR = 0.005

train_dataset = DateDataset(split="train")
BOS_IDX = train_dataset.voc["<bos>"]
EOS_IDX = train_dataset.voc["<eos>"]
PAD_IDX = train_dataset.voc["<pad>"]

def generate_batch(data_batch):
    s_batch, t_batch = [], []
    for (s_item, t_item) in data_batch:
        s_batch.append(torch.cat([torch.tensor([BOS_IDX]), s_item, torch.tensor([EOS_IDX])], dim=0))
        t_batch.append(torch.cat([torch.tensor([BOS_IDX]), t_item, torch.tensor([EOS_IDX])], dim=0))
        
    s_batch = pad_sequence(s_batch, padding_value=PAD_IDX)
    return s_batch.T.to(gpu_device), torch.stack(t_batch).to(gpu_device)

train_dataloader = DataLoader(train_dataset, batch_size=18000, collate_fn=generate_batch)

test_dataset = DateDataset(split="test")
test_dataloader = DataLoader(test_dataset, batch_size=4000, collate_fn=generate_batch)


vocab_size = len(train_dataset.voc)
enc_hidden_size = 256
encoder = EncoderRNN(vocab_size, enc_hidden_size).to(gpu_device)
decoder = DecoderRNN(enc_hidden_size, vocab_size).to(gpu_device)


encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
criterion = nn.NLLLoss()

import time
start = time.time()
plot_losses, val_losses, val_accs = [], [], []
print_loss_total = 0  # Reset every print_every
val_loss_total = 0

print_every=1
tq_obj = tqdm(range(1,n_epochs+1))

for epoch in tq_obj:
    loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    
    val_loss, val_acc = val_epoch(test_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    val_loss_total += val_loss
    val_accs.append(val_acc.to(cpu_device))


    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        plot_losses.append(print_loss_avg)
        
        val_loss_avg = val_loss_total/print_every
        val_losses.append(val_loss_avg)
        val_loss_total = 0
        
        tq_obj.set_description_str(f"train loss: {round(print_loss_avg, 3)} val loss: {round(val_loss_avg, 3)} val acc: {val_acc} Time: {(timeSince(start, epoch / n_epochs))}")

plt.figure()
plt.plot(plot_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.savefig(f"plots/enc-dec-simple-losses-{n_epochs}.pdf")

plt.figure()
plt.plot(val_accs, label="val")
plt.legend()
plt.savefig(f"plots/enc-dec-simple-acc-{n_epochs}.pdf")