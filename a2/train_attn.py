import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset.dataloader import DateDataset, get_dataloader
from models.enc_dec_attention import EncoderRNN, DecoderRNN
from train_utils import asMinutes, timeSince, train_epoch, val_epoch

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
gpu_device, cpu_device
n_epochs = 75
LR = 0.005
enc_hidden_size = 256

train_dataset, train_dataloader = get_dataloader(split="train", batch_size=18000)
test_dataset, test_dataloader = get_dataloader(split="test", batch_size=4000)

vocab_size = len(train_dataset.voc)
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

ckpt_name = f"attntion-{LR}"
torch.save(encoder.state_dict(), f"ckpts/{ckpt_name}-enc.pt")
torch.save(decoder.state_dict(), f"ckpts/{ckpt_name}-dec.pt")


plt.figure()
plt.plot(plot_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.savefig(f"plots/enc-dec-attn-losses-{n_epochs}.pdf")

plt.figure()
plt.plot(val_accs, label="val")
plt.legend()
plt.savefig(f"plots/enc-dec-attn-acc-{n_epochs}.pdf")