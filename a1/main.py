from embeddings import GloveEmbedding, FastTextEmbedding
from models.dan import DAN 
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
 
from tqdm import tqdm
import pandas as pd
import torch
from dataset.dataloader import ClassificationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

fasttext = FastTextEmbedding()
glove = GloveEmbedding('wikipedia_gigaword', d_emb=300, show_progress=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

emb_model = fasttext

BATCH_SIZE = 64
train_dataset = ClassificationDataset(emb_model, split="train", dataset_no=0)
test_dataset = ClassificationDataset(emb_model, split="valid", dataset_no=0)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)




nn_model = DAN().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nn_model.parameters(),lr=0.01,momentum=0.9)

running_loss = 0.
last_loss = 0.

train_losses = []
val_losses = []
val_accs = []
EPOCHS = 100

tq = tqdm(range(EPOCHS))
for epoch, _ in enumerate(tq):
    for X_train, y_train in train_dataloader:
        # Every data instance is an input + label pair
        inputs, labels = X_train, y_train
        # import pdb; pdb.set_trace()
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = nn_model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
    val_accuracy = nn_model.get_accuracy(nn_model(test_dataset.X), test_dataset.y)    
    tq.set_description_str(f"EPOCH: {epoch+1} LOSS: {BATCH_SIZE*running_loss/len(train_dataset)} Val Accuracy: {val_accuracy} Val Loss: {loss_fn(nn_model(test_dataset.X), test_dataset.y)}")
    

    train_losses.append(BATCH_SIZE*running_loss/len(train_dataset))
    val_losses.append(loss_fn(nn_model(test_dataset.X), test_dataset.y))
    val_accs.append(val_accuracy)
    
    running_loss = 0



plt.figure()
plt.plot(torch.tensor(train_losses).to("cpu"), label="Train loss")
plt.plot(torch.tensor(val_losses).to("cpu"), label="Val loss")
plt.legend()
plt.savefig(f"plots/losses_fasttext_{EPOCHS}_{BATCH_SIZE}.pdf")

