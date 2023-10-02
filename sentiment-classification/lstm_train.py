import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D", default=0, type=int)
parser.add_argument("-E", default=100, type=int)
parser.add_argument("-B", default=128, type=int)
parser.add_argument("-H", default=400, type=int) #hidden-size
args = parser.parse_args()

import os
from embeddings import GloveEmbedding, FastTextEmbedding
from models.lstm import LSTM
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
 
from tqdm import tqdm
import sklearn
import torch
from dataset.dataloader_lstm import ClassificationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# %reload_ext autoreload
# %autoreload 2

cpu_device = torch.device("cpu")

# import argparse
# parser = argparse.ArgumentParser()
# parser
# parser.parse_args()


# Embeddings =================================================================
fasttext = FastTextEmbedding()
glove = GloveEmbedding('wikipedia_gigaword', d_emb=300, show_progress=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

emb_model = fasttext

# Global Variables =================================================================

BATCH_SIZE = args.B
DATASET_NO = args.D
EPOCHS = args.E
HIDDEN_SIZE = args.H
classes_dict = {0:3, 1:2, 2:5} # DATASET_NO: N_Classes
N_CLASSES = classes_dict[DATASET_NO]
print("N Classes", N_CLASSES)

# Dataloader =================================================================

train_dataset = ClassificationDataset(emb_model, split="train", dataset_no=DATASET_NO)
test_dataset = ClassificationDataset(emb_model, split="valid", dataset_no=DATASET_NO)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset)//4)

# Model & Train=================================================================

nn_model = LSTM(n_classes=N_CLASSES, emb_size=300, hidden_size=HIDDEN_SIZE).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(nn_model.parameters(), lr=0.0005)


running_loss = 0.
last_loss = 0.

metrics = {
"train_losses" : [],
"val_losses" : [],
"val_accs" : [],
"val_f1_score" : []
}

train_losses = []
val_losses = []
val_accs = []

tq = tqdm(range(EPOCHS))

for epoch, _ in enumerate(tq):
    for X_train, y_train in train_dataloader:
        # Every data instance is an input + label pair
        # inputs, labels = X_train, y_train
        h = torch.tensor(np.random.uniform(low = float(torch.min(X_train)), high = float(torch.max(X_train)), size = (1,len(y_train), HIDDEN_SIZE)), dtype=X_train.dtype).to(device)
        c = torch.tensor(np.random.uniform(low = float(torch.min(X_train)), high = float(torch.max(X_train)), size = (1,len(y_train), HIDDEN_SIZE)), dtype=X_train.dtype).to(device)
        inputs, labels = (X_train, h, c), y_train

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
    
    metrics["train_losses"].append(BATCH_SIZE*running_loss/len(train_dataset))

    # Testing metrics
    with torch.no_grad():
        test_logits = []
        for (X_test, y_test) in test_dataloader:
            h = torch.tensor(np.random.uniform(low = float(torch.min(X_test)), high = float(torch.max(X_test)), size = (1,len(y_test), HIDDEN_SIZE)), dtype=X_test.dtype).to(device)
            c = torch.tensor(np.random.uniform(low = float(torch.min(X_test)), high = float(torch.max(X_test)), size = (1,len(y_test), HIDDEN_SIZE)), dtype=X_test.dtype).to(device)
            test_logit = nn_model((X_test, h, c))
            test_logits.append(test_logit)
        
        y_test = test_dataset.y
        test_logits = torch.cat(test_logits)

        y_pred = torch.argmax(test_logits, axis=-1)
        metrics["val_losses"].append(loss_fn(test_logits, y_test))
        
        val_accuracy = nn_model.get_accuracy(y_test, y_pred)
        metrics["val_accs"].append(val_accuracy)

        f1_score = nn_model.get_f1_score(y_test, y_pred)
        metrics["val_f1_score"].append(f1_score)

            

    tq.set_description_str(f"EPOCH: {epoch+1} LOSS: {BATCH_SIZE*running_loss/len(train_dataset)} F1-Score: {f1_score} Val Loss: {loss_fn(test_logits, y_test)} Val Accuracy: {val_accuracy} ")

    running_loss = 0


# Post Analysis =================================================================

model_name = "LSTM"
plt.figure()
plt.plot(torch.tensor(metrics["train_losses"]).to("cpu"), label="Train loss")
plt.plot(torch.tensor(metrics["val_losses"]).to("cpu"), label="Val loss")
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.legend()
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_losses_fasttext_D{DATASET_NO}.pdf")

plt.figure()
plt.plot(torch.tensor(metrics["val_accs"]).to("cpu"), label="Val-Acc")
plt.legend()
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_accs_fasttext_D{DATASET_NO}.pdf")

plt.figure()
plt.plot(torch.tensor(metrics["val_f1_score"]).to("cpu"), label="Val-F1-score")
plt.legend()
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_f1_score_fasttext_D{DATASET_NO}.pdf")

print(f'BEST VAL ACCURACY - {max(metrics["val_accs"])}')
print(f'BEST VAL F1-SCORE - {max(metrics["val_f1_score"])}')

cf = sklearn.metrics.confusion_matrix(y_test.to(cpu_device), y_pred.to(cpu_device))
print(cf)

# save the resulyts
if not os.path.exists("results/"):
    os.makedirs("results/")

with open(f"results/LSTM_{DATASET_NO}_{EPOCHS}_{BATCH_SIZE}.txt", "w") as fp:
    fp.write(f"Class Counts: {test_dataset.y.unique(return_counts=True)[1]}\n")
    fp.write(f"Class Proportion: {100*test_dataset.y.unique(return_counts=True)[1]/float(len(test_dataset.y))}\n")
    fp.write(f'BEST VAL ACCURACY - {max(metrics["val_accs"])}\n')
    fp.write(f'BEST VAL F1-SCORE - {max(metrics["val_f1_score"])}\n')
    fp.write(f'{cf}')