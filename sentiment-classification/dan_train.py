import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D", default=0, type=int)
parser.add_argument("-E", default=100, type=int)
parser.add_argument("-B", default=128, type=int)
args = parser.parse_args()

import os
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
import sklearn

# %reload_ext autoreload
# %autoreload 2

cpu_device = torch.device("cpu")
EMB_MODEL ="Glove"
if EMB_MODEL == "fast-text":
    emb_model = FastTextEmbedding()
else:
    emb_model = GloveEmbedding('wikipedia_gigaword', d_emb=300, show_progress=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

BATCH_SIZE = args.B
DATASET_NO = args.D
EPOCHS = args.E
AVG_F1 = "micro" 
OVERSAMPLING = False
WEIGHT = torch.tensor((5,1,1), dtype=torch.float32).to(device)

classes_dict = {0:3, 1:2, 2:5} # DATASET_NO: N_Classes
N_CLASSES = classes_dict[DATASET_NO]
train_dataset = ClassificationDataset(emb_model, split="train", dataset_no=DATASET_NO, oversampling=OVERSAMPLING, oversampling_ratio=0.5)
test_dataset = ClassificationDataset(emb_model, split="valid", dataset_no=DATASET_NO)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("N Classes: ", N_CLASSES)
print("Class Counts: ", test_dataset.y.unique(return_counts=True)[1])
print("Class Proportion: ", 100*test_dataset.y.unique(return_counts=True)[1]/float(len(test_dataset.y)))

# nn_model = DAN(n_classes=N_CLASSES).to(device)
nn_model = DAN(n_classes=N_CLASSES, drop_prob=0).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=WEIGHT)
optimizer = optimizer = torch.optim.Adam(nn_model.parameters())
# optimizer = optimizer = torch.optim.SGD(nn_model.parameters(), lr = 0.05)

running_loss = 0.
last_loss = 0.

metrics = {
"train_losses" : [],
"val_losses" : [],
"val_accs" : [],
"val_f1_score" : [],
"val_macro_f1_score": [],
"val_weighted_f1_score": []
}

train_losses = []
val_losses = []
val_accs = []

tq = tqdm(range(EPOCHS))

for epoch, _ in enumerate(tq):
    for X_train, y_train in train_dataloader:
 
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
    with torch.no_grad():
        test_logits = nn_model(test_dataset.X)
        y_pred = torch.argmax(test_logits, axis=-1)
        metrics["train_losses"].append(BATCH_SIZE*running_loss/len(train_dataset))
        metrics["val_losses"].append(loss_fn(test_logits, test_dataset.y))
        val_accuracy = nn_model.get_accuracy(test_dataset.y, y_pred)
        metrics["val_accs"].append(val_accuracy)

        
        f1_score = nn_model.get_f1_score(test_dataset.y, y_pred, average=AVG_F1)
        metrics["val_f1_score"].append(f1_score)

        macro_f1_score = nn_model.get_f1_score(test_dataset.y, y_pred, average="macro")
        metrics["val_macro_f1_score"].append(macro_f1_score)

        macro_f1_score = nn_model.get_f1_score(test_dataset.y, y_pred, average="weighted")
        metrics["val_weighted_f1_score"].append(macro_f1_score)

        tq.set_description_str(f"EPOCH: {epoch+1} LOSS: {BATCH_SIZE*running_loss/len(train_dataset)} {AVG_F1} F1-Score: {f1_score} Val Loss: {loss_fn(test_logits, test_dataset.y)} Val Accuracy: {val_accuracy} ")


    running_loss = 0

model_name = "DAN"
# save the plots
for d in range(3):
    if not os.path.exists("plots/D{d}"):
        os.makedirs("plots/D{d}")

plt.figure()
plt.plot(torch.tensor(metrics["train_losses"]).to("cpu"), label="Train loss")
plt.plot(torch.tensor(metrics["val_losses"]).to("cpu"), label="Val loss")
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.legend()
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_losses_{EMB_MODEL}_D{DATASET_NO}.pdf")

plt.figure()
plt.plot(torch.tensor(metrics["val_accs"]).to("cpu"), label="Val-Acc")
plt.legend()
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_accs_{EMB_MODEL}_D{DATASET_NO}.pdf")

plt.figure()
plt.plot(torch.tensor(metrics["val_f1_score"]).to("cpu"), label="Val-F1-score")
plt.legend()
plt.title(f"Epochs - {EPOCHS} Batch Size - {BATCH_SIZE}")
plt.savefig(f"plots/D{DATASET_NO}/{model_name}_val_f1_score_{EMB_MODEL}_D{DATASET_NO}.pdf")

print(f'BEST VAL ACCURACY - {max(metrics["val_accs"])}')
print(f'BEST VAL {AVG_F1} F1-SCORE - {max(metrics["val_f1_score"])}')

cf = sklearn.metrics.confusion_matrix(test_dataset.y.to(cpu_device), y_pred.to(cpu_device))
print(cf)

# save the results
if not os.path.exists("results/"):
    os.makedirs("results/")

with open(f"results/{model_name}_{EMB_MODEL}_{DATASET_NO}_{EPOCHS}_{BATCH_SIZE}.txt", "a") as fp:
    fp.write(f"OVERSAMPLING:  {OVERSAMPLING} \n")
    fp.write(f"WEIGHT: {WEIGHT} \n")
    fp.write(f"Class Counts: {test_dataset.y.unique(return_counts=True)[1]}\n")
    fp.write(f"Class Proportion: {100*test_dataset.y.unique(return_counts=True)[1]/float(len(test_dataset.y))}\n")
    fp.write(f'BEST VAL ACCURACY - {max(metrics["val_accs"])}\n')
    fp.write(f'BEST VAL Micro-F1-SCORE - {max(metrics["val_f1_score"])}\n')
    fp.write(f'BEST VAL MAcro-F1-SCORE - {max(metrics["val_macro_f1_score"])}\n')
    fp.write(f'BEST VAL Weighted-F1-SCORE - {max(metrics["val_weighted_f1_score"])}\n')
    fp.write(f'{cf}')
    fp.write("==================================================\n\n")