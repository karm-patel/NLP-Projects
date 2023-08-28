import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import pandas as pd

class ClassificationDataset(Dataset):
	
	def __init__(self, emb_model, device = torch.device("cuda"), split = "train", dataset_no=0):
		assert split in ["train","valid"], "split must be in [train,valid]"
		assert dataset_no in [0,1,2], "dataset_no must be in [0,1,2]"
		
		file_path = f"dataset/ClassificationDataset-{split}{dataset_no}.xlsx"

		df = pd.read_excel(file_path, names=["label","text"])
		labels = list(df["label"].unique())
    
		data = df["text"].apply(lambda x: word_tokenize(x)).to_list()
		self.X = torch.tensor(np.array([np.array([emb_model.emb(word.lower(), default="random") for word in sent]).astype(np.float32).mean(axis=0) for i,sent in enumerate(data)])).to(device)
		self.y = torch.tensor(df["label"].apply(lambda x: labels.index(x))).to(device)

	
	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]
