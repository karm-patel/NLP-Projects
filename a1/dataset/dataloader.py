import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import pandas as pd
import string

class ClassificationDataset(Dataset):
	
	def __init__(self, emb_model, device = torch.device("cuda"), split = "train", dataset_no=0, embed_dim=300, mean=True, oversampling=1):
		assert split in ["train","valid"], "split must be in [train,valid]"
		assert dataset_no in [0,1,2], "dataset_no must be in [0,1,2]"
		
		file_path = f"dataset/ClassificationDataset-{split}{dataset_no}.xlsx"

		df = pd.read_excel(file_path, names=["label","text"]).sample(frac=1, random_state=0).dropna().reset_index(drop=True)
		labels = sorted(list(df["label"].unique()))
		print(f"LABELS - {labels}")
		label_dict = {label:ind for ind,label in enumerate(labels)}
		print(f"LABELS DICT- {label_dict}")

		# preprocessing
		df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
		data = df["text"].apply(lambda x: word_tokenize(x)).to_list()

		# debug
		self.X = []
		for sent in data:
			if sent:
				embs = []
				for word in sent:
					e = emb_model.emb(word.lower(), default="random")
					# if torch.isnan(torch.tensor(e).mean(axis=0)):
					# 	print(word)
					embs.append(e)
			else:
				embs.append(torch.zeros(embed_dim))
			if mean:
				self.X.append(torch.tensor(embs).mean(axis = 0))
			else:
				self.X.append(torch.tensor(embs))

		# Padding 
		if not mean:
			max_len = max(len(sentence) for sentence in self.X)
			for ind, sentence in enumerate(self.X):
				padded_sentence = torch.full((max_len, embed_dim), 0, dtype=sentence.dtype)
				padded_sentence[:len(sentence)] = sentence
				self.X[ind] = padded_sentence
		
		self.X = torch.stack(self.X).to(device)
		self.y = torch.tensor(df["label"].apply(lambda x: label_dict[x])).to(device)

		# oversampling
		if split != "valid" and oversampling:
			class_counts = torch.unique(self.y, return_counts=True)[1]
			max_count = torch.max(class_counts).item()
			new_X, new_y = [], []
			
			for label,_ in enumerate(labels):
				cnt = class_counts[label]
				r = 1
				if cnt == max_count:
					r = 1
				X_i = self.X[self.y == label].repeat(((r*max_count)//cnt,1,1))
				y_i = self.y[self.y == label].repeat(((r*max_count)//cnt))
				print(label, X_i.shape)
				new_X.append(X_i)
				new_y.append(y_i)
			
			self.X = torch.cat(new_X)
			self.y = torch.cat(new_y)
	
	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]
