import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import transformers

class BertDataset(Dataset):
	
	def __init__(self, tokenizer, device = torch.device("cuda"), split = "train", dataset_no=0, max_length=100, mean=True, oversampling=1, oversampling_ratios=[18,1,2]):
		assert split in ["train","valid"], "split must be in [train,valid]"
		assert dataset_no in [0,1,2], "dataset_no must be in [0,1,2]"
		print(f"{split} DATASET")
		file_path = f"dataset/ClassificationDataset-{split}{dataset_no}.xlsx"
		self.device = device
		self.tokenizer = tokenizer
		self.max_length = max_length

		self.df = pd.read_excel(file_path, names=["label","text"]).sample(frac=1, random_state=0).dropna().reset_index(drop=True)
		labels = sorted(list(self.df["label"].unique()))
		print(f"LABELS - {labels}")
		label_dict = {label:ind for ind,label in enumerate(labels)}
		print(f"LABELS DICT- {label_dict}")

		# update labels
		self.df["label"] = torch.tensor(self.df["label"].apply(lambda x: label_dict[x]), dtype=torch.int64)

		# preprocessing
		self.df["text"] = self.df["text"].apply(lambda x: x.lower())

		label_counts = {label: len(self.df[self.df.label==label]) for label in label_dict.values()}
		print(f"Label Counts Before Oversampling: {label_counts} Total={len(self.df)}")
		# oversampling
		if oversampling:
			temp_dfs = []
			
			max_cnt = max([label_counts[label] for label in label_counts])
			for label in label_counts: 		
				curr_cnt = label_counts[label]
				n_repeat = oversampling_ratios[label]
				temp_dfs.append(self.df.loc[self.df[self.df.label == label].index.repeat(n_repeat)].reset_index(drop=True))
			
			self.df = pd.concat(temp_dfs).sample(frac=1, random_state=0).reset_index(drop=True)
			label_counts = {label: len(self.df[self.df.label==label]) for label in label_dict.values()}
			print(f"Label Counts After Oversampling: {label_counts} Total={len(self.df)}")

		# print frequency stats
		lens = torch.tensor([len(each["text"].split()) for i,each in self.df.iterrows()], dtype=torch.float32)
		print(f"{split} SENTENCES LENGTH - MAX: {torch.max(lens).item()} MIN: {torch.min(lens).item()} AVG: {torch.mean(lens).item()}")
		self.max_length = int(torch.max(lens).item())

	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		
		text1 = self.df.iloc[index]["text"]
		
		inputs = self.tokenizer.encode_plus(text1 ,None, pad_to_max_length=True, add_special_tokens=True,return_attention_mask=True, max_length=self.max_length)
		ids = inputs["input_ids"]
		token_type_ids = inputs["token_type_ids"]
		mask = inputs["attention_mask"]

		return {
			'ids': torch.tensor(ids, dtype=torch.long).to(self.device),
			'mask': torch.tensor(mask, dtype=torch.long).to(self.device),
			'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
			'target': torch.tensor(self.df.iloc[index]["label"], dtype=torch.long).to(self.device)
			}

if __name__ == "__main__":
	# testing dataloader
	tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
	valid_dataset= BertDataset(tokenizer, max_length=100, split="valid", oversampling=0)
	valid_dataloader=DataLoader(dataset=valid_dataset,batch_size=64)
	next(iter(valid_dataloader))