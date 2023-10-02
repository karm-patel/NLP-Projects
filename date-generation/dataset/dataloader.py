import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchtext.vocab import vocab
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
gpu_device = torch.device("cuda")

# todos 1) Make generalize tokenizer 2) make preprocessing different so we can use while custom dataset
class DateDataset(Dataset):
    def __init__(self, DATASET_PATH = "dataset/Assignment2aDataset.txt", split="train"):
        df = pd.read_csv(DATASET_PATH, names = ["source", "target"])
        df["source"] = df["source"].apply(lambda x: x.strip()[1:-1].replace("/", "-"))
        df["target"] = df["target"].apply(lambda x: x.strip()[1:-1])
        df_train, df_test = train_test_split(df, random_state=42, test_size=0.1)
        
        # tokenize
        en_tokenizer = get_tokenizer('spacy', language='en')   
        counter = Counter() # dict of {token: Freq}     
        for source in df["source"]:
            counter.update(en_tokenizer(source))

        for source in df["target"]:
            counter.update(en_tokenizer(source))
        
        voc = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])    
        
        # create data
        if split == "train":
            self.data_df = df_train
        else:
            self.data_df = df_test
            
        data = []
        for (source, target) in zip(self.data_df["source"], self.data_df["target"]):
            s_tensor_ = torch.tensor([voc[token] for token in en_tokenizer(source)])
            t_tensor_ = torch.tensor([voc[token] for token in en_tokenizer(target)])
            data.append((s_tensor_, t_tensor_))
        
        self.voc = voc
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]\
        

def generate_batch(data_batch, special_tokens):
    BOS_IDX = special_tokens["BOS_IDX"]
    PAD_IDX = special_tokens["PAD_IDX"]
    EOS_IDX = special_tokens["EOS_IDX"]

    s_batch, t_batch = [], []
    for (s_item, t_item) in data_batch:
        s_batch.append(torch.cat([torch.tensor([BOS_IDX]), s_item, torch.tensor([EOS_IDX])], dim=0))
        t_batch.append(torch.cat([torch.tensor([BOS_IDX]), t_item, torch.tensor([EOS_IDX])], dim=0))
        
    s_batch = pad_sequence(s_batch, padding_value=PAD_IDX)
    return s_batch.T.to(gpu_device), torch.stack(t_batch).to(gpu_device)

def get_dataloader(split="train", batch_size=4000):
    dataset = DateDataset(split=split)
    special_tokens = {}
    special_tokens["BOS_IDX"] = dataset.voc["<bos>"]
    special_tokens["EOS_IDX"] = dataset.voc["<eos>"]
    special_tokens["PAD_IDX"] = dataset.voc["<pad>"]

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch, 
                            sp_tokens = special_tokens : generate_batch(batch, sp_tokens))

    return dataset, dataloader

