import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import urllib.request
import os

torch.manual_seed(60232849)

batch_size     = 1
output_dim     = 8
context_length = 4
stride         = 4

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    dataset    = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


dataloader = create_dataloader_v1(
    text,
    batch_size=batch_size,
    max_length=context_length,
    stride=stride,
    shuffle=False,
    drop_last=True,
)

data_iter       = iter(dataloader)
inputs, targets = next(data_iter)

token_embedding_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings      = token_embedding_layer(inputs)

pos_embedding_layer = nn.Embedding(context_length, output_dim)
pos_embeddings      = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings

print("### 첫 번째 배치 입력 임베딩 결과 ###")
print(f"입력 텐서 형태 (Batch, Context, Embedding): {input_embeddings.shape}")
print("\n첫 번째 배치의 입력 임베딩 값:")
print(input_embeddings)