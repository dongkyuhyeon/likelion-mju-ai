import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.manual_seed(60212178)   

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"데이터셋 총 문자 수: {len(text):,}")

batch_size     = 1
output_dim     = 8
context_length = 4
stride         = 4

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text)
vocab_size = tokenizer.n_vocab        

print(f"토큰 수: {len(token_ids):,}")
print(f"어휘 크기: {vocab_size:,}")

class GPTDatasetV1(Dataset):
    """슬라이딩 윈도우로 (입력, 타겟) 쌍을 생성하는 Dataset."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


dataset = GPTDatasetV1(text, tokenizer, max_length=context_length, stride=stride)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"\n데이터셋 샘플 수: {len(dataset):,}")
print(f"배치 수: {len(dataloader):,}")

token_embedding_layer    = nn.Embedding(vocab_size, output_dim)
position_embedding_layer = nn.Embedding(context_length, output_dim)

pos_ids = torch.arange(context_length)

inputs, targets = next(iter(dataloader))   # shape: (batch_size, context_length)

token_embeddings    = token_embedding_layer(inputs)     # (B, T, D)
position_embeddings = position_embedding_layer(pos_ids) # (T, D) → 자동 브로드캐스트

input_embeddings = token_embeddings + position_embeddings  # (B, T, D)

print("\n### 첫 번째 배치 입력 임베딩 결과 ###")
print(f"입력 텐서 형태 (Batch, Context, Embedding): {input_embeddings.shape}")
print("\n첫 번째 배치의 입력 임베딩 값:")
print(input_embeddings)