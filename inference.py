import torch
from safetensors.torch import load_file
from models.dense_transformer import DenseTransformer
import sentencepiece as spm

model = DenseTransformer()
model.load_state_dict(load_file('dense.safetensors'), strict=False)

tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/c4_bpe8k.model")
token_ids = tokenizer.encode("The capital of France is ")
input = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0)

model.to("cuda")
input = input.to("cuda").detach()

with torch.no_grad():
  for _ in range(6):
    logits = model(input)
    greedy_token_id = torch.argmax(logits[0][-1])
    input = torch.cat([input, greedy_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

print(input)
print(tokenizer.decode(input.squeeze(0).tolist()))

# The capital of France is the capital of the capital of