import torch
from safetensors.torch import load_file
from models.moeut import MoEUT
import sentencepiece as spm

# 1. Setup Model
model = MoEUT()
model.load_state_dict(load_file('moeut.safetensors'), strict=False)
model.to("cuda")

# CRITICAL: Put the model in evaluation mode to freeze Dropout/BatchNorm
model.eval() 

# 2. Setup Tokenizer & Input
tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/c4_bpe8k.model")
token_ids = tokenizer.encode("The capital of France is ")
# Note: .detach() isn't strictly necessary here if you aren't passing it to an optimizer, 
# but it doesn't hurt.
input = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0).to("cuda")

# 3. Generation Loop
# Use inference_mode() instead of no_grad() for maximum performance
with torch.inference_mode():
    for _ in range(6):
        logits = model(input)
        greedy_token_id = torch.argmax(logits[0][-1])
        input = torch.cat([input, greedy_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

print(input)
print(tokenizer.decode(input.squeeze(0).tolist()))