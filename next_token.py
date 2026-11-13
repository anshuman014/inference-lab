"""One forward pass through GPT-2: tokenize, score, print the top-5 next tokens.

Run inside your environment:
    source ~/inference/bin/activate
    python next_token.py
The first run downloads GPT-2 small (about 500 MB).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The cat sat on the"

# 1. Text -> token IDs
ids = tok(prompt, return_tensors="pt").input_ids
print("token IDs:", ids[0].tolist())
print("tokens:   ", [tok.decode(i) for i in ids[0]])

# 2. One forward pass through all 12 layers
with torch.no_grad():
    logits = model(ids).logits
print("logits shape:", tuple(logits.shape))  # (batch, tokens, vocabulary)

# 3. Keep only the last position: the scores for the token after "the"
last = logits[0, -1]

# 4. Logits -> probabilities, then show the top 5 guesses
probs = torch.softmax(last, dim=-1)
print("probabilities sum to:", round(probs.sum().item(), 4))
top = torch.topk(probs, 5)
for p, i in zip(top.values, top.indices):
    print(f"{tok.decode(i)!r:>10}  logit={last[i].item():6.2f}  prob={p.item():.1%}")

# 5. Greedy pick
print("greedy next token:", repr(tok.decode(last.argmax())))
