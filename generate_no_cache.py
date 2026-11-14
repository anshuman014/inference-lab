"""Greedy text generation with GPT-2, written by hand, with no KV cache.

Every step feeds the whole sequence back through the model, takes the
highest-scoring next token, appends it, and repeats. Each step is timed
so you can see the cost grow as the sequence gets longer.

Run:
    source ~/inference/bin/activate
    python generate_no_cache.py
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The cat sat on the"
new_tokens = 200

ids = tok(prompt, return_tensors="pt").input_ids   # shape (1, 5)
prompt_len = ids.shape[1]
times = []

with torch.no_grad():
    model(ids, use_cache=False)                    # warm-up pass, not timed

    for step in range(new_tokens):
        start = time.perf_counter()

        # Run the whole sequence so far through the model (no KV cache).
        logits = model(ids, use_cache=False).logits

        # Greedy pick: the highest score at the last position.
        next_id = logits[0, -1].argmax()

        times.append(time.perf_counter() - start)

        # Append the new token and go around again.
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)

print(tok.decode(ids[0]))
print()
print(f"{'step':>5} {'tokens in':>10} {'ms':>8}")
for step in range(0, new_tokens, 20):
    print(f"{step:>5} {prompt_len + step:>10} {times[step] * 1000:>8.1f}")

first = sum(times[:10]) / 10 * 1000
last = sum(times[-10:]) / 10 * 1000
print()
print(f"average of first 10 steps: {first:.1f} ms")
print(f"average of last 10 steps:  {last:.1f} ms  ({last / first:.1f}x slower)")
print(f"total time: {sum(times):.2f} s for {new_tokens} tokens")
