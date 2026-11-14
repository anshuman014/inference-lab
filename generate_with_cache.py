"""Greedy generation with GPT-2: no KV cache vs. KV cache, timed side by side.

Without the cache, every step feeds the whole sequence through the model.
With the cache, the prompt goes through once (prefill), then every step feeds
only the newest token and reuses the saved keys and values (decode).

Run:
    source ~/inference/bin/activate
    python generate_with_cache.py
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The cat sat on the"
new_tokens = 200
prompt_ids = tok(prompt, return_tensors="pt").input_ids
prompt_len = prompt_ids.shape[1]


def generate_no_cache(ids, n):
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.perf_counter()
            logits = model(ids, use_cache=False).logits       # whole sequence, every step
            next_id = logits[0, -1].argmax()
            times.append(time.perf_counter() - start)
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    return ids, times


def generate_with_cache(ids, n):
    times = []
    with torch.no_grad():
        # Prefill: the whole prompt goes through once and fills the cache.
        start = time.perf_counter()
        out = model(ids, use_cache=True)
        cache = out.past_key_values                           # saved keys and values, all layers
        next_id = out.logits[0, -1].argmax()
        times.append(time.perf_counter() - start)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)

        # Decode: only the newest token goes in; the cache supplies the rest.
        for _ in range(n - 1):
            start = time.perf_counter()
            out = model(next_id.view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values                       # now one token longer
            next_id = out.logits[0, -1].argmax()
            times.append(time.perf_counter() - start)
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    return ids, times


with torch.no_grad():
    model(prompt_ids, use_cache=False)                        # warm-up pass, not timed

ids_slow, t_slow = generate_no_cache(prompt_ids, new_tokens)
ids_fast, t_fast = generate_with_cache(prompt_ids, new_tokens)

print("same text from both versions:", torch.equal(ids_slow, ids_fast))
print()
print(f"{'step':>5} {'tokens so far':>14} {'no cache ms':>12} {'cache ms':>9}")
for step in range(0, new_tokens, 20):
    print(f"{step:>5} {prompt_len + step:>14} {t_slow[step] * 1000:>12.1f} {t_fast[step] * 1000:>9.1f}")

print()
print(f"last 10 steps, no cache: {sum(t_slow[-10:]) / 10 * 1000:6.1f} ms per token")
print(f"last 10 steps, cache:    {sum(t_fast[-10:]) / 10 * 1000:6.1f} ms per token")
print(f"total, no cache: {sum(t_slow):.2f} s")
print(f"total, cache:    {sum(t_fast):.2f} s  ({sum(t_slow) / sum(t_fast):.1f}x faster)")

# What the cache costs: a key and a value per layer for every token, 4 bytes per number.
total_tokens = prompt_len + new_tokens
cache_bytes = 2 * model.config.n_layer * model.config.n_embd * 4 * total_tokens
print(f"cache at the end: about {cache_bytes / 1e6:.1f} MB for {total_tokens} tokens")
