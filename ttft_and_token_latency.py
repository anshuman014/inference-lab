"""Time to first token (TTFT) and inter-token latency (ITL) with GPT-2.

One request has two phases:
  prefill: the whole prompt goes through the model in one pass and fills the cache
  decode:  one new token per pass, reusing the cache

This script times both phases for a short, a medium and a long prompt, so you
can see which metric the prompt length moves and which it leaves alone.

Run:
    source ~/inference/bin/activate
    python ttft_and_token_latency.py
"""
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

NEW_TOKENS = 50
PROMPT_LENGTHS = [5, 200, 800]      # in tokens; GPT-2 holds at most 1,024
REPEATS = 3                         # time each prompt 3 times, keep the median

# One long piece of text to cut prompts of any length from.
text = "The cat sat on the mat and looked out of the window at the rain. " * 60
all_ids = tok(text, return_tensors="pt").input_ids
assert all_ids.shape[1] >= max(PROMPT_LENGTHS), "the text is too short for the longest prompt"


def timed_generate(prompt_ids, n):
    """Greedy generation with the KV cache, timed.

    Returns (ttft, gaps):
      ttft: seconds from the start until the first new token is picked
      gaps: a list of seconds, one for each later token (n - 1 of them)
    """
    gaps = []
    with torch.no_grad():
        # Prefill: the whole prompt goes through once and fills the cache.
        start = time.perf_counter()
        out = model(prompt_ids, use_cache=True)
        cache = out.past_key_values
        next_id = out.logits[0, -1].argmax()
        ttft = time.perf_counter() - start

        # Decode: only the newest token goes in; the cache supplies the rest.
        for _ in range(n - 1):
            start = time.perf_counter()
            out = model(next_id.view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[0, -1].argmax()
            gaps.append(time.perf_counter() - start)
    return ttft, gaps


with torch.no_grad():
    model(all_ids[:, :5], use_cache=True)                     # warm-up pass, not timed

print(f"{NEW_TOKENS} new tokens per request, median of {REPEATS} runs")
print()
print(f"{'prompt tokens':>13} {'TTFT ms':>9} {'ITL ms':>8} {'tokens/s':>9} {'total ms':>9} {'prefill ms per prompt token':>28}")
for length in PROMPT_LENGTHS:
    prompt_ids = all_ids[:, :length]
    runs = [timed_generate(prompt_ids, NEW_TOKENS) for _ in range(REPEATS)]

    ttft = statistics.median(run[0] for run in runs)                      # seconds
    itl = statistics.median(statistics.mean(run[1]) for run in runs)      # seconds per token

    tokens_per_s = 1 / itl                                                # while the text streams
    total = ttft + (NEW_TOKENS - 1) * itl                                 # first token, then n - 1 gaps

    print(f"{length:>13} {ttft * 1000:>9.1f} {itl * 1000:>8.1f} {tokens_per_s:>9.0f} "
          f"{total * 1000:>9.0f} {ttft * 1000 / length:>28.2f}")
