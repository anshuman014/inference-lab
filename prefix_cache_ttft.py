"""Prefix caching with GPT-2: prefill a shared system prompt once, then reuse it.

An AI agent sends many requests that all start with the same long system prompt
(its rules and tool definitions) and end with a short new part. Without prefix
caching, every request prefills the whole prompt again. With prefix caching, the
system prompt goes through the model once, its keys and values are kept, and
each request prefills only its own new tokens on top of them.

Why it is safe: the causal mask. A token's keys and values depend only on the
tokens before it, so the system prompt's K and V come out the same whatever
question follows it.

Three parts:
  1. TTFT with and without the prefix cache, for system prompts of 100 to 900 tokens
  2. one full answer both ways, to show the text is identical
  3. why the match must be exact: change one early token and every later
     position's keys change too, so nothing after it can be reused

Run:
    source ~/inference/bin/activate
    python prefix_cache_ttft.py
"""
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PREFIX_LENGTHS = [100, 300, 600, 900]   # system prompt tokens; GPT-2 holds at most 1,024
REPEATS = 3                             # time each request 3 times, keep the median
ANSWER_TOKENS = 20

SYSTEM_PROMPT = """You are Atlas, the operations agent for Fernwood Goods, a small online store that sells kitchenware. Today is Monday, 9 November 2026.

Rules:
1. Always look up an order with get_order before you change it.
2. Never refund more than the amount the customer paid.
3. If a customer is upset, apologise once, then fix the problem.
4. Ask before any action that cannot be undone, such as cancelling a shipped order.
5. Keep replies under 80 words. Use plain language and no jargon.
6. If a tool returns an error, try once more, then tell the customer what happened.

Tools. Call a tool by writing its name and JSON arguments on one line.

get_order: Look up one order.
  arguments: {"order_id": "string, five digits"}
  returns: {"order_id": "string", "status": "placed | packed | shipped | delivered | cancelled", "items": [{"sku": "string", "name": "string", "quantity": "integer", "price": "number"}], "total": "number", "address": "string", "placed_at": "date"}

list_orders: List orders that match a filter.
  arguments: {"status": "placed | packed | shipped | delivered | cancelled", "from_date": "date", "to_date": "date", "limit": "integer, at most 50"}
  returns: {"orders": [{"order_id": "string", "status": "string", "total": "number", "placed_at": "date"}]}

track_shipment: Find where a shipped order is.
  arguments: {"order_id": "string"}
  returns: {"carrier": "string", "tracking_number": "string", "last_scan": "string", "last_scan_at": "datetime", "expected_delivery": "date"}

refund_item: Refund one item from an order.
  arguments: {"order_id": "string", "sku": "string", "quantity": "integer", "reason": "damaged | missing | wrong_item | late | other"}
  returns: {"refund_id": "string", "amount": "number", "status": "pending | done"}

cancel_order: Cancel an order that has not shipped.
  arguments: {"order_id": "string", "reason": "string"}
  returns: {"cancelled": "boolean", "refund_amount": "number"}

update_address: Change the delivery address of an order that has not shipped.
  arguments: {"order_id": "string", "street": "string", "city": "string", "state": "string", "postcode": "string"}
  returns: {"updated": "boolean", "address": "string"}

check_stock: How many of an item are left.
  arguments: {"sku": "string", "warehouse": "oakland | reno | all"}
  returns: {"sku": "string", "name": "string", "on_hand": "integer", "reserved": "integer", "next_delivery": "date"}

find_product: Search the catalogue by name, colour or material.
  arguments: {"query": "string", "colour": "string", "max_price": "number"}
  returns: {"products": [{"sku": "string", "name": "string", "price": "number", "colour": "string"}]}

check_discount: Check whether a discount code is valid.
  arguments: {"code": "string"}
  returns: {"valid": "boolean", "percent_off": "integer", "expires_at": "date", "minimum_order": "number"}

send_email: Send an email to the customer on an order.
  arguments: {"order_id": "string", "subject": "string", "body": "string, at most 120 words"}
  returns: {"sent": "boolean", "message_id": "string"}

create_ticket: Hand a problem to a person on the support team.
  arguments: {"order_id": "string", "summary": "string", "priority": "low | normal | urgent"}
  returns: {"ticket_id": "string", "assigned_to": "string"}

Answer format: first think about which tool to call, then call it, then reply to the customer.

Examples of good replies:
- "Your order 10398 shipped on Thursday with UPS and should arrive tomorrow."
- "I have refunded 18.50 dollars for the cracked bowl. You will see it in 3 to 5 days."
- "That order has already shipped, so I cannot change the address. Shall I ask the carrier to hold it?"
"""

QUESTIONS = [
    "Where is order 10482? The customer says it has not arrived.",
    "Refund the second item in order 10417, it arrived broken.",
    "How many blue mugs do we have left in the Oakland warehouse?",
    "Change the shipping address on order 10501 to 22 Pine Street, Portland.",
    "Draft a short apology email for the late delivery of order 10399.",
    "Which orders from last week are still waiting to be packed?",
    "Is the discount code AUTUMN15 still valid?",
    "Cancel order 10522 before it ships.",
]


@torch.no_grad()
def prefill(model, ids, cache):
    """One forward pass over `ids`. Their keys and values are added to `cache`.
    Returns the scores for the next token."""
    out = model(ids, past_key_values=cache, use_cache=True)
    return out.logits[0, -1]


def build_prefix_cache(model, prefix_ids):
    """Done once: run the system prompt through the model and keep its keys and values."""
    cache = DynamicCache()
    prefill(model, prefix_ids, cache)
    return cache


def first_token_without_prefix_cache(model, prefix_ids, question_ids):
    """Every request starts from nothing and prefills the whole prompt, system prompt included."""
    ids = torch.cat([prefix_ids, question_ids], dim=1)
    return prefill(model, ids, DynamicCache())


def first_token_with_prefix_cache(model, prefix_cache, question_ids):
    """Prefill only the question, on top of the saved system prompt.

    The question's keys and values land in the same cache, so afterwards they are
    cut off again: the cache goes back to holding only the shared system prompt,
    ready for the next request.
    """
    prefix_len = prefix_cache.get_seq_length()
    scores = prefill(model, question_ids, prefix_cache)
    cut_back_to(prefix_cache, prefix_len)
    return scores


def cut_back_to(cache, length):
    """Forget every cached position from `length` on. (A negative number tells
    crop how many positions to remove; that form works in transformers 4 and 5.)"""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


def greedy_answer(model, cache, ids, n):
    """Prefill `ids` onto `cache`, then pick n tokens greedily. Returns their ids."""
    picked = []
    scores = prefill(model, ids, cache)
    while True:
        next_id = scores.argmax().view(1, 1)
        picked.append(next_id.item())
        if len(picked) == n:
            return picked
        scores = prefill(model, next_id, cache)            # decode: one token per pass


def last_layer_keys(cache):
    """The last layer's keys as one row per position: [positions, heads x head size]."""
    keys = cache.layers[-1].keys if hasattr(cache, "layers") else cache.key_cache[-1]
    return keys[0].transpose(0, 1).flatten(1)


def timed(fn, *args):
    """Run fn and pick the next token. Returns (token id, seconds)."""
    start = time.perf_counter()
    token = fn(*args).argmax().item()                      # TTFT ends when the token is picked
    return token, time.perf_counter() - start


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    system_ids = tok(SYSTEM_PROMPT, return_tensors="pt").input_ids
    assert system_ids.shape[1] >= max(PREFIX_LENGTHS), "the system prompt is too short for the longest prefix"
    questions = [tok(f"\n\nUser: {q}\nAtlas:", return_tensors="pt").input_ids for q in QUESTIONS]
    question_len = statistics.mean(q.shape[1] for q in questions)

    prefill(model, system_ids[:, :20], DynamicCache())    # warm-up pass, not timed

    # Part 1: TTFT with and without the prefix cache.
    print(f"Part 1: time to first token, {len(QUESTIONS)} requests per row, each timed {REPEATS} times (medians)")
    print(f"Every request = system prompt + a question of about {question_len:.0f} tokens\n")
    print(f"{'system prompt':>13} {'TTFT without':>13} {'TTFT with':>10} {'saved':>8} {'faster':>7} "
          f"{'built once':>11} {'same first token':>17}")
    print(f"{'tokens':>13} {'prefix cache':>13} {'prefix':>10} {'ms':>8} {'':>7} {'ms':>11}")
    for length in PREFIX_LENGTHS:
        prefix_ids = system_ids[:, :length]
        start = time.perf_counter()
        cache = build_prefix_cache(model, prefix_ids)       # the one-time cost
        built = time.perf_counter() - start

        without, with_, agree, worst = [], [], 0, 0.0
        for q in questions:
            a = first_token_without_prefix_cache(model, prefix_ids, q)
            b = first_token_with_prefix_cache(model, cache, q)
            agree += int(a.argmax() == b.argmax())
            worst = max(worst, (a - b).abs().max().item())
            for _ in range(REPEATS):
                without.append(timed(first_token_without_prefix_cache, model, prefix_ids, q)[1])
                with_.append(timed(first_token_with_prefix_cache, model, cache, q)[1])
            assert cache.get_seq_length() == length, "the cache should hold only the system prompt again"

        t_without, t_with = statistics.median(without) * 1000, statistics.median(with_) * 1000
        print(f"{length:>13} {t_without:>13.1f} {t_with:>10.1f} {t_without - t_with:>8.1f} "
              f"{t_without / t_with:>6.1f}x {built * 1000:>11.1f} {f'{agree} of {len(questions)}':>17}")
    print(f"\nLargest difference between the two ways' next-token scores: {worst:.1e} (rounding noise)")

    # Part 2: a whole answer both ways, with the longest system prompt.
    prefix_ids = system_ids[:, :max(PREFIX_LENGTHS)]
    q = questions[0]
    plain = greedy_answer(model, DynamicCache(), torch.cat([prefix_ids, q], dim=1), ANSWER_TOKENS)
    cache = build_prefix_cache(model, prefix_ids)
    reused = greedy_answer(model, cache, q, ANSWER_TOKENS)
    print(f"\nPart 2: one answer, {ANSWER_TOKENS} tokens, to: {QUESTIONS[0]!r}")
    print(f"  without prefix cache: {tok.decode(plain)!r}")
    print(f"  with prefix cache:    {tok.decode(reused)!r}")
    print(f"  identical: {plain == reused}")

    # Part 3: why only an exact prefix can be reused.
    changed = tok(SYSTEM_PROMPT.replace("Monday", "Tuesday", 1), return_tensors="pt").input_ids
    n = 100
    original_keys = last_layer_keys(build_prefix_cache(model, system_ids[:, :n]))
    changed_keys = last_layer_keys(build_prefix_cache(model, changed[:, :n]))
    differs = (original_keys - changed_keys).abs().max(dim=1).values > 1e-4
    first = int(differs.nonzero()[0])
    print(f"\nPart 3: change 'Monday' to 'Tuesday' in the system prompt (token position {first} of {n})")
    print(f"  last-layer keys at positions 0 to {first - 1}: {'identical' if not differs[:first].any() else 'changed'}")
    print(f"  last-layer keys at positions {first} to {n - 1}: {int(differs[first:].sum())} of {n - first} changed")
    print("  So a cache built for one date cannot be reused for the other after that token.")


if __name__ == "__main__":
    main()
