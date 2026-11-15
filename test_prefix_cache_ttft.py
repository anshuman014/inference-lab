"""Checks for prefix_cache_ttft.py, on a tiny random GPT-2 so nothing is downloaded.

Run:
    python test_prefix_cache_ttft.py
"""
import torch
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel

from prefix_cache_ttft import (build_prefix_cache, first_token_with_prefix_cache,
                               first_token_without_prefix_cache, greedy_answer, last_layer_keys)

torch.manual_seed(0)
MODEL = GPT2LMHeadModel(GPT2Config(n_layer=3, n_embd=64, n_head=4, n_positions=256, vocab_size=500,
                                   bos_token_id=0, eos_token_id=0)).eval()
PREFIX = torch.randint(0, 500, (1, 60))
QUESTIONS = [torch.randint(0, 500, (1, n)) for n in (5, 12, 1, 30)]


def test_same_scores_with_and_without_the_prefix_cache():
    cache = build_prefix_cache(MODEL, PREFIX)
    for q in QUESTIONS:
        a = first_token_without_prefix_cache(MODEL, PREFIX, q)
        b = first_token_with_prefix_cache(MODEL, cache, q)
        assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
        assert a.argmax() == b.argmax()


def test_cache_goes_back_to_the_prefix_after_every_request():
    cache = build_prefix_cache(MODEL, PREFIX)
    before = last_layer_keys(cache).clone()
    for q in QUESTIONS:
        first_token_with_prefix_cache(MODEL, cache, q)
        assert cache.get_seq_length() == PREFIX.shape[1]
    assert torch.equal(last_layer_keys(cache), before)          # the shared part was never changed


def test_whole_answer_is_identical():
    for q in QUESTIONS:
        plain = greedy_answer(MODEL, DynamicCache(), torch.cat([PREFIX, q], dim=1), 15)
        cache = build_prefix_cache(MODEL, PREFIX)
        assert greedy_answer(MODEL, cache, q, 15) == plain


def test_prefix_keys_do_not_depend_on_what_follows():
    a = build_prefix_cache(MODEL, torch.cat([PREFIX, QUESTIONS[0]], dim=1))
    b = build_prefix_cache(MODEL, torch.cat([PREFIX, QUESTIONS[3]], dim=1))
    n = PREFIX.shape[1]
    assert torch.allclose(last_layer_keys(a)[:n], last_layer_keys(b)[:n], atol=1e-5)


def test_one_changed_token_changes_every_later_position():
    changed = PREFIX.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 500
    a = last_layer_keys(build_prefix_cache(MODEL, PREFIX))
    b = last_layer_keys(build_prefix_cache(MODEL, changed))
    differs = (a - b).abs().max(dim=1).values > 1e-5
    assert not differs[:10].any()                                # before the change: reusable
    assert differs[10:].all()                                    # from the change on: all different


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"\n{len(tests)} checks passed")
