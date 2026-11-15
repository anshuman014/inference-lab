"""Checks for paged_kv_cache_simulator.py. No test framework needed.

Run:
    python test_paged_kv_cache_simulator.py
"""
import random

from paged_kv_cache_simulator import Paged, ReserveMax, serve

SMALL = [("A", 0, 3, 2), ("B", 0, 2, 4), ("C", 0, 5, 4)]


class CheckedPaged(Paged):
    """Paged, but it re-checks the rules after every change."""

    def __init__(self, memory, block_size):
        super().__init__(memory, block_size)
        self.total_blocks = len(self.free_blocks)

    def check(self):
        held = [b for blocks in self.table.values() for b in blocks]
        assert len(held) == len(set(held)), "a block is held by two requests"
        assert not set(held) & set(self.free_blocks), "a held block is also on the free list"
        assert len(held) + len(self.free_blocks) == self.total_blocks, "a block was lost or invented"
        for name, blocks in self.table.items():
            empty = len(blocks) * self.block_size - self.used[name]
            assert 0 <= empty < self.block_size, "only the last block may be partly empty"

    def admit(self, name, prompt):
        ok = super().admit(name, prompt)
        self.check()
        return ok

    def grow(self, name):
        ok = super().grow(name)
        self.check()
        return ok

    def release(self, name):
        super().release(name)
        self.check()


def random_requests(n, seed, gap=3):
    random.seed(seed)
    requests, t = [], 0
    for i in range(n):
        t += random.randint(0, gap)
        requests.append((f"r{i}", t, random.randint(1, 300), random.randint(1, 100)))
    return requests


def test_video_example_reserve_max():
    result, done, log, pushed_out = serve(SMALL, ReserveMax(24, max_tokens=12))
    assert result == {"A": (0, 2), "B": (0, 4), "C": (2, 6)}        # C waits for A's reservation
    assert done == 6 and pushed_out == 0
    assert log[0] == (2, 24, 7)                                      # step 1: 24 slots reserved, 7 hold tokens


def test_video_example_paged():
    memory = Paged(24, block_size=4)
    result, done, log, pushed_out = serve(SMALL, memory)
    assert result == {"A": (0, 2), "B": (0, 4), "C": (0, 4)}        # all three run from step 0
    assert done == 4 and pushed_out == 0
    assert log[0] == (3, 16, 13)                                     # step 1: 4 blocks held, 13 tokens
    assert log[1] == (3, 20, 16)                                     # step 2: A took a fifth block
    assert sorted(memory.free_blocks) == [0, 1, 2, 3, 4, 5]          # everything was given back


def test_block_table_and_lookup():
    memory = Paged(24, block_size=4)
    for name, prompt in [("A", 3), ("B", 2), ("C", 5)]:
        assert memory.admit(name, prompt)
    assert memory.table == {"A": [0], "B": [1], "C": [2, 3]}
    memory.grow("A")                                                 # 4th token fills block 0
    memory.grow("A")                                                 # 5th token needs a new block
    assert memory.table["A"] == [0, 4]                               # not neighbours, and that is fine
    assert memory.where("A", 3) == (0, 3)                            # 4th token: block 0, last slot
    assert memory.where("A", 4) == (4, 0)                            # 5th token: block 4, first slot
    places = [memory.where("C", i) for i in range(5)]
    assert len(set(places)) == 5                                     # every token has its own slot


def test_rules_hold_on_random_traffic():
    for seed in range(5):
        for block_size in (1, 4, 16, 64):
            requests = random_requests(120, seed)
            memory = CheckedPaged(4096, block_size)
            result, done, log, _ = serve(requests, memory)
            assert set(result) == {r[0] for r in requests}           # everyone finished
            for name, arrival, prompt, output in requests:
                start, finish = result[name]
                assert start >= arrival and finish - start >= output
            for running, reserved, used in log:
                assert used <= reserved <= 4096
                assert reserved - used < running * block_size + 1    # under one block wasted per request
            assert len(memory.free_blocks) == memory.total_blocks and not memory.table


def test_one_huge_block_is_the_same_as_reserving_the_max():
    requests = random_requests(120, seed=7)
    requests = [(n, a, p, o) for n, a, p, o in requests if p + o <= 512]
    a = serve(requests, ReserveMax(4096, max_tokens=512))
    b = serve(requests, Paged(4096, block_size=512))
    assert a == b


def test_pushed_out_requests_still_finish():
    # 8 blocks of 4. Three prompts fill 6 blocks, then all three need to grow.
    requests = [("A", 0, 8, 9), ("B", 0, 8, 9), ("C", 0, 8, 9)]
    memory = CheckedPaged(32, block_size=4)
    result, done, log, pushed_out = serve(requests, memory)
    assert pushed_out >= 1
    assert set(result) == {"A", "B", "C"}
    assert result["A"] == (0, 9)                                     # the oldest is never pushed out
    assert result["C"][1] > 9                                        # the newest had to restart
    assert max(reserved for _, reserved, _ in log) <= 32


def test_paging_beats_reserving_on_a_busy_server():
    requests = random_requests(200, seed=1, gap=1)
    _, done_reserve, log_reserve, _ = serve(requests, ReserveMax(4096, max_tokens=400))
    _, done_paged, log_paged, _ = serve(requests, Paged(4096, block_size=16))
    assert done_paged < done_reserve
    assert max(row[0] for row in log_paged) > max(row[0] for row in log_reserve)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"\n{len(tests)} checks passed")
