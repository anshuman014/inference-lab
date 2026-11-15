"""Paged KV cache vs. reserving the maximum, simulated.

batching_simulator.py gave the GPU a fixed number of seats. Where does that
number come from? Memory: a request can only sit down if there is room for its
KV cache. So here the seats are gone and KV cache memory is the only limit.

Memory is counted in token slots. One slot holds the K and V of one token.
A request is (name, arrival step, prompt tokens, output tokens). Sitting down
puts the whole prompt in the cache at once (prefill, treated as instant). Then
every step adds one token to every seated request, as before.

Two ways to hand out the memory:

  ReserveMax   every request gets room for the longest possible conversation,
               up front, in one piece. Simple, but most of it stays empty.
  Paged        memory is cut into small equal blocks. A request gets one more
               block only when its last block is full. A block table per
               request lists which blocks it holds, in order. (PagedAttention)

The serve() loop is continuous_batching() from batching_simulator.py with two
changes: "is a seat free?" became "does the memory let it in?", and every new
token has to ask the memory for a slot.

Run:
    source ~/inference/bin/activate
    python paged_kv_cache_simulator.py
"""
import random

KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2      # Llama 3.1 8B, 16-bit: 128 KB per token
GB = 1024 ** 3


class ReserveMax:
    """The old way: reserve room for max_tokens the moment a request sits down."""

    def __init__(self, memory, max_tokens):
        self.memory, self.max_tokens = memory, max_tokens
        self.used = {}                                    # name -> tokens stored so far

    def admit(self, name, prompt):
        if self.reserved() + self.max_tokens > self.memory:
            return False                                  # no room for one more full reservation
        self.used[name] = prompt
        return True

    def grow(self, name):
        self.used[name] += 1                              # the room was reserved up front
        return True

    def release(self, name):
        del self.used[name]

    def reserved(self):
        return len(self.used) * self.max_tokens           # slots nobody else may touch


class Paged:
    """PagedAttention's way: fixed-size blocks, handed out one at a time."""

    def __init__(self, memory, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(memory // block_size))   # numbers of the blocks nobody holds
        self.cushion = len(self.free_blocks) // 100       # keep 1% free so running requests can grow
        self.table = {}                                   # name -> its block numbers, in order
        self.used = {}                                    # name -> tokens stored so far

    def admit(self, name, prompt):
        need = -(-prompt // self.block_size)              # blocks for the prompt, rounded up
        if need + self.cushion > len(self.free_blocks):
            return False
        self.table[name] = [self.free_blocks.pop(0) for _ in range(need)]
        self.used[name] = prompt
        return True

    def grow(self, name):
        if self.used[name] == len(self.table[name]) * self.block_size:   # last block is full
            if not self.free_blocks:
                return False                              # out of blocks: the caller must free some
            self.table[name].append(self.free_blocks.pop(0))             # any free block will do
        self.used[name] += 1
        return True

    def release(self, name):
        self.free_blocks += self.table.pop(name)          # every block is reusable right away
        del self.used[name]

    def reserved(self):
        return sum(len(blocks) for blocks in self.table.values()) * self.block_size

    def where(self, name, position):
        """Where token number `position` of a request lives: (block number, slot in that block)."""
        nth_block, slot = divmod(position, self.block_size)
        return self.table[name][nth_block], slot


def serve(requests, memory):
    """Continuous batching where memory decides who runs.

    Returns {name: (start, finish)}, the step at which everything is done,
    a per-step log of (requests running, slots reserved, slots holding tokens),
    and how many times a request was pushed out to free blocks.
    """
    todo = sorted(requests, key=lambda r: r[1])           # by arrival
    seated = {}                                           # name -> (the request, tokens still needed)
    result, start, log, pushed_out, t = {}, {}, [], 0, 0
    while todo or seated:
        while todo and todo[0][1] <= t and memory.admit(todo[0][0], todo[0][2]):
            request = todo.pop(0)                         # its prompt is now in the cache
            seated[request[0]] = (request, request[3])
            start.setdefault(request[0], t)
        t += 1                                            # one step: one token for everyone seated
        for name in list(seated):
            if name not in seated:                        # pushed out earlier in this step
                continue
            while not memory.grow(name):                  # out of blocks: push out the newest request
                newest = list(seated)[-1]
                memory.release(newest)
                todo.insert(0, seated.pop(newest)[0])     # it restarts from its prompt later
                pushed_out += 1
                if newest == name:
                    break
            else:
                request, left = seated[name]
                seated[name] = (request, left - 1)
        log.append((len(seated), memory.reserved(), sum(memory.used.values())))
        for name in list(seated):
            if seated[name][1] == 0:                      # finished: its memory is free right away
                result[name] = (start[name], t)
                memory.release(name)
                del seated[name]
    return result, t, log, pushed_out


def report(label, requests, result, done, log, pushed_out, total):
    waits = [result[name][0] - arrival for name, arrival, _, _ in requests]
    busy = [row for row in log if row[0]]                 # steps where something was running
    running = sum(row[0] for row in busy) / len(busy)
    reserved = sum(row[1] for row in busy) / len(busy)
    used = sum(row[2] for row in busy) / len(busy)
    print(f"{label:<15} average wait {sum(waits) / len(waits):7.1f}   running at once {running:5.1f}   "
          f"all done at step {done:5d}   reserved {reserved / total:4.0%} of memory   "
          f"empty inside reservations {1 - used / reserved:4.0%}   pushed out {pushed_out}")
    return reserved - used                                # average slots reserved but empty


if __name__ == "__main__":
    # The example from the video: 24 slots, at most 12 tokens per request, blocks of 4.
    small = [("A", 0, 3, 2), ("B", 0, 2, 4), ("C", 0, 5, 4)]
    print("Video example: 24 slots of KV cache memory, 3 requests")
    for label, memory in [("reserve max", ReserveMax(24, max_tokens=12)), ("paged", Paged(24, block_size=4))]:
        result, done, log, pushed_out = serve(small, memory)
        report(label, small, result, done, log, pushed_out, 24)
        print("   ", "  ".join(f"{n}: start {s}, finish {f}" for n, (s, f) in sorted(result.items())))

    # A busy server. 8 GB of KV cache memory for Llama 3.1 8B = 65,536 token slots.
    # 300 requests with prompts of 50 to 1,500 tokens and answers of 10 to 300 tokens.
    # The longest conversation the server allows is 2,048 tokens.
    TOTAL, MAX_TOKENS = 8 * GB // KV_BYTES_PER_TOKEN, 2048
    random.seed(0)
    big, t = [], 0
    for i in range(300):
        t += random.randint(0, 3)                         # steps until the next request arrives
        big.append((f"r{i}", t, random.randint(50, 1500), random.randint(10, 300)))
    print(f"\nBusy server: {TOTAL:,} slots (8 GB), 300 requests, at most {MAX_TOKENS:,} tokens each")
    for label, memory in [("reserve max", ReserveMax(TOTAL, MAX_TOKENS)), ("paged, 16", Paged(TOTAL, block_size=16))]:
        result, done, log, pushed_out = serve(big, memory)
        empty = report(label, big, result, done, log, pushed_out, TOTAL)
        print(f"    on average {empty * KV_BYTES_PER_TOKEN / GB:.2f} GB reserved but empty")

    # How big should a block be? Small blocks waste less, but mean longer block tables.
    print("\nSame busy server, different block sizes")
    for block_size in [1, 16, 64, 256, 1024, 2048]:
        result, done, log, pushed_out = serve(big, Paged(TOTAL, block_size))
        report(f"blocks of {block_size}", big, result, done, log, pushed_out, TOTAL)
