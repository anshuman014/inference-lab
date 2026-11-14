"""Static vs. continuous batching, simulated.

A GPU has a fixed number of seats: how many requests it can carry in one pass.
Time moves in steps. In one step, every seated request gets one new token.
A request is (name, arrival step, tokens it needs).

  static batching:      fill the seats, lock them, run until the longest
                        request in the batch is finished
  continuous batching:  after every step, finished requests leave and
                        waiting requests take the free seats

For each request we record its wait: the steps between arriving and starting.

Run:
    source ~/inference/bin/activate
    python batching_simulator.py
"""
import random


def static_batching(requests, seats):
    """Returns {name: (start, finish)} and the step at which everything is done."""
    todo = sorted(requests, key=lambda r: r[1])          # by arrival
    result, t = {}, 0
    while todo:
        waiting = [r for r in todo if r[1] <= t]
        if not waiting:                                   # nobody here yet: skip ahead
            t = todo[0][1]
            continue
        batch = waiting[:seats]                           # first come, first served
        for name, arrival, tokens in batch:
            result[name] = (t, t + tokens)
        t += max(tokens for _, _, tokens in batch)        # seats stay locked until the longest ends
        todo = [r for r in todo if r not in batch]
    return result, t


def continuous_batching(requests, seats):
    """Same output as static_batching, but the batch is rebuilt after every step."""
    todo = sorted(requests, key=lambda r: r[1])
    seated = {}                                           # name -> tokens still needed
    result, start, t = {}, {}, 0
    while todo or seated:
        while todo and todo[0][1] <= t and len(seated) < seats:
            name, arrival, tokens = todo.pop(0)           # a waiting request takes a free seat
            seated[name], start[name] = tokens, t
        t += 1                                            # one step: one token for everyone seated
        for name in list(seated):
            seated[name] -= 1
            if seated[name] == 0:                         # finished: the seat is free right away
                result[name] = (start[name], t)
                del seated[name]
    return result, t


def report(label, requests, result, done, seats):
    waits = [result[name][0] - arrival for name, arrival, _ in requests]
    work = sum(tokens for _, _, tokens in requests)
    print(f"{label:<12} average wait {sum(waits) / len(waits):7.1f}   longest wait {max(waits):5d}   "
          f"all done at step {done:5d}   seats in use {work / (done * seats):4.0%}")


# The example from the video: 3 seats, 5 requests.
small = [("A", 0, 2), ("B", 0, 6), ("C", 0, 3), ("D", 1, 2), ("E", 2, 3)]
print("Video example: 3 seats, 5 requests")
for label, simulate in [("static", static_batching), ("continuous", continuous_batching)]:
    result, done = simulate(small, seats=3)
    report(label, small, result, done, 3)
    print("   ", "  ".join(f"{n}: start {s}, finish {f}" for n, (s, f) in sorted(result.items())))

# A busier server: 200 requests of very different lengths, 8 seats.
random.seed(0)
big, t = [], 0
for i in range(200):
    t += random.randint(0, 32)                            # steps until the next request arrives
    big.append((f"r{i}", t, random.randint(5, 200)))      # needs 5 to 200 tokens
print("\nBusy server: 8 seats, 200 requests")
for label, simulate in [("static", static_batching), ("continuous", continuous_batching)]:
    result, done = simulate(big, seats=8)
    report(label, big, result, done, 8)
