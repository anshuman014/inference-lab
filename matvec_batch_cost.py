"""Why batching is cheap: one big matrix times 1 vector vs. a batch of vectors.

The matrix W stands in for a model's weights. It is 0.27 GB, far too big for
the cache, so every multiply has to stream all of W in from memory.

  W @ one vector   -> W is read once, and each weight is used once
  W @ 64 vectors   -> W is still read once, but each weight is used 64 times

If reading W is the slow part, the time per vector should fall as the batch
grows. This script measures exactly that.

Run:
    source ~/inference/bin/activate
    python matvec_batch_cost.py
"""
import time

import torch

torch.manual_seed(0)

ROWS, COLS = 8192, 8192                 # 67 million weights x 4 bytes = 0.27 GB
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
REPEATS = 5                             # time each multiply 5 times, keep the fastest

W = torch.randn(ROWS, COLS)             # the "weights"
weight_gb = W.numel() * 4 / 1e9


def fastest_seconds(X):
    """Time W @ X several times and return the fastest run."""
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        Y = W @ X                        # shape (ROWS, batch): one result column per input vector
        best = min(best, time.perf_counter() - start)
    return best


W @ torch.randn(COLS, 1)                # warm-up multiply, not timed

print(f"weights: {ROWS} x {COLS} = {W.numel() / 1e6:.0f} million numbers = {weight_gb:.2f} GB")
print()
print(f"{'batch':>5} {'total ms':>9} {'ms per vector':>14} {'cost vs. batch 1':>17}")
per_vector_at_1 = None
for batch in BATCH_SIZES:
    X = torch.randn(COLS, batch)         # each column is one input vector
    seconds = fastest_seconds(X)
    per_vector = seconds / batch
    if per_vector_at_1 is None:
        per_vector_at_1 = per_vector
        seconds_at_1 = seconds
    print(f"{batch:>5} {seconds * 1000:>9.2f} {per_vector * 1000:>14.3f} {per_vector / per_vector_at_1:>16.0%}")

print()
print(f"at batch 1 the weights streamed through at about {weight_gb / seconds_at_1:.0f} GB per second")
