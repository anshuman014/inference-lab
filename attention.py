"""Single-head attention by hand, on tiny 2-number vectors.

Part 1: one token ("sat") looks back at the tokens so far.
Part 2: every token at once, with the causal mask.
Part 3: add a 4th token and check that the earlier rows do not change.

Run:
    source ~/inference/bin/activate
    python attention.py
"""
import torch

tokens = ["The", "cat", "sat"]
Q = torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])    # one query per token
K = torch.tensor([[-1.0, 2.0], [1.0, 1.0], [1.0, -1.0]])  # one key per token
V = torch.tensor([[1.0, 0.0], [2.0, 4.0], [0.0, 1.0]])    # one value per token

# ---- Part 1: "sat" scores its query against every key ----
q_sat = Q[2]
scores = K @ q_sat                       # dot product with each key
weights = torch.softmax(scores, dim=-1)  # scores -> weights that sum to 1
output = weights @ V                     # blend the values using the weights

print("Part 1: sat looks back")
print("  scores :", scores.tolist())
print("  weights:", [round(w, 3) for w in weights.tolist()])
print("  output :", [round(x, 2) for x in output.tolist()])


# ---- Part 2: all tokens at once, each one only looking backward ----
def attention(Q, K, V):
    n = Q.shape[0]
    scores = Q @ K.T                                         # row i = token i's scores
    future = torch.triu(torch.ones(n, n), diagonal=1).bool()  # True above the diagonal
    scores = scores.masked_fill(future, float("-inf"))        # hide tokens that come later
    weights = torch.softmax(scores, dim=-1)
    return weights, weights @ V


w3, out3 = attention(Q, K, V)
print("\nPart 2: attention weights (row = token asking, column = token looked at)")
for name, row in zip(tokens, w3):
    print(f"  {name:>4}", [round(x, 3) for x in row.tolist()])

# ---- Part 3: a 4th token arrives ----
Q4 = torch.cat([Q, torch.tensor([[1.0, 2.0]])])
K4 = torch.cat([K, torch.tensor([[0.0, 2.0]])])
V4 = torch.cat([V, torch.tensor([[3.0, 3.0]])])
w4, out4 = attention(Q4, K4, V4)

print("\nPart 3: after adding 'on'")
for name, row in zip(tokens + ["on"], w4):
    print(f"  {name:>4}", [round(x, 3) for x in row.tolist()])
print("  outputs for The, cat, sat unchanged:", torch.allclose(out4[:3], out3))
