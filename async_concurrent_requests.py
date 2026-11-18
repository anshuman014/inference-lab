"""asyncio: send 20 requests to llama-server one at a time, then all at once, and compare.

A request to a model server is almost all waiting: Python sends a few bytes, then sits idle while
the server computes. asyncio lets one Python thread start all 20 requests and wait for them
together. Each `await` means "I am waiting here, so let another request use the thread."

Setup, once:
    brew install llama.cpp
    pip install httpx huggingface_hub
    hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/models

Start the server in one Terminal tab and leave it running (-np 4 gives it 4 slots, so it can
work on 4 requests at the same moment; -c 8192 is the context shared out among the slots):
    llama-server -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -np 4 -c 8192 --port 8080

Run in a second tab:
    source ~/inference/bin/activate
    python async_concurrent_requests.py
    python async_concurrent_requests.py --requests 20 --max-tokens 64 --url http://127.0.0.1:8080
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

TOPICS = ["a KV cache", "a token", "attention", "a GPU", "memory bandwidth", "batching", "quantization",
          "a thread", "a process", "a cache line", "prefill", "decoding", "a logit", "softmax", "an embedding",
          "a system prompt", "a hash", "a queue", "latency", "throughput"]


def make_questions(n):
    return [f"In three sentences, explain what {TOPICS[i % len(TOPICS)]} is." for i in range(n)]


async def one_request(client, number, question, max_tokens, round_start):
    """Send one question, read the answer as it streams back, and time it.

    The server streams one line per token: `data: {...json...}`. The last line is `data: [DONE]`.
    """
    body = {"messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    first_token = None
    pieces, tokens = [], None
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:   # waits for the headers
        response.raise_for_status()
        async for line in response.aiter_lines():                                      # waits for each line
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if chunk.get("usage"):                       # the last chunk carries the server's own token count
                tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                piece = choice["delta"].get("content")
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    pieces.append(piece)
    finished = time.perf_counter()
    if first_token is None:                              # an empty answer: count the whole wait as TTFT
        first_token = finished
    return {"number": number,
            "sent": sent - round_start,                  # seconds since the round began
            "finished": finished - round_start,
            "ttft": first_token - sent,                  # time to first token, as this client saw it
            "total": finished - sent,
            "tokens": tokens if tokens is not None else len(pieces),
            "text": "".join(pieces)}


async def one_at_a_time(client, questions, max_tokens):
    """Round 1: send a request, wait until it has completely finished, then send the next."""
    start = time.perf_counter()
    results = []
    for number, question in enumerate(questions, 1):
        results.append(await one_request(client, number, question, max_tokens, start))
    return results, time.perf_counter() - start


async def all_at_once(client, questions, max_tokens):
    """Round 2: start every request, then wait for all of them together."""
    start = time.perf_counter()
    jobs = [one_request(client, number, question, max_tokens, start)       # nothing runs yet: these are
            for number, question in enumerate(questions, 1)]               # 20 paused jobs (coroutines)
    results = await asyncio.gather(*jobs)                                  # run them all; wait for the last
    return list(results), time.perf_counter() - start


def most_in_flight(results):
    """The largest number of requests that were open at the same moment."""
    events = [(r["sent"], 1) for r in results] + [(r["finished"], -1) for r in results]
    open_now = most = 0
    for _, change in sorted(events):                     # at equal times, -1 sorts first: close, then open
        open_now += change
        most = max(most, open_now)
    return most


def summarize(results, wall_seconds):
    """The three metrics from the prefill/decode lesson, for one round."""
    tokens = sum(r["tokens"] for r in results)
    per_token = [(r["total"] - r["ttft"]) / (r["tokens"] - 1) for r in results if r["tokens"] > 1]
    return {"wall": wall_seconds,
            "in_flight": most_in_flight(results),
            "tokens": tokens,
            "throughput": tokens / wall_seconds,                             # tokens per second, all users
            "ttft": statistics.median(r["ttft"] for r in results),           # seconds
            "per_token": statistics.median(per_token) if per_token else 0.0}  # seconds between tokens


async def server_slots(client):
    """How many requests the server works on at once (llama-server reports it at /props)."""
    try:
        return (await client.get("/props")).json().get("total_slots")
    except (httpx.HTTPError, ValueError):
        return None


def print_report(slots, serial, serial_wall, parallel, parallel_wall):
    n = len(parallel)
    print(f"\nRound 2, request by request (seconds). The server has {slots or 'an unknown number of'} slots.\n")
    print(f"{'request':>7} {'waited for':>12} {'finished':>10} {'tokens':>7}")
    print(f"{'':>7} {'first token':>12} {'at':>10} {'':>7}")
    for r in sorted(parallel, key=lambda r: r["ttft"]):
        print(f"{r['number']:>7} {r['ttft']:>12.2f} {r['finished']:>10.2f} {r['tokens']:>7}")

    a, b = summarize(serial, serial_wall), summarize(parallel, parallel_wall)
    print(f"\n{'':<28} {'one at a time':>14} {'all at once':>13}")
    rows = [("time for all " + str(n) + " requests", f"{a['wall']:.2f} s", f"{b['wall']:.2f} s"),
            ("most requests open at once", a["in_flight"], b["in_flight"]),
            ("tokens generated", a["tokens"], b["tokens"]),
            ("throughput, tokens/s", f"{a['throughput']:.0f}", f"{b['throughput']:.0f}"),
            ("median time to first token", f"{a['ttft']:.2f} s", f"{b['ttft']:.2f} s"),
            ("median time per token", f"{a['per_token'] * 1000:.1f} ms", f"{b['per_token'] * 1000:.1f} ms")]
    for name, x, y in rows:
        print(f"{name:<28} {x:>14} {y:>13}")
    print(f"\nAll at once was {a['wall'] / b['wall']:.1f}x as fast as one at a time. Python did the same work in both "
          f"rounds. Any gain comes\nfrom the server: with several requests in its slots, one pass over the weights "
          f"makes several tokens.")
    print(f"\nFirst answer: {' '.join(parallel[0]['text'].split())[:150]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    questions = make_questions(args.requests)

    # timeout=None: a queued request may wait many seconds for its first byte, and httpx gives up after 5 by default.
    async with httpx.AsyncClient(base_url=args.url, timeout=None) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except httpx.HTTPError:
            raise SystemExit(f"No llama-server at {args.url}. Start it first: see the top of this file.")
        slots = await server_slots(client)
        await one_request(client, 0, "Say hello.", 8, time.perf_counter())      # warm-up, not timed

        print(f"Round 1: {args.requests} requests, one at a time ...")
        serial, serial_wall = await one_at_a_time(client, questions, args.max_tokens)
        print(f"Round 2: {args.requests} requests, all at once ...")
        parallel, parallel_wall = await all_at_once(client, questions, args.max_tokens)
    print_report(slots, serial, serial_wall, parallel, parallel_wall)


if __name__ == "__main__":
    asyncio.run(main())
