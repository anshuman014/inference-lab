"""Checks for async_concurrent_requests.py. Needs no model and no llama.cpp (only `pip install httpx`).

A small stand-in server plays the part of llama-server: it has a fixed number of slots, makes
extra requests queue, and streams one `data:` line per token.

Run:
    python test_async_concurrent_requests.py
"""
import asyncio
import inspect
import json
import time
from contextlib import asynccontextmanager

import httpx

import async_concurrent_requests as acr

TOKEN_SECONDS = 0.02            # the stand-in server "computes" one token in 20 ms


@asynccontextmanager
async def fake_llama_server(slots, split_last_token=False):
    """Yields (url, seen). `seen` collects every request body the server received."""
    free_slots = asyncio.Semaphore(slots)
    seen = []

    async def handle(reader, writer):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        method, path = head.split()[:2]
        length = next((int(h.split(":")[1]) for h in head.split("\r\n") if h.lower().startswith("content-length")), 0)
        body = json.loads(await reader.readexactly(length)) if length else None

        def send(text):
            writer.write(text.encode())

        if method == "GET":
            reply = json.dumps({"status": "ok", "total_slots": slots})
            send(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(reply)}\r\n"
                 f"Connection: close\r\n\r\n{reply}")
        else:
            seen.append(body)
            send("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            await writer.drain()                             # headers go out at once, like llama-server
            async with free_slots:                           # no free slot: this request waits in the queue
                n = body["max_tokens"]
                for i in range(n):
                    await asyncio.sleep(TOKEN_SECONDS)
                    if split_last_token and i == n - 1:      # e.g. half an emoji: a token with no text yet
                        continue
                    delta = {"choices": [{"delta": {"content": f"w{i} "}}]}
                    send(f"data: {json.dumps(delta)}\n\n")
                    await writer.drain()
            usage = {"choices": [], "usage": {"completion_tokens": n}}
            send(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", seen
    finally:
        server.close()
        await server.wait_closed()


async def test_one_request_reads_the_stream_and_times_it():
    async with fake_llama_server(slots=1) as (url, seen):
        async with httpx.AsyncClient(base_url=url, timeout=None) as client:
            r = await acr.one_request(client, 7, "What is a token?", 5, time.perf_counter())
    assert r["number"] == 7 and r["tokens"] == 5
    assert r["text"] == "w0 w1 w2 w3 w4 "
    assert 0 < r["ttft"] < r["total"]                                    # the first token arrives before the last
    assert r["ttft"] >= TOKEN_SECONDS * 0.9 and r["total"] >= 5 * TOKEN_SECONDS * 0.9
    assert seen[0]["stream"] is True and seen[0]["max_tokens"] == 5
    assert seen[0]["messages"] == [{"role": "user", "content": "What is a token?"}]


async def test_token_count_comes_from_the_server_not_from_counting_lines():
    async with fake_llama_server(slots=1, split_last_token=True) as (url, _):
        async with httpx.AsyncClient(base_url=url, timeout=None) as client:
            r = await acr.one_request(client, 1, "hi", 5, time.perf_counter())
    assert r["text"] == "w0 w1 w2 w3 "                                   # only 4 lines carried text ...
    assert r["tokens"] == 5                                              # ... but the server made 5 tokens


async def test_one_at_a_time_never_overlaps_and_all_at_once_does():
    questions = acr.make_questions(8)
    async with fake_llama_server(slots=4) as (url, _):
        async with httpx.AsyncClient(base_url=url, timeout=None) as client:
            serial, serial_wall = await acr.one_at_a_time(client, questions, 5)
            parallel, parallel_wall = await acr.all_at_once(client, questions, 5)
    assert acr.most_in_flight(serial) == 1
    assert acr.most_in_flight(parallel) == 8
    assert [r["number"] for r in parallel] == [1, 2, 3, 4, 5, 6, 7, 8]   # gather keeps the order it was given
    assert all(r["tokens"] == 5 for r in serial + parallel)
    # 8 requests x 5 tokens x 20 ms: about 0.8 s in a row, about 0.2 s with 4 slots.
    assert serial_wall >= 8 * 5 * TOKEN_SECONDS * 0.9
    assert parallel_wall < serial_wall / 2


async def test_requests_beyond_the_slots_wait_in_a_queue():
    async with fake_llama_server(slots=2) as (url, _):
        async with httpx.AsyncClient(base_url=url, timeout=None) as client:
            results, _ = await acr.all_at_once(client, acr.make_questions(4), 5)
    waits = sorted(r["ttft"] for r in results)
    assert waits[1] < 3 * TOKEN_SECONDS                                  # two got a slot straight away
    assert waits[2] > 5 * TOKEN_SECONDS                                  # two waited for a whole answer first


def test_most_in_flight():
    assert acr.most_in_flight([{"sent": 0, "finished": 1}, {"sent": 1, "finished": 2}]) == 1      # back to back
    assert acr.most_in_flight([{"sent": 0, "finished": 3}, {"sent": 1, "finished": 2},
                               {"sent": 2.5, "finished": 4}]) == 2


def test_summarize_computes_the_three_metrics():
    results = [{"sent": 0.0, "finished": 1.2, "ttft": 0.2, "total": 1.2, "tokens": 11},     # 1.0 s / 10 gaps
               {"sent": 0.0, "finished": 2.4, "ttft": 0.4, "total": 2.4, "tokens": 11},     # 2.0 s / 10 gaps
               {"sent": 0.0, "finished": 3.6, "ttft": 0.6, "total": 3.6, "tokens": 11}]     # 3.0 s / 10 gaps
    s = acr.summarize(results, wall_seconds=3.6)
    assert s["tokens"] == 33 and s["in_flight"] == 3
    assert round(s["throughput"], 2) == round(33 / 3.6, 2)
    assert round(s["ttft"], 3) == 0.4 and round(s["per_token"], 3) == 0.2


def test_make_questions_gives_the_number_asked_for():
    assert len(acr.make_questions(20)) == 20 and len(set(acr.make_questions(20))) == 20
    assert len(acr.make_questions(45)) == 45                             # topics repeat past 20


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        if inspect.iscoroutinefunction(test):
            asyncio.run(test())
        else:
            test()
        print("ok  ", test.__name__)
    print(f"\n{len(tests)} checks passed")
