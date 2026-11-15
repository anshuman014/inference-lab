"""Checks for quantization_compare.py. Needs no model and no llama.cpp.

Run:
    python test_quantization_compare.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import quantization_compare as qc

# Two records in the shape llama-bench -o json prints (trimmed to the fields that matter here).
BENCH_JSON = json.dumps([
    {"model_type": "qwen2 ?B Q8_0", "model_size": 159857952, "model_n_params": 150367488,
     "n_prompt": 64, "n_gen": 0, "avg_ns": 37677741, "avg_ts": 1698.6},
    {"model_type": "qwen2 ?B Q8_0", "model_size": 159857952, "model_n_params": 150367488,
     "n_prompt": 0, "n_gen": 16, "avg_ns": 92915000, "avg_ts": 172.2},
])


def test_parse_bench_reads_prefill_and_decode():
    s = qc.parse_bench(BENCH_JSON)
    assert s == {"size_bytes": 159857952, "params": 150367488, "prefill_tps": 1698.6, "decode_tps": 172.2}


def test_bits_per_weight():
    assert qc.bits_per_weight(2 * 1000, 1000) == 16                    # F16: 2 bytes a weight
    assert round(qc.bits_per_weight(159857952, 150367488), 1) == 8.5   # Q8_0: 8 bits + a scale per 32


def test_same_words_counts_up_to_the_first_difference():
    assert qc.same_words("the capital is Canberra", "the capital is Canberra") == (4, 4)
    assert qc.same_words("the capital is Canberra", "the capital was Canberra") == (2, 4)
    assert qc.same_words("391", "The answer is 391") == (0, 1)
    assert qc.same_words("a b c", "a b") == (2, 3)


def test_is_correct():
    assert qc.is_correct("The answer is 391.", "391")
    assert not qc.is_correct("It is 401.", "391")
    assert qc.is_correct("canberra", "Canberra")                       # case does not matter
    assert qc.is_correct("anything", None) is None                     # open questions are not scored


def test_every_question_has_a_scorable_answer_or_none():
    for q, expected in qc.QUESTIONS:
        assert q.strip() and (expected is None or expected.strip())


def test_ask_sends_a_greedy_request_and_reads_the_reply():
    seen = {}

    class FakeServer(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps({"choices": [{"message": {"content": "  391\n"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=server.handle_request, daemon=True).start()
    answer = qc.ask(server.server_address[1], "What is 17 times 23?")
    server.server_close()
    assert answer == "391"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["top_k"] == 1
    assert seen["body"]["messages"] == [{"role": "user", "content": "What is 17 times 23?"}]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"\n{len(tests)} checks passed")
