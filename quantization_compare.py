"""Quantization: one model at 16, 8, 4 and 2 bits per weight, compared on memory, speed and answers.

Uses two llama.cpp programs:
  llama-bench   times prefill (prompt processing) and decode (token generation)
  llama-server  answers the same questions greedily (always the top token), so the
                answers can be compared word by word against the 16-bit model

Setup, once:
    brew install llama.cpp
    pip install -U huggingface_hub
    hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \\
        qwen2.5-1.5b-instruct-fp16.gguf qwen2.5-1.5b-instruct-q8_0.gguf \\
        qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-1.5b-instruct-q2_k.gguf \\
        --local-dir ~/models
Any of the four files may be left out; the script uses the ones it finds.

Run:
    source ~/inference/bin/activate
    python quantization_compare.py                # looks in ~/models
    python quantization_compare.py --models-dir /some/other/folder
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from contextlib import contextmanager

MODELS = [                                  # (name, file), most bits first: the first one found is the reference
    ("F16", "qwen2.5-1.5b-instruct-fp16.gguf"),
    ("Q8_0", "qwen2.5-1.5b-instruct-q8_0.gguf"),
    ("Q4_K_M", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("Q2_K", "qwen2.5-1.5b-instruct-q2_k.gguf"),
]

QUESTIONS = [                               # (question, a word the right answer must contain, or None)
    ("What is 17 times 23? Reply with just the number.", "391"),
    ("What is the capital of Australia? Reply with one word.", "Canberra"),
    ("A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. "
     "What time does it arrive? Reply with just the time.", "6:15"),
    ("Which planet is closest to the Sun? Reply with one word.", "Mercury"),
    ("In two sentences, explain what a KV cache is in a language model.", None),
    ("Write a short Python function that checks whether a number is prime.", None),
]

PROMPT_TOKENS, GEN_TOKENS, REPEATS = 512, 128, 3
ANSWER_TOKENS = 80
PORT = 8091


def find_program(name):
    """Path of a llama.cpp program: on PATH (brew), or in $LLAMA_CPP_BIN."""
    folder = os.environ.get("LLAMA_CPP_BIN")
    path = os.path.join(folder, name) if folder else shutil.which(name)
    if not path or not os.path.exists(path):
        raise SystemExit(f"Cannot find {name}. Install llama.cpp (brew install llama.cpp) "
                         f"or set LLAMA_CPP_BIN to the folder that holds it.")
    return path


def parse_bench(json_text):
    """llama-bench -o json prints one record per test. Returns the numbers this lesson needs."""
    out = {}
    for test in json.loads(json_text):
        out["size_bytes"], out["params"] = test["model_size"], test["model_n_params"]
        if test["n_prompt"] > 0 and test["n_gen"] == 0:
            out["prefill_tps"] = test["avg_ts"]              # prompt tokens per second
        elif test["n_gen"] > 0 and test["n_prompt"] == 0:
            out["decode_tps"] = test["avg_ts"]               # new tokens per second
    return out


def bench(model_path):
    """Time prefill of a 512-token prompt and decode of 128 tokens, 3 runs each, averaged."""
    cmd = [find_program("llama-bench"), "-m", model_path, "-p", str(PROMPT_TOKENS),
           "-n", str(GEN_TOKENS), "-r", str(REPEATS), "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_bench(result.stdout)


def bits_per_weight(size_bytes, params):
    return size_bytes * 8 / params


@contextmanager
def serve(model_path, port=PORT):
    """Start llama-server on one model, wait until it is ready, stop it afterwards."""
    cmd = [find_program("llama-server"), "-m", model_path, "--port", str(port), "-c", "4096", "-np", "1"]
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 180
        while True:                                      # /health answers 200 once the model is loaded
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if server.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"llama-server did not start for {model_path}")
            time.sleep(0.5)
        yield port
    finally:
        server.terminate()
        server.wait(timeout=30)


def ask(port, question, max_tokens=ANSWER_TOKENS):
    """Greedy answer: temperature 0 and top_k 1 always take the most likely token."""
    body = {"messages": [{"role": "user", "content": question}],
            "temperature": 0, "top_k": 1, "seed": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def same_words(reference, answer):
    """How many words the answer shares with the reference before the first difference."""
    a, b = reference.split(), answer.split()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n, len(a)


def is_correct(answer, expected):
    return None if expected is None else expected.lower() in answer.lower()


def short(text, width=72):
    line = " ".join(text.split())
    return line if len(line) <= width else line[:width - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=os.path.expanduser("~/models"))
    args = parser.parse_args()
    found = [(name, os.path.join(args.models_dir, f)) for name, f in MODELS
             if os.path.exists(os.path.join(args.models_dir, f))]
    if not found:
        raise SystemExit(f"No model files in {args.models_dir}. See the setup steps at the top of this file.")

    # Part 1: memory and speed.
    print(f"Part 1: memory and speed (llama-bench: {PROMPT_TOKENS}-token prompt, {GEN_TOKENS} new tokens, "
          f"{REPEATS} runs)\n")
    print(f"{'format':<8} {'weights GB':>10} {'bits per':>9} {'prefill':>9} {'decode':>8} {'GB/s read':>10}")
    print(f"{'':<8} {'':>10} {'weight':>9} {'tok/s':>9} {'tok/s':>8} {'decoding':>10}")
    stats = {}
    for name, path in found:
        s = stats[name] = bench(path)
        gb = s["size_bytes"] / 1e9
        print(f"{name:<8} {gb:>10.2f} {bits_per_weight(s['size_bytes'], s['params']):>9.1f} "
              f"{s['prefill_tps']:>9.0f} {s['decode_tps']:>8.1f} {s['decode_tps'] * gb:>10.0f}")
    print("\nEvery decoded token reads all the weights once, so decode tok/s x weights GB is roughly "
          "the memory bandwidth in use.")

    # Part 2: answers.
    answers = {}
    for name, path in found:
        with serve(path) as port:
            answers[name] = [ask(port, q) for q, _ in QUESTIONS]
    ref = found[0][0]
    print(f"\nPart 2: greedy answers, compared word by word with {ref}\n")
    for k, (q, expected) in enumerate(QUESTIONS):
        print(f"Q{k + 1}. {q}")
        for name, _ in found:
            a = answers[name][k]
            n, total = same_words(answers[ref][k], a)
            ok = is_correct(a, expected)
            mark = "" if ok is None else ("  right" if ok else "  WRONG")
            match = "reference" if name == ref else f"same as {ref} for {n} of {total} words"
            print(f"  {name:<7} {short(a):<75} [{match}]{mark}")
        print()

    # Part 3: summary.
    checkable = [k for k, (_, e) in enumerate(QUESTIONS) if e is not None]
    print("Part 3: summary\n")
    print(f"{'format':<8} {'right':>7} {'answers identical':>18} {'words matching':>18}")
    print(f"{'':<8} {'':>7} {f'to {ref}':>18} {'before 1st change':>18}")
    for name, _ in found:
        right = sum(bool(is_correct(answers[name][k], QUESTIONS[k][1])) for k in checkable)
        pairs = [same_words(answers[ref][k], answers[name][k]) for k in range(len(QUESTIONS))]
        identical = sum(answers[name][k] == answers[ref][k] for k in range(len(QUESTIONS)))
        share = sum(n for n, _ in pairs) / max(1, sum(t for _, t in pairs))
        print(f"{name:<8} {f'{right} of {len(checkable)}':>7} {f'{identical} of {len(QUESTIONS)}':>18} {share:>18.0%}")


if __name__ == "__main__":
    main()
