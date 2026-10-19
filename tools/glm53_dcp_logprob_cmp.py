"""Compare prompt and greedy-decode logprobs between two SGLang servers.

usage: glm53_dcp_logprob_cmp.py <port_a> <port_b> [repeat] [max_new_tokens]
"""

import json
import sys
import urllib.request

port_a, port_b = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 40
max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 16

text = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May? "
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn? Betty is saving money for a new wallet "
    "which costs $100. Betty has only half of the money she needs. Her parents "
    "decided to give her $15 for that purpose, and her grandparents twice as much "
    "as her parents. How much more money does Betty need to buy the wallet? "
) * repeat


def generate(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


payload = {
    "text": text,
    "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
    "return_logprob": True,
    "logprob_start_len": 0,
}
res = {p: generate(p, payload) for p in (port_a, port_b)}
inp = {p: res[p]["meta_info"]["input_token_logprobs"] for p in res}
out = {p: res[p]["meta_info"]["output_token_logprobs"] for p in res}
print(f"prompt tokens: {len(inp[port_a])}")
diffs = []
for i, (a, b) in enumerate(zip(inp[port_a], inp[port_b])):
    if a[0] is None or b[0] is None:
        continue
    diffs.append((i, abs(a[0] - b[0]), a[0], b[0]))
buckets = [0, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 1 << 30]
for lo, hi in zip(buckets, buckets[1:]):
    sel = [d for d in diffs if lo <= d[0] < hi]
    if sel:
        print(
            f"prompt pos [{lo},{hi}): max|dlp|={max(d[1] for d in sel):.3f} "
            f"mean|dlp|={sum(d[1] for d in sel) / len(sel):.4f} "
            f"mean lp a={sum(d[2] for d in sel) / len(sel):.3f} "
            f"b={sum(d[3] for d in sel) / len(sel):.3f}"
        )
for p in res:
    print(p, "out:", repr(res[p]["text"][:200]))
