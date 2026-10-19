"""Needle lookups whose prefix must come back from the HiCache host tier.

usage: hicache_reload_needle.py <port> [n_pairs] [n_flood] [n_q]
Caches doc 0, floods the device pool with n_flood other docs so doc 0 is
evicted to host, then asks doc 0 again and checks the answers.
"""

import json
import random
import sys
import urllib.request

port = sys.argv[1]
n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
n_flood = int(sys.argv[3]) if len(sys.argv) > 3 else 6
n_q = int(sys.argv[4]) if len(sys.argv) > 4 else 5


def make_doc(seed):
    rng = random.Random(seed)
    keys = rng.sample(range(10000, 99999), n_pairs)
    pairs = {f"K{k}": f"{rng.randint(100000, 999999)}" for k in keys}
    doc = "Memorize this table of codes.\n" + "\n".join(
        f"{k} -> {v}" for k, v in pairs.items()
    )
    return doc, pairs, rng


def ask(doc, key):
    body = {
        "model": "x",
        "messages": [
            {
                "role": "user",
                "content": f"{doc}\n\nWhat code does {key} map to? "
                "Reply with only the 6-digit code.",
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    usage = res["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    msg = res["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return text.strip(), usage["prompt_tokens"], cached


def run(tag, doc, pairs, keys):
    hits = 0
    for key in keys:
        ans, n_prompt, cached = ask(doc, key)
        ok = pairs[key] in ans
        hits += ok
        print(f"{tag} {key} ok={ok} prompt={n_prompt} cached={cached} got={ans[-30:]!r}")
    return hits


doc0, pairs0, rng0 = make_doc(0)
keys0 = rng0.sample(list(pairs0), 2 * n_q)
first = run("warm", doc0, pairs0, keys0[:n_q])
for s in range(1, n_flood + 1):
    doc, pairs, rng = make_doc(s)
    run(f"flood{s}", doc, pairs, rng.sample(list(pairs), 1))
again = run("reload", doc0, pairs0, keys0[n_q:])
print(f"port {port}: warm {first}/{n_q}, reload {again}/{n_q}")
