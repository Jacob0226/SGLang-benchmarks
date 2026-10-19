"""Key-value retrieval over a shared prefix, to exercise radix-cache hits.

usage: glm53_dcp_radix_needle.py <port> [n_pairs] [n_questions] [seed]
Every question after the first reuses the cached key-value list.
"""

import json
import random
import sys
import urllib.request

port = sys.argv[1]
n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
n_q = int(sys.argv[3]) if len(sys.argv) > 3 else 20
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)

keys = rng.sample(range(10000, 99999), n_pairs)
pairs = {f"K{k}": f"{rng.randint(100000, 999999)}" for k in keys}
doc = "Memorize this table of codes.\n" + "\n".join(
    f"{k} -> {v}" for k, v in pairs.items()
)


def ask(key):
    body = {
        "model": "x",
        "messages": [
            {
                "role": "user",
                "content": f"{doc}\n\nWhat code does {key} map to? "
                "Reply with only the 6-digit code.",
            }
        ],
        "max_tokens": 1024,
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


hits = 0
for i, key in enumerate(rng.sample(list(pairs), n_q)):
    ans, n_prompt, cached = ask(key)
    ok = pairs[key] in ans
    hits += ok
    print(f"q{i:02d} {key} want={pairs[key]} got={ans[-60:]!r} ok={ok} prompt={n_prompt}")
print(f"port {port}: {hits}/{n_q} correct")
