# Agentic helper scripts

Reading tools, not benchmark harnesses. The sweep itself is
`../ix_agentx_glm53flash.sh` plus a `../recipe_glm53flash_fp4_<chip>_sglang_mtp.sh`;
nothing here launches a server. The one-shot A/B drivers that used to sit
alongside these were removed in 1661b3c and remain in git history.

| Script | Answers |
|---|---|
| `ix_agentx_summarize.py` | How fast was it? One row per point, on the board's two axes. |
| `check_health.py` | May I quote it? Scheduler crashes, `submission_valid`, error rate. |
| `window_sensitivity.py` | What would a shorter window have reported? No GPU time. |
| `monitor.sh` | What is running right now? One line per live sweep, per interval. |

```bash
./helper/ix_agentx_summarize.py --hw b200 <result-dir>...   # b200 | mi355x | b300
./helper/check_health.py <sweep-dir>
IX=/home/jacchang/InferenceX ./helper/window_sensitivity.py --windows 600,1200,3600 <sweep-dir>
INTERVAL=600 ./helper/monitor.sh                            # one line per live point
```

Run `check_health.py` before quoting anything. The performance table will
report a perfectly good-looking number for a point whose scheduler died and
restarted mid-run, and it has no column that would tell you.

Two columns read as failure rates and are not. `profiled/all` in the
summariser divides by a denominator that includes warmup records, so it is the
profiled share. `check_health.py`'s `err%` counts errors among profiled
records only, which is what AIPerf's own 10% gate uses; warmup runs at
`max_tokens=1` and AIPerf files every one of those as
`InvalidInferenceResultError`, hundreds per run, all harmless.
