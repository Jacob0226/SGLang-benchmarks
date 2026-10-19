import json, glob, sys
for d in sys.argv[1:]:
    f = glob.glob(d + "/*_local-mi355x.json")[0]
    j = json.load(open(f))
    r = j["request_metrics"]
    t = r["throughput"]; l = r["latency"]
    print(d.split("bench-Agentic-")[1][:60],
          "in=%.0f out=%.0f out/gpu=%.0f ttft_p50=%.2f ttft_p90=%s intvty_p50=%.1f intvty_mean=%.1f dur=%.0f" % (
          t["input"]["tokens_per_second"], t["output"]["tokens_per_second"], t["per_gpu"]["output_tput_tps"],
          l["ttft"]["p50"], l["ttft"].get("p90"), l["intvty"]["p50"], l["intvty"]["mean"], t["duration_seconds"]))
