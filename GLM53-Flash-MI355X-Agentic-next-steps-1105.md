# GLM-5.3-Flash TP4 Agentic conc64：BCG 修正之後的兩個待辦（交接）

日期：2026-11-05（11/06 更新見下一節）

## 11/06 更新：C 做完、B 做完比較，Agentic 差距縮小

**Agentic conc64，全部在 node 150（crsuse2-m2m-v3-150）：**

| 設定 | total throughput / GPU | P90 interactivity | P50 |
|---|---|---|---|
| 11/05 BCG 修正（node 152） | 75,203 | 58.9 | 94.2 |
| + HIP KDA 進 graph | 75,242 / 75,353 | 59.2 / 59.1 | 95.9 / 96.1 |
| + prefill graph 上限 2,560 | 75,797 / 75,432 | 59.7 / 61.0 | 96.1 / 96.2 |
| + TP4 MoE tuned CSV | **76,982 / 76,962** | **63.8 / 62.8** | **101.4 / 101.7** |
| B200（11/04，dgx-025） | 79,585 | 69.0 | 124.2 |

斜線兩邊是兩次獨立的 run。最後兩列的第二次是兩條 lane 同時跑的。和 B200 的差距從 throughput −5.5%、P90 −15% 縮到 −3.3%、−8.3%。

**1. HIP KDA 進 graph（任務 C）：** SGLang commit `b1c1c92cdc`（branch `glm53-1105-bcg-hybrid-dsa`，沒有 push）。
- BCG 原本讓 KDA prefill 退回 Triton `chunk_kda`（4K 新 token 時 10.3 ms，HIP 只要 5.3 ms）。
- 現在 KDA extend 是 graph 裡的一個 eager 斷點，用 batch 的真實 row 數和 token 數跑 gfx950 HIP kernel，和 eager 逐 bit 相同（新的單元測試 `TestBreakableHipKDAPrefillGraph`）。
- GSM8K 96.44%。

**2. prefill graph 上限 2,560（任務 C）：** recipe 已加 `--cuda-graph-max-bs-prefill 2560`。
- 3K 以上的 eager extend 本來就是 GPU 瓶頸（3,072 時 GPU 95 ms、CPU 88 ms），graph 省不到，只多了 bucket padding。
- 小 prefill（cached 99,840 prefix，ms）：

  | 新 token 數 | 16 | 512 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
  |---|---|---|---|---|---|---|---|
  | 不開 BCG | 99.4 | 96.6 | 99.1 | 113.5 | 138.1 | 185.0 | 196.2 |
  | BCG，全部 bucket | 37.1 | 51.6 | 90.7 | 115.7 | 141.1 | 188.8 | 199.0 |
  | BCG，上限 2,560 | 37.4 | 51.9 | 91.2 | 113.7 | 138.5 | 185.0 | 197.3 |
  | 上限 + MoE CSV | 38.4 | 51.9 | 90.5 | 112.8 | 134.6 | 180.6 | 196.0 |
  | B200 預設（全部 bucket） | 55.0 | 64.0 | 90.8 | 106.6 | 128.3 | 163.5 | 199.1 |
  | B200 上限 2,560 | 55.3 | 64.7 | 92.5 | 174.0 | 174.4 | 192.3 | 212.2 |

- 公平性：B200 的 Agentic 本來就開 BCG（SGLang 在 CUDA 的預設）、沒有上限。B200 設上限會讓 3K–6K 慢 30–70 ms，所以 B200 維持預設才是它的最佳設定。

**3. TP4 MoE tuned CSV：** image 裡的 `glm53_fp4_tuned_fmoe.csv` 是另一種切法的 key（inter_dim 2048、288 experts、topk 8），TP4 的 key（512、289、9）完全對不到，所以 TP4 的 MoE 一直在用 aiter 的 untuned default。
- 改用 10/20 調好的 `tmp/moe2_tune/glm53_flash_fp4_tuned_fmoe.csv`（FlyDSL kernel，token 32–16,384），透過 `AITER_CONFIG_FMOE` 載入。
- GSM8K 96.44%。**還沒進 recipe**。
- **aiter 已經有人做過：** [ROCm/aiter#6435](https://github.com/ROCm/aiter/pull/6435)（10/26 合併）加了 `model_configs/glm53flash_fp4_tuned_fmoe.csv`，有 TP4 的 key（token 1–16,384）。它同時改了 `MXFP4_MOE_SUPPORTED_SHAPES` 和 `gen_instances.py`，所以只複製 CSV 不夠，要新版 aiter。1026 和 1102 的 sgl-dev image 都還沒包含它。
- 和我們的 CSV 比（tuning 時間，µs）：decode 大小 upstream 較快（32 token 96.7 對 154.9、128 token 151.4 對 160.2），1K–8K prefill 我們較快（2,048 token 274.7 對 227.8）。下一步是在 node 150 用 aiter main 跑 Agentic conc64 對比。

**4. decode 逐 kernel 比較（任務 B）：** 同條件 fixed-length profile（128K、OSL 300、FP8 KV、golden AL 3.56、MoE CSV 之前）。每個 decode 循環（一次 VERIFY 加 DRAFT）的 GPU kernel 時間（ms）：

| 類別 | MI c32 | B200 c32 | MI c64 | B200 c64 | c64 差 |
|---|---|---|---|---|---|
| allreduce | 3.02 | 2.31 | 5.33 | 2.15 | **+3.18** |
| MoE | 7.79 | 7.50 | 9.73 | 8.31 | **+1.42** |
| other | 0.69 | 0.21 | 0.93 | 0.31 | +0.63 |
| indexer | 2.41 | 2.05 | 4.04 | 3.70 | +0.34 |
| mHC | 1.55 | 1.08 | 2.07 | 1.76 | +0.31 |
| KDA | 2.23 | 2.13 | 4.03 | 3.84 | +0.19 |
| sparse attn | 1.23 | 0.97 | 1.56 | 1.40 | +0.16 |
| GEMM | 3.78 | 4.14 | 4.35 | 4.89 | −0.54 |
| **合計** | 23.97 | 21.57 | 33.47 | 27.83 | **+5.64** |

- **allreduce：** 每次 AR 是 33.7 µs 對 12.4 µs（約 3 MB，384 rows × 4096）。這接近 xGMI 的頻寬上限，是硬體差距。
  - INT4 quick reduce 在 3 MB 時 34.7 µs，沒比 custom AR 快，而且有 12% 相對誤差（`tools/ar_qr_decode_bench.py`）。
  - SGLang 只要 custom AR 能處理就優先用它，所以 decode 本來就不會走 quick reduce。
- **MoE：** 每次呼叫 moe1 106.7 對 97.2 µs，moe2 76.2 對 52.8 µs。上面的 MoE CSV 就是針對這一項。
- 工具：`tools/decode_cycle_breakdown.py`。結果：
  - MI：`results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.22-rocm10-mi35x-20261026/prof-Fixed-MTP-GoldenAL-MXFP4-TP4-1106-i128k-fp8kv`
  - B200：dgx-003 的 `/mnt/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.23-cu130/prof-Fixed-MTP-GoldenAL-NVFP4-TP4-1106-i128k`

**下一步：**
- MoE：用 aiter main（含 #6435）跑 Agentic conc64，和我們的 CSV 比，再決定用哪一份。
- 剩下的 P90 差距主要是 allreduce（每循環 +3.2 ms），在不用 multi stream 的前提下，只能減少 AR 次數或資料量（例如 fuse 進相鄰 kernel）。
- B200 現在在 dgx-003（dgx-025 已連不上）。

## 0. 目前狀態

**做完的事：** MI355X 的 breakable prefill CUDA graph（BCG）以前一直沒生效，現在已修正。

- **原因：** SGLang 在 ROCm 上不讓「有 cached prefix 的 prefill」用 graph，只有 DSA 例外。但判斷是不是 DSA 時只看最外層的 attention backend。GLM-5.3-Flash 最外層是包著 KDA 的 hybrid backend，所以被判成「不是 DSA」，graph 雖然做出來了（佔 17.6 GB），卻從來沒有用上。
- **修正：** `/shared_nfs/jacchang/exp/sglang-fp8kv-1103`，branch `glm53-1105-bcg-hybrid-dsa`，commit `e4cd94409d`。`_attn_backend_is_dsa` 先拆開 `full_attn_backend` 再判斷。這個 commit 建立在 `f0a7a579d5`（FP8 KV Triton）之上，還沒有 push。
- **Recipe：** `Agentic/recipe_glm53flash_fp4_mi355x_sglang_mtp.sh` 已經加上 `--cuda-graph-backend-prefill breakable`，還沒 commit。這個 flag 只有在帶著上面修正的 SGLang 上才有用；沒有修正的話，graph 會做出來但不會被用到。

**Agentic conc64（node 152 同時跑兩條；B200 是 11/04 dgx-025 那一次）：**

| | MI355X 預設 | MI355X BCG 修正 | B200 |
|---|---|---|---|
| total throughput / GPU | 73,098 | 75,203（+2.9%） | 79,585 |
| P90 interactivity | 54.0 | 58.9（+9.1%） | 69.0 |
| TTFT P90 | 1.31 s | 1.23 s | 2.55 s |
| GSM8K | — | 96.44% | — |

結果目錄：`results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.22-rocm10-mi35x-20261026/bench-Agentic-TP4_EP1-mtp5-fp8kv1103-int4-1105-{152base,152bcg}-golden3.56-dsa-triton`。執行腳本：`/shared_nfs/jacchang/agentic_1105/{lane.sh,driver.sh}`。

**小 prefill 的隔離測試（已 cache 99,840 token prefix，每格是 5 次的中位數）：**

| 新 token 數 | MI355X 預設 | MI355X BCG 修正 | B200 |
|---|---|---|---|
| 16 | 101.2 ms | 40.2 ms | 93.0 ms |
| 512 | 99.5 ms | 54.8 ms | 99.6 ms |
| 2,048 | 101.0 ms | 93.2 ms | 114.6 ms |
| 4,096 | 136.2 ms | 146.6 ms | 150.4 ms |

測試工具是 `tools/small_prefill_probe.py`。結果和 trace 在 `/shared_nfs/jacchang/bcg_1105/` 和 `/shared_nfs/jacchang/chunk_1104/`（`trace_mi_small_base`、`trace_b200_small`）。

剩下的差距（throughput 落後 5.5%，P90 落後 15%）主要來自下面第 1 項。

## 1. 待辦 B：decode 每一步比 B200 慢 1.3 倍

**證據：**

- **Agentic（同樣 decode batch，一步 VERIFY 加上接著的 DRAFT 的循環時間）：**

  | decode batch | MI355X | B200 | 倍數 |
  |---|---|---|---|
  | 25–32 | 26.5 ms | 20.5 ms | 1.29× |
  | 33–40 | 30.1 ms | 23.7 ms | 1.27× |
  | 41–48 | 33.4 ms | 25.3 ms | 1.32× |

- **Fixed-length：** 兩邊都用 decode CUDA graph，GPU 忙碌比例 97–99.5%，所以不是 CPU 的問題，是 GPU 計算慢。
  - MI355X：70K、conc32 每步 VERIFY 19.6 ms；conc64 每步 25.8 ms。
  - B200：128K、conc32 每步 16.3 ms。B200 的 context 更長，卻比較快。

**現有 fixed-length profile 不能直接比，原因有兩個：**

- **MI355X**（`prof-Fixed-MTP-Mono-GoldenAL-MXFP4-TP4-1026-prstack1030-t1`）：ISL 70K，`kv_cache_dtype=bfloat16`。Agentic 用的是 FP8 KV。
- **B200**（`prof-Fixed-MTP-GoldenAL-NVFP4-TP4-1104-i128k`）：ISL 128K，FP8 KV，只跑了 conc32。
- 兩邊都是 `enable_amd_monokernel=False`，這一點和 Agentic 一致，可以比。

**要做的事：**

1. **兩邊用同樣條件重跑 fixed-length profile：** 用 `BenchFixedLength/GLM.sh`，同 ISL（建議 128K；也可選 Agentic 的平均 ISL，約 130K）、OSL 300、conc 32 和 64。
   - MI355X 要用 FP8 KV、Triton DSA、INT4 quick reduce，`PYTHONPATH` 指向 fp8kv tree（現在是 `glm53-1105-bcg-hybrid-dsa`）。
   - 跑之前要確認 GLM.sh 的 MI355X 設定和 Agentic recipe 一致（mamba ratio、HiCache 不影響 decode）。
2. **做 kernel 對 kernel 的比較：** 用 graph 模式的 trace 拿每步時間，用 `no-cuda-graph` 的 trace 拆分類。
   - 分類可以沿用 `tools/prefill_chunk_breakdown.py` 的 indexer、sparse_attn、moe、allreduce、kda、mhc、gemm、elementwise。
   - 看每一類 MI355X 比 B200 多幾 ms，排出前 3 名。
3. **已知線索（還沒驗證）：**
   - 大 chunk 的 prefill 裡，MI355X 的 DSA indexer 慢 1.37 倍、MoE 是 60 對 36 ms；all-reduce 和 GEMM 則是 MI355X 比較快。decode 的分布不一定一樣，要重新量。
   - aiter 對 MoE M=32…16384 都印出 "no tuned FlyDSL config"，用的是 heuristic。decode 的 M 很小，可能值得 tune。
4. **限制：** 不考慮 multi stream。

## 2. 待辦 C：讓 BCG 也能幫到 4K–16K 的 prefill

**現象：**

- **4,096 個新 token 反而變慢：** 從 136 ms 變成 146 ms。
- **2,048 個只快 8%。**
- **大 prefill 完全用不到 graph：** 新 token 超過 8,192 時不會用 graph，因為 prefill graph 最多只做到 8,192 tokens。
- **在 Agentic 的影響：** 後段長 prompt 比較多，這段 BCG 版的 graph 使用率從大約 80% 降到 62%，throughput 的領先也縮成 0。

**要查的事：**

1. **4K 為甚麼變慢：** 用 `tools/small_prefill_probe.py --new 3072,4096,6144,8192`，比較 BCG 和預設的 trace（`trace_*` 目錄每個長度各一份）。要看三件事：
   - **padding：** 4,256 個 token 會被補到哪一個捕捉大小（`_pad_to_bucket` / `capture_num_tokens`）。
   - **breakable 每一段的切換成本：** 看 graph 段落之間的空檔。
   - **eager 段落：** DSA indexer 或 sparse MLA 在 breakable 模式下是不是跑成 eager，而且比較慢。
2. **按長度切換：** 如果 4K 以上本來就不划算，可以試著把 prefill graph 的上限降到 2,048 或 4,096，順便省下一部分 17.6 GB。需要找 SGLang 對應的 server arg（`cuda_graph_config.py` 和 `prefill_cuda_graph_runner.py` 的 `max_num_tokens`、`capture_num_tokens`）。
3. **8K–16K：** 這個區間是 GPU 瓶頸（16K chunk 的 GPU 時間約 275 ms，CPU 約 140 ms），graph 的幫助有限，優先度最低。

## 3. 環境備忘

- **MI355X：** image `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261026`。
  - 模型在 `/shared_nfs/jacchang/models/amd/GLM-5.3-Flash-Quark-MXFP4`（`/shared_nfs/models/amd/` 也有一份），container 掛成 `/data/huggingface/hub`。
  - 建 container 參考 `/shared_nfs/jacchang/bcg_1105/setup.sh`；sgl-eval 要另外裝 `a231b7a4`。
  - GSM8K 用 1,319 threads 時，要先 `ulimit -n 65536`。
- **B200：** dgx-025，container `jacchang_GLM53-Flash`，透過跳板機 193.187.68.23 連線。B200 的小 prefill 測試腳本在 `/mnt/home/jacchang/chunk_1104/b200_small.sh`。
- **跑任何東西之前先問；不要 push。**
