# GLM-5.3-Flash MI355X vs B200：決定 P90 Int 的 decode batch（bs6 / bs16）逐 kernel 對照（11/11）

撰寫：2026-11-11 13:30（台灣時間）。兩邊都用最新 CSV 的設定（ReplaySSM 開），今天在 node 152 和 dgx-029 重新 profile。不考慮 multi stream。

## 0. 結論

1. **conc16 的 P90 跑在 decode batch 6（36 verify row），conc32 的 P90 跑在 decode batch 14–16（96 verify row）。** 要提升這兩個點的 P90 Int，應該看 bs6 和 bs16，不是 bs48。
2. **差距全部在 45 層 target verify。** tail（lm_head、取樣、ReplaySSM commit、5 次 MTP draft）兩邊打平，bs16 時 MI355X 還快 0.3–0.5 ms。下表**全部是 wall time**（GPU timeline 上從開始到結束的時間），B200 的 PDL 和 side stream 重疊都已經算進去。

   | 每個 cycle 的 wall（ms） | MI355X 45 層 | B200 45 層 | 差 | MI355X tail | B200 tail | MI355X cycle | B200 cycle | Agentic 實測 chunk gap（MI355X / B200） |
   |---|---|---|---|---|---|---|---|---|
   | bs6 | 10.51 | 8.00 | **+2.51** | 3.05 | 3.01 | 13.58 | 11.02 | 13.41 / 11.45 |
   | bs16 | 14.46 | 10.09 | **+4.37** | 3.66–3.87 | 4.14 | 18.1–18.3 | 14.3 | 17.80 / 14.21 |

   - cycle = pre + 45 層 + tail，從這次 verify 的第一個 kernel 到下一次 verify 的第一個 kernel，中間 GPU 沒事做的時間（CPU 排程、launch）也算在裡面。
   - bs16 的 tail 和 cycle 已經扣掉 rank 不同步的等待（第 2 節）。
   - 兩邊的 cycle wall 都跟 Agentic 實測的 chunk gap 對得上，所以這組 profile 可以代表 P90 的 request。
3. **MI355X 的 GPU 沒有空檔，每層的 wall 就等於 kernel 加總。B200 每層的 wall 比 kernel 加總少 15–20%。** 主要靠同一條 stream 上的 PDL（下一個 kernel 提早啟動，跟前一個 kernel 的尾巴疊在一起），其次是 shared expert 放在 side stream。
4. **MI355X 的小 kernel 每個至少 4.2–4.5 µs。** `compare_scalar`、bf16→fp32 copy 這種幾乎不做事的 kernel 也一樣。B200 同樣的 kernel 實際只佔 0.2–3 µs。每層 KDA+MoE 有 14–15 個這種小 kernel，bs16 時合計約 72 µs，B200 對應的部分只佔約 45 µs。**不用 multi stream，最直接的方法就是 fusion，每拿掉一個 kernel 約省 4.3 µs。**
5. **依類別看每個 cycle 的差距**（第 5 節）：
   - bs6：KDA core 0.44 ms、DSA 的 MLA 投影和 quant 0.37、KDA 投影 0.29、MoE router 到 sort 0.28、mHC 0.24、allreduce 0.22、indexer topk 0.20。
   - bs16：MoE GEMM 0.67、KDA core 0.64、allreduce 0.64、router 到 sort 0.52、KDA 投影 0.50、MLA 投影 0.36、indexer logits 0.31、topk 0.28。
6. **一個新發現：ReplaySSM 在 B200 上讓 KDA kernel 減半，在 MI355X 上幾乎沒變。**
   - B200 的 `cutlass_kda_decode_mtp`：bs16 從 27.1 降到 14.5 µs，bs6 從 16.6 降到 9.5 µs。
   - MI355X 的 Triton `fused_sigmoid_gating_delta_rule_update`：bs16 只從 31.4 降到 26.1 µs，bs6 從 16.9 降到 14.8 µs。
   - 如果 MI355X 也能做到 B200 的比例，bs16 每 cycle 約省 0.4 ms（2.2%），bs6 約省 0.18 ms（1.3%）。這是第 8 節的第一項。
   - 11/12 已解決（第 13 節）：新的 FlyDSL kernel 把 KDA verify 降到 bs6 7.4 µs、bs16 12.3 µs，比 B200 的 9.4 和 14.5 還快。
7. **要追上的量：**
   - P90 Int +5% 大約要 cycle 短 5%，也就是 bs6 省 0.66 ms、bs16 省 0.89 ms。
   - 要追平 B200 的 P90 Int（conc16 +14%、conc32 +11%），要拿掉 bs6 的 45 層差距約 2.0 ms、bs16 約 2.3 ms。所以光靠一兩個 kernel 不夠，要整層一起 fuse。

## 1. 哪個 decode batch 決定 P90 Int

用最新 CSV 那兩條 ReplaySSM line 的原始資料跑 `tools/agentic_tail.py`。決定 P90 的是 TPOT 排在 P85–P95 的 request：

| | conc16 MI355X | conc16 B200 | conc32 MI355X | conc32 B200 |
|---|---|---|---|---|
| P90 Int | 224.8 | 256.8 | 148.4 | 164.4 |
| P85–P95 的 TPOT（ms） | 4.50 | 3.94 | 6.85 | 6.18 |
| 其中 decode step（ms / token） | 3.87 | 3.05 | 5.22 | 3.80 |
| 其中被 prefill 卡住（ms / token） | 0.64 | 0.89 | 1.63 | 2.38 |
| 中位 chunk gap（ms） | 13.41 | 11.45 | 17.80 | 14.21 |
| decode batch 平均 | 5.97 | 5.64 | 14.29 | 12.57 |
| decode batch p10 / p50 / p90 | 3 / 6 / 8 | 3 / 6 / 8 | 8 / 16 / 19 | 6 / 14 / 17 |
| context（ISL 平均） | 144K | 139K | 150K | 151K |

- 兩個 conc 都是 B200 被 prefill 卡住的時間比較多，MI355X 輸的是 decode step 本身：conc16 慢 27%，conc32 慢 37%。
- conc32 的 MI355X 平均 batch 比 B200 大（14.3 對 12.6），因為 decode 比較慢，同時在 decode 的 request 就比較多。所以 bs16 是對 MI355X 比較準的代表點。
- 資料來源：
  - MI355X：`/shared_nfs/jacchang/agentic_1110_rep/rep/`
  - B200：`/shared_nfs/jacchang/b200_agentic/dgx029_1110/bench-Agentic-TP4-v0523-1108full-replayssm-golden3.56/`

### 1.1 decode batch 和 chunk gap 怎麼算

- **哪些 request 決定 P90**：P90 Int = 1 / p90(TPOT)。每個 request 的 TPOT 由 AIPerf 算出：(E2EL − TTFT) / (OSL − 1)。依 TPOT 排序，取 P85–P95 的 request（conc16 是 265 個）。
- **decode batch**：
  - TP0 的 server log 每隔一段 decode step 會印一行 `Decode batch, #running-req: N`，conc16 整個 run 有 7,215 行，時間戳精度 1 秒。
  - 對每個 request，取它的 decode 時間（第一個 token 到結束，平均 OSL 873、約 3–4 秒）裡的這些行，算 #running-req 的平均。
  - 265 個 request 的平均值分佈是 p10 / p50 / p90 = 3 / 6 / 8，平均 5.97。所以 bs6 是中心，不是唯一值；實際在 bs3–8 之間。
  - 這裡的 p90 = 8 是「這 265 個 request 的平均 batch」的第 90 百分位，不是 P90 Int 對應的 batch。這些 request 的 TPOT 是整段 decode 時間的平均，每個 request 在不同時間會跑在不同的 batch。
- **P90 request 的 decode 時間花在哪些 batch**：把每個 request 視窗內的 log 行攤平，每個 request 權重 1。log 行數約正比於 step 數，大 batch 的 step 比較久，所以大 batch 會被稍微低估。

  | | p10 | p25 | p50 | p75 | p90 | 主要分佈 |
  |---|---|---|---|---|---|---|
  | MI355X conc16 | 3 | 4 | **6** | 8 | 9 | bs4 13%、bs5 15%、bs6 15%、bs7 13%、bs8 11%、bs9 10% |
  | B200 conc16 | 2 | 4 | 6 | 7 | 9 | 幾乎相同 |
  | MI355X conc32 | 7 | 11 | **15** | 18 | 20 | bs11–20 合計約 75%，bs16 最多（11%） |
  | B200 conc32 | 6 | 9 | 13 | 16 | 18 | |

  - TPOT 是各個 batch 的 step 時間依時間比例加權。只優化 bs8，只會影響 conc16 P90 request 約 11% 的時間。
  - 所以要優化的是一個區間：conc16 是 bs4–9（24–54 verify row），conc32 是 bs11–20（66–120 row）。bs6 和 bs16 是這兩個區間的中心，所以拿來做 kernel 分析。
  - 驗證時要看區間兩端，因為 kernel 會隨 M 換路徑：in-proj 在 M=36 走 FlyDSL、M=96 走 hipBLASLt；fused sort_quant 在 bs6 有走、bs16 沒走。microbench 建議至少量 M = 24、36、48、54 和 66、96、120。
- **chunk gap**：
  - AIPerf 記錄每個 request 相鄰兩個 stream chunk 的間隔（`inter_chunk_latency`）。每個 chunk 平均 3.56–3.59 個 token，跟 golden AL 3.56 相同，所以一個 chunk 就是一次 verify cycle。
  - 先取每個 request 自己 gap 的中位數，再對 265 個 request 平均，得到 13.41 ms。
  - 用中位數是因為它不受偶爾一次的 prefill 卡住影響。
- **跟 profile 對照**：
  - profile 的 cycle wall 是從一次 verify 的第一個 kernel 到下一次 verify 的第一個 kernel，跟 chunk gap 量的是同一件事，只是一個從 client 量、一個從 GPU 量。
  - conc6 的 fixed-length profile 正好是 6 個 request 同時 decode，也就是 bs6。
  - 結果：MI355X 13.58 對 13.41 ms（+1.3%），B200 11.02 對 11.45 ms（−3.8%）。
  - bs16 的 profile 對 conc32（Agentic 平均 bs14.3 / 12.6）：MI355X 18.1–18.3 對 17.80，B200 14.3 對 14.21。
  - 這是一致性檢查，不是證明。它說明 P90 request 的 decode step 時間落在 bs6 / bs16 profile 的範圍內，所以用這兩個點的 kernel 分析是對的。

## 2. 資料與方法

| 項目 | MI355X | B200 |
|---|---|---|
| 節點 | node 152（`jacchang_pr48875`） | dgx-029（`jacchang_prof1111_on` / `_off`） |
| 軟體 | `wt-glm53-1106-prstack` 09111fe9a3（10 個 open PR）、aiter #6435 patch 和 MoE CSV、PTPC / Hadamard env、Triton DSA、FP8 KV、breakable prefill graph、INT4 quick reduce。**11/17 起 #44661 從 stack 拿掉**（跟 #44417 疊在一起有 double Hadamard bug，第 16.2 節）：新的 stack 是 `wt-glm53-1117-prstack` `6fd8d4b50a`（9 個 open PR），`SGLANG_DSA_FUSE_HADAMARD_QUANT` 不要設 | `lmsysorg/sglang:v0.5.23-cu130` stock |
| profile | `GLM.sh --mtp --golden-al --prof`，ISL 128000 / OSL 300，MTP5，golden AL 3.56，`--enable-linear-replayssm-spec`，mem-fraction 0.85、chunked 16384、mamba ratio 0.9 | 同樣的 GLM.sh 參數加 ReplaySSM |
| 對照 | 同時跑了 ReplaySSM 關的 bs16 | 同時跑了 ReplaySSM 關的 bs16 |

- 這次的 server 設定跟 CSV 那條 MI355X ReplaySSM line 的 conc 16 / 32 完全相同（已對照 server log 的 `server_args`）。
- 分析工具是新寫的 `tools/decode_layer_wall.py`。切層規則跟 `decode_layer_table.py` 相同（在 mHC attention-pre GEMM 切開），另外加上 timestamp：
  - **wall**：這一層第一個 kernel 開始，到下一層第一個 kernel 開始。
  - **busy**：這段時間裡 GPU 至少有一個 kernel 在跑的時間。
  - **實際佔用（excl）**：依 GPU 開始順序，每個 kernel 讓 busy 往後延長了多少。全部加起來就是 busy。B200 有 PDL，trace 裡的 kernel duration 包含等前一個 kernel 的時間，所以**逐 kernel 對照一律用 B200 的實際佔用**，括號裡才是原始 duration。MI355X 沒有重疊，實際佔用就等於 duration。
- **哪些數字是 wall**：
  - 第 0、3 節的層、45 層、tail、cycle 都是 wall。
  - 第 4、5 節的逐 kernel 和類別數字是實際佔用，加起來等於 busy。45 層裡 GPU 沒事做的時間很少（B200 bs6 0.056 ms、bs16 0.055 ms；MI355X 0.001 ms），所以 busy 幾乎等於 wall。B200 的 PDL 和 side stream 重疊也已經扣掉，不會讓 B200 看起來比實際慢。
  - B200 的 kernel 加總（例如 bs6 45 層 9.58 ms，wall 只有 8.00 ms）只用來說明重疊了多少，不拿來比。
- 每個 trace 有 2 個完整 cycle，每種層的數字是所有層、所有 cycle 的平均。B200 的 kernel 序列在部分層不一樣（例如 bs6 KDA+MoE 層只有 7 / 62 層完全相同），逐 kernel 表只用序列相同的那些層。
- **雜訊**：有幾次 allreduce 等了 1.7–3.4 ms，是 profiler 造成的 rank 不同步，不是 kernel 慢。這些都落在 45 層之外，bs6 的兩邊都沒有。

  | bs16 | cycle 0 | cycle 1 | 採用的 tail wall |
  |---|---|---|---|
  | MI355X tail | 5.54（含一次 1.67 ms allreduce 等待） | 3.66 | 3.66–3.87 |
  | B200 pre | 3.42（一次 3.41 ms allreduce 等待） | 0.02 | |
  | B200 tail | 6.10（GPU 空了 2.48 ms） | 7.25（eh_norm 和 allreduce 一起等了 3.11 ms） | 4.14（cycle 1 扣掉等待） |

  B200 cycle 0 的 tail 空了 2.48 ms，其他 cycle 都只空約 0.5 ms，所以不採用。採用 4.14 ms 時，cycle 是 14.3 ms，跟 Agentic 實測的 chunk gap 14.21 ms 對得上。

### 2.1 扣掉這些等待合理嗎？會不會是 MI355X P90 比較差的原因？

- **這是什麼**：rank 0 的 GPU 卡在 allreduce，等其他 rank。其他 rank 晚到，通常是 CPU 端晚 launch（profiler、GC、排程）。只有 TP-0 的 trace，所以看不到是哪個 rank 晚到、為什麼晚到。「profiler 造成」是推測，沒有證據。
- **profile 裡兩邊都有，B200 更多**：bs16 的 trace（ReplaySSM 開和關）兩邊各有 4 個 cycle。
  - MI355X 有 2 次：一次 allreduce 等了 1.67 ms；另一次（ReplaySSM 關的 cycle 0）tail 裡 GPU 空了 2.06 ms，是 CPU 端晚 launch，其他 cycle 只空 0.17 ms。
  - B200 有 4 次：allreduce 等了 3.41 ms 和 1.94 ms、eh_norm 和 allreduce 一起等了 3.11 ms、tail 裡 GPU 空了 2.48 ms。
  - bs6 的兩邊都沒有。不扣的話，B200 bs16 的 cycle 平均是 18.48 ms，跟 Agentic 的 14.21 ms 完全對不上，所以 profile 裡的等待不是 Agentic 的常態。
- **Agentic 的 chunk gap 分佈**（P85–P95 的 request，每個 gap 除以該 request 的中位 gap）：

  | | p25 | p75 | p90 | 落在中位數 1.08–2 倍的 gap | 2 倍以內的平均 / 中位數 |
  |---|---|---|---|---|---|
  | MI355X conc16 | 0.94 | 1.06 | 1.24 | 12.2% | 1.027 |
  | B200 conc16 | 0.88 | 1.10 | 1.32 | 18.7% | 0.952 |
  | MI355X conc32 | 0.94 | 1.06 | 1.44 | 12.7% | 1.047 |
  | B200 conc32 | 0.70 | 1.13 | 2.38 | 19.7% | 0.968 |

  - 1.7 ms 的等待在 bs6 是 cycle 的 +13%，在 bs16 是 +9%。如果像 profile 那樣每兩個 cycle 就有一次，會有約一半的 gap 落在 1.08 倍以上，p75 也會超過 1.1。實際上 MI355X 只有 12–13%（這還包括 batch 變大、context 變長等其他原因），p75 只有 1.06。MI355X 的 gap 反而比 B200 穩定。
  - 即使把 MI355X 所有 2 倍以內、高於中位數的部分都當成等待，上限是 conc16 每個 cycle 0.36 ms（2.7%）、conc32 0.83 ms（4.7%）。中位 gap 不受這些影響，本身就慢了 17% / 25%（13.41 對 11.45、17.80 對 14.21）。
  - B200 的平均比中位數低（< 1），是因為一個長 gap 之後 client 會連續收到幾個 chunk，產生很短的 gap。所以兩邊的「平均 / 中位數」不能直接相減。
- **結論**：profile 裡的等待在 Agentic 不常發生，不是 MI355X P90 落後的主因，扣掉它們來比 decode step 是合理的。conc32 那 4.7% 的上限值得之後再查。要確認，可以在 Agentic 跑的時候用 `--prof` 抓所有 rank 的 trace，數 allreduce 等待的頻率。

## 3. 一個 cycle 依層別拆開

每層的 wall（µs）：

| 層 | 層數 | MI355X bs6 | B200 bs6 | 差 × 層數（ms） | MI355X bs16 | B200 bs16 | 差 × 層數（ms） |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 201.1 | 158.3 | **+1.33** | 280.9 | 198.3 | **+2.56** |
| DSA+MoE | 11 | 351.8 | 255.7 | **+1.06** | 477.4 | 331.5 | **+1.60** |
| KDA+dense | 3 | 134.2 | 93.3 | +0.12 | 165.4 | 99.5 | +0.20 |
| **45 層** | | 10.51 ms | 8.00 ms | **+2.51** | 14.46 ms | 10.09 ms | **+4.37** |

B200 每層 kernel 加總和 wall 的差（µs / 層，也就是重疊掉多少）：

| 層 | bs6 kernel 加總 → wall | 同 stream（PDL） | side stream | bs16 kernel 加總 → wall | 同 stream（PDL） | side stream |
|---|---|---|---|---|---|---|
| KDA+MoE | 190.0 → 158.3 | 18.8 | 13.3 | 236.7 → 198.3 | 25.5 | 13.3 |
| DSA+MoE | 306.7 → 255.7 | 19.2 | 35.5 | 389.6 → 331.5 | 25.9 | 36.0 |

side stream 這一欄是被主 stream 蓋掉的 side stream kernel 時間（kernel 加總 − busy − 同 stream 重疊）。KDA+MoE 層是 shared expert；DSA+MoE 層是 indexer 的 `wk` / `weights_proj` 和 k-pool 寫入。

## 4. 一層 decode kernel 依序對照

每個 MoE 層都分成兩段：

- **attention 段**：從 mHC pre 到 attention 後的 allreduce 和 mHC post。KDA 層和 DSA 層只差在這一段。
- **MoE 段**：從 FFN 的 mHC pre 到 MoE 後的 allreduce 和 mHC post。KDA 層和 DSA 層用的是同一組 MoE kernel，所以放到 4.5 一起看。

每一段的實際佔用（µs / 層）和乘上層數後的差距（ms / cycle）：

| 段 | 層數 | bs16 MI355X | bs16 B200 | bs16 差 × 層數 | bs6 MI355X | bs6 B200 | bs6 差 × 層數 |
|---|---|---|---|---|---|---|---|
| KDA attention | 31 | 106.4 | 60.1 | **+1.44** | 79.7 | 50.7 | **+0.90** |
| DSA attention | 11 | 301.0 | 188.8 | **+1.23** | 205.2 | 126.5 | **+0.87** |
| MoE（在 KDA 層） | 31 | 174.5 | 137.8 | **+1.14** | 121.4 | 110.3 | +0.34 |
| MoE（在 DSA 層） | 11 | 176.4 | 131.4 | +0.50 | 98.0 | 87.9 | +0.11 |
| DSA 層的 allreduce + mHC（bs6 沒有分段） | 11 | | | | 48.6 | 36.7 | +0.13 |
| **合計** | | | | **4.31** | | | **2.35** |

- bs16 的差距：attention 2.67 ms，MoE 1.64 ms。bs6 的差距：attention 約 1.8–1.9 ms，MoE 約 0.45–0.6 ms。
- **兩個 batch 都是 attention 輸比較多。** MoE 只有在 bs16 才明顯輸，主要是 GEMM1 和 router 到 sort（見 4.5）。
- bs6 的 DSA 層原本只有類別合計，allreduce 和 mHC 沒有分到 attention 或 MoE，所以單獨列一行。這一行的 MoE 只有 router 到 sort 和 MoE GEMM。

### 4.1 KDA attention，bs16（96 verify row）

MI355X 的順序就是 GPU 實際執行的順序。B200 對齊到同一個功能，數字是實際佔用，括號是原始 duration。

| # | 功能 | MI355X kernel | µs | B200 kernel | µs |
|---|---|---|---|---|---|
| 1 | mHC pre GEMM + sqrsum | `mhc_pre_gemm_sqrsum` | 4.61 | `hc_prenorm_gemm`（tf32） | 4.14 |
| 2 | mHC pre + RMSNorm | `mhc_pre_big_fuse_rmsnorm` | 5.79 | `mhc_pre_big_fuse_with_norm` | 6.03 |
| 3 | KDA in-proj（N=6416、K=4096，BF16） | Cijk MT32x96x256（hipBLASLt） | **23.77** | nvjet 96x64 | 12.97 |
| 4 | f_b / g_b proj | Cijk MT64x32x64 | 5.53 | nvjet 64x48 | 2.38（3.99） |
| 5 | compare_scalar | elementwise | 4.39 | elementwise | 1.72 |
| 6 | conv1d update | `_causal_conv1d_update` | 4.99 | （併在 kda_decode_mtp 裡） | — |
| 7 | KDA delta rule | `fused_sigmoid_gating_delta_rule_update`（Triton） | **26.11** | `cutlass_kda_decode_mtp` | 14.52 |
| 8 | gated o_norm | `layer_norm_gated_fwd` | 4.30 | `layer_norm_gated_fwd` | 2.82（8.49） |
| 9 | o_proj | FlyDSL `hgemm_bf16_t64x32x128` | 8.59 | nvjet 64x48 | 6.56 |
| 10 | allreduce | `cross_device_reduce_2stage` | **13.58** | `all_reduce_2shot_push` | 6.01（7.66） |
| 11 | mHC post | `mhc_post` | 4.76 | `mhc_post_tilelang` | 2.93（7.52） |
| | **合計** | 11 個 kernel | **106.4** | 10 個 | **60.1** |

| 類別 | MI355X | B200 | 差 | × 31（ms） |
|---|---|---|---|---|
| KDA core（compare、conv、delta、o_norm） | 39.8 | 19.1 | **+20.7** | 0.64 |
| KDA 投影（in、f_b/g_b、o） | 37.9 | 21.9 | **+16.0** | 0.50 |
| allreduce | 13.6 | 6.0 | **+7.6** | 0.24 |
| mHC pre + post | 15.2 | 13.1 | +2.1 | 0.06 |
| **合計** | 106.4 | 60.1 | **+46.3** | **1.44** |

- **in-proj 在 bs16 走到 hipBLASLt Cijk，要 23.8 µs。** bs6（M=36）走 FlyDSL `hgemm_bf16_t64x64x64x6_kst4`，只要 14.9 µs。權重 52.6 MB，23.8 µs 等於 2.2 TB/s，還是 weight-bound 的區間，照理不該比 bs6 慢這麼多。要確認 tuned CSV 有沒有涵蓋 M=72–120。
- KDA core 的 5–7 號是第 13 節重寫的對象。

### 4.2 DSA attention，bs16

B200 這層有 51 個 kernel，很多是 1–3 µs 的轉換或 metadata kernel，所以依功能分組。

| # | 功能 | MI355X kernel | µs | B200 kernel | µs |
|---|---|---|---|---|---|
| 1–2 | mHC pre | gemm_sqrsum + big_fuse_rmsnorm | 4.66 + 5.81 | prenorm + big_fuse | 4.14 + 5.98 |
| 3–4 | q_a + kv_a（K=4096） | `dynamic_per_group_scaled_quant` + CK fp8 blockscale | 4.24 + 10.01 | TgvGemm（BF16） | 6.85 |
| 5 | q / kv RMSNorm | `fused_qk_rmsnorm` | 4.43 | RMSNorm × 2 | 1.96 + 0.30 |
| 6–7 | q_b | quant + CK fp8 blockscale | 4.26 + 9.77 | nvjet | 6.01 |
| 8–14 | indexer 投影、k LayerNorm、Hadamard、quant、k-pool 寫入 | Cijk fp32 8.59、LN 4.62、Cijk 9.89、Hadamard 4.40、kpool 寫入 6.68、copy 4.34、`_hadamard_quant` 4.74 | 43.26 | nvjet + splitK × 2（side stream）、simt sgemm fp32、splitKreduce、Hadamard、act_quant、kpool 寫入（side stream） | 32.77 |
| 15 | **indexer logits** | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | **65.71** | `sm100_paged_mqa_logits` | 37.77 |
| 16 | **k-pool top-512** | `kpool_topk_transform<512>` | **40.00** | `topk_persistent_cluster` + `topk_main` | 1.80 + 13.20 |
| 17 | W_UK absorb | Cijk MT64x48x128 | 7.99 | nvjet 128x48 | 3.16 |
| 18 | KV FP8 寫入 | `set_mla_kv_buffer_fp8_quant` | 4.49 | fp8 convert × 3 + `set_mla_kv_buffer_norope` | 7.42 + 1.77 |
| 19–20 | sparse MLA | `_sparse_mla_split_k` + `_sparse_mla_reduce` | **41.23 + 7.19** | metadata × 3 + `fmhaSm100f` sparse | 5.20 + 35.89 |
| 21 | W_UV | Cijk MT256x96x64 | **14.07** | nvjet 64x48 | 4.13 |
| 22–23 | o_proj | quant + CK fp8 blockscale | 4.20 + 11.02 | nvjet 64x64 | 9.77 |
| 24–25 | allreduce + mHC post | | 13.94 + 4.71 | | 7.74 + 3.01 |
| | **合計** | 25 個 kernel | **301.0** | | **188.8** |

| 類別 | MI355X | B200 | 差 | × 11（ms） | 主要原因 |
|---|---|---|---|---|---|
| MLA 投影 + quant + KV 寫入 | 74.5 | 41.4 | **+33.1** | 0.36 | 3 個獨立 quant（12.7 µs）；W_UK + W_UV 22.1 對 7.3 µs |
| indexer logits | 65.7 | 37.8 | **+27.9** | 0.31 | 同一個 request 的 6 個 verify row 各讀一次 K |
| k-pool topk | 40.0 | 15.0 | **+25.0** | 0.28 | |
| indexer 投影等 | 43.3 | 32.8 | +10.5 | 0.12 | B200 有 2 個 GEMM 和 k-pool 寫入在 side stream |
| sparse MLA | 48.4 | 41.1 | +7.3 | 0.08 | |
| allreduce | 13.9 | 7.7 | +6.2 | 0.07 | |
| mHC pre + post | 15.2 | 13.1 | +2.1 | 0.02 | |
| **合計** | 301.0 | 188.8 | **+112.2** | **1.23** | |

B200 這層只有 13 / 22 層的 kernel 序列相同，這裡只用這 13 層，它們的實際佔用比所有層平均少 7 µs。所以 DSA 層的兩段加起來（1.73 ms）比第 3 節的 1.60 ms 大一些。

### 4.3 KDA attention，bs6（36 verify row）

| # | 功能 | MI355X kernel | µs | B200 kernel | µs |
|---|---|---|---|---|---|
| 1–2 | mHC pre | gemm_sqrsum + big_fuse_rmsnorm | 4.40 + 5.58 | prenorm + big_fuse | 4.00 + 5.83 |
| 3 | KDA in-proj | FlyDSL `hgemm_bf16_t64x64x64x6_kst4` | **14.91** | TgvGemm 64x64 | 10.68 |
| 4 | f_b / g_b | Cijk MT32x16x64 | 4.54 | nvjet 64x24 | 2.19 |
| 5 | compare_scalar | | 4.38 | | 1.76 |
| 6 | conv1d update | `_causal_conv1d_update` | 4.62 | （併在 kda_decode_mtp 裡） | — |
| 7 | KDA delta rule | Triton | **14.76** | `cutlass_kda_decode_mtp` | 9.42 |
| 8 | gated o_norm | `layer_norm_gated_fwd` | 4.38 | | 2.71（6.58） |
| 9 | o_proj | Triton `_gemm_a16_w16` | 8.77 | TgvGemm 64x32 | 5.98 |
| 10 | allreduce | `cross_device_reduce_2stage` | 8.92 | `all_reduce_1shot_push` | 6.18 |
| 11 | mHC post | | 4.46 | | 1.99 |
| | **合計** | 11 個 | **79.7** | 10 個 | **50.7** |

| 類別 | MI355X | B200 | 差 | × 31（ms） |
|---|---|---|---|---|
| KDA core | 28.1 | 13.9 | **+14.3** | 0.44 |
| KDA 投影 | 28.2 | 18.9 | **+9.4** | 0.29 |
| allreduce | 8.9 | 6.2 | +2.7 | 0.08 |
| mHC pre + post | 14.4 | 11.8 | +2.6 | 0.08 |
| **合計** | 79.7 | 50.7 | **+29.0** | **0.90** |

### 4.4 DSA attention，bs6

這一層原本只記了類別合計。allreduce 和 mHC 沒有分到 attention 或 MoE，所以放在最後一行。

| 類別 | MI355X kernel（µs） | 合計 | B200 | 差 | × 11（ms） |
|---|---|---|---|---|---|
| MLA 投影 + quant + KV 寫入 | q_a/kv_a 4.17 + 10.26、qk_rmsnorm 4.39、q_b 4.12 + 9.69、W_UK 7.06、KV 4.51、W_UV 12.41、o_proj 4.22 + 10.26 | 71.1 | 37.6 | **+33.5** | 0.37 |
| k-pool topk | `kpool_topk_transform` 34.08 | 34.1 | 16.2 | **+17.9** | 0.20 |
| indexer 投影等 | fp32 Cijk 11.35、LN 4.56、wq_b 7.70、Hadamard 4.36、kpool 寫入 6.82、copy 4.32、hadamard_quant 4.70 | 43.8 | 28.6 | **+15.2** | 0.17 |
| indexer logits | 27.60 | 27.6 | 17.7 | +9.9 | 0.11 |
| sparse MLA | split_k 24.15 + reduce 4.46 | 28.6 | 26.4 | +2.3 | 0.03 |
| **attention 合計（不含 allreduce、mHC）** | | **205.2** | **126.5** | **+78.7** | **0.87** |
| allreduce × 2 + mHC × 4（attention 和 MoE 一起） | allreduce 9.19 + 10.18 | 48.6 | 36.7 | +11.9 | 0.13 |

- bs6 的 DSA 層，最大的是 **MLA 投影的 quant 和 W_UK / W_UV**：
  - q_a/kv_a、q_b、o_proj 前面各有一個 4.2 µs 的 `dynamic_per_group_scaled_quant`，B200 是 BF16 GEMM，不用 quant。
  - W_UK + W_UV 是 19.5 對 6.7 µs。
- sparse MLA 兩邊差不多。

### 4.5 MoE 段（KDA 層和 DSA 層共用）

逐 kernel 用 KDA 層的數字。編號接在 attention 後面；DSA 層同樣的 kernel 是 26–37 號。s4855 是 B200 的 side stream。

| # | 功能 | MI355X kernel | bs16 | bs6 | B200 kernel | bs16 | bs6 |
|---|---|---|---|---|---|---|---|
| 12–13 | mHC pre（FFN） | gemm_sqrsum + big_fuse_rmsnorm | 4.54 + 5.87 | 4.54 + 5.60 | prenorm + big_fuse | 3.80 + 5.97 | 3.80 + 6.05 |
| 14 | router GEMM（N=288） | Cijk MT16x16x1024 / MT16x16x256 | 7.97 | 7.73 | nvjet splitK + splitKreduce | 5.57 + 0.62 | 4.89（reduce 被蓋掉） |
| 15 | bf16→fp32 | copy | 4.37 | 4.22 | — | — | — |
| 16 | grouped topk | `grouped_topk` | 5.25 | 5.41 | routing BlockScores + Cluster | 1.10 + 4.67 | 3.51 + 4.98 |
| 17 | shared expert | `_fused_append_shared_experts`（併成第 289 個 expert） | 4.20 | 4.46 | s4855：TgvGemm + silu + nvjet | 1.52 + 0.62 + 3.39（13.4） | 5.69 |
| 18 | sort + quant | bs16：opus sort P0 + P23；bs6：fused `sort_quant_kernel_impl<289,9,16>` | 4.22 + 4.49 | 4.40 | nvfp4 quant | 0.81 | 0.17 |
| 19 | MoE GEMM1 | `gemm1_a4w4…bm16_iq`（bs6 是 `_hpf`） | **75.21** | 46.92 | `bmm_E2m1` | 60.57 | 44.52 |
| 20 | MoE GEMM2 | `gemm2_a4w4…bm16` | 38.87 | 24.29 | bmm + finalize | 33.22 + 6.16 | 23.55 + 4.15 |
| 21 | allreduce | `cross_device_reduce_2stage` | **14.79** | 9.26 | `all_reduce_2shot_push` / `1shot` | 7.01 | 7.21 |
| 22 | mHC post | `mhc_post` | 4.71 | 4.52 | `mhc_post_tilelang` | 2.75 | 1.77 |
| | **合計** | | **174.5** | **121.4** | | **137.8** | **110.3** |

依類別（µs / 層）：

| 類別 | bs16 MI355X | bs16 B200 | bs16 差 | bs6 MI355X | bs6 B200 | bs6 差 |
|---|---|---|---|---|---|---|
| MoE GEMM | 114.1 | 100.0 | **+14.1** | 71.2 | 72.2 | −1.0 |
| router 到 sort + quant（B200 含 shared expert） | 30.5 | 18.3 | **+12.2** | 26.2 | 19.2 | +7.0 |
| allreduce | 14.8 | 7.0 | **+7.8** | 9.3 | 7.2 | +2.1 |
| mHC pre + post | 15.1 | 12.5 | +2.6 | 14.7 | 11.6 | +3.0 |
| **合計（KDA 層）** | 174.5 | 137.8 | **+36.7** | 121.4 | 110.3 | **+11.1** |
| × 31 層（ms） | | | 1.14 | | | 0.34 |

DSA 層的 MoE 段是同一組 kernel：

| | bs16 MI355X | bs16 B200 | bs16 差 | bs6 MI355X | bs6 B200 | bs6 差 |
|---|---|---|---|---|---|---|
| MoE GEMM | 115.0（GEMM1 76.2、GEMM2 38.8） | 93.8（GEMM1 55.9、GEMM2 31.7） | +21.2 | 71.3 | 66.4 | +4.9 |
| router 到 sort | 31.2 | 18.3 | +12.9 | 26.7 | 21.5 | +5.2 |
| allreduce + mHC | 30.3 | 19.3 | +11.0 | 沒有分段（見 4.4） | | |
| **合計** | 176.4 | 131.4 | **+45.0** | 98.0（不含 allreduce、mHC） | 87.9 | +10.1 |
| × 11 層（ms） | | | 0.50 | | | 0.11 |

- **bs6 的 MoE GEMM 兩邊打平。** bs6 的 MoE 段只輸 11 µs，大多是 router 到 sort 的小 kernel。
- **bs16 的 MoE GEMM1 比 B200 多 15–20 µs。** aiter #6435 的 tuned CSV（inter_dim 512）token 數只有 1、2、4、…、64、128，沒有 96。
- **bs16 的 sort 沒有走 fused `sort_quant`**（bs6 有），變成 opus P0 + P23 兩個 kernel，多了一個 kernel。
- bs16 的 allreduce 是 B200 的 2 倍（14.8 對 7.0 µs），attention 段的 allreduce 也一樣。

### 4.6 KDA+dense 層（layer 0–2）

bs16 是 165.4 對 99.0 µs。前半段跟 KDA attention 一樣（in-proj 24.4 對 14.0、delta 24.2 對 15.4、allreduce 13.1 / 12.3 對 6.6 / 6.1）；FFN 是 FP8 quant + 2 個 CK blockscale GEMM（15.1 + 9.0）加 `clamp_silu_mul`，B200 是 NVFP4 dense（7.1 + 5.6）。只有 3 層，合計 0.20 ms。

## 5. 整個 cycle 依類別彙總（MI355X − B200，ms / cycle）

| 類別 | bs6 | bs16 | 相關的層 |
|---|---|---|---|
| KDA core（compare、conv、delta rule、o_norm） | **0.44** | **0.64** | 31 + 3 層 KDA |
| MoE GEMM | −0.02 | **0.67** | 42 層 |
| allreduce | 0.22 | **0.64** | 每層 2 次 |
| router 到 sort + quant | 0.28 | **0.52** | 42 層 |
| KDA 投影（in、f_b/g_b、o） | 0.29 | **0.50** | 34 層 |
| MLA 投影 + quant + W_UK / W_UV + KV 寫入 | **0.37** | 0.36 | 11 層 |
| indexer logits | 0.11 | 0.31 | 11 層 |
| k-pool topk | 0.20 | 0.28 | 11 層 |
| mHC | 0.24 | 0.20 | 每層 4 個 kernel |
| indexer 投影等 | 0.17 | 0.12 | 11 層 |
| sparse MLA | 0.03 | 0.08 | 11 層 |
| KDA+dense 層 | 0.12 | 0.20 | 3 層 |
| **合計** | **2.49** | **4.52** | 第 3 節實測 45 層差距：2.51 / 4.37 |

類別合計用的是序列相同的代表層，所以 bs16 比實測多 0.15 ms。

## 6. B200 為什麼每層比較快

- **大 kernel 沒有輸很多。** bs6 的 MoE GEMM 打平，KDA delta rule 在 ReplaySSM 關的時候 MI355X 反而比較快（bs16 31.4 對 27.1 µs）。
- **小 kernel 的固定成本差很多。** MI355X 在 HIP graph 裡，最小的 kernel 也要 4.2–4.5 µs。B200 同樣的 kernel 是 1.2–2 µs，再加上 PDL，很多 kernel 實際只佔 0.2–1 µs（`splitKreduce` 0.6、nvfp4 quant 0.8、`mhc_post` 2.9、`layer_norm_gated` 2.8）。
- **數量**：MI355X 一次 target verify 約 1,100 多個 kernel（每層 KDA+MoE 22–23 個、DSA+MoE 36–37 個）。B200 約 1,390 個（25 × 31 + 51 × 11 + 19 × 3），比較多，但每個便宜很多。
- **side stream**：B200 的 shared expert、indexer 的兩個 GEMM 和 k-pool 寫入跑在 side stream，每層 KDA+MoE 藏掉 13 µs、DSA+MoE 藏掉 35–36 µs。MI355X 把 shared expert 併成第 289 個 expert，不用另外算，只多一個 4.2 µs 的 append kernel，這部分其實沒輸。DSA 層的 side stream 部分就要靠 fusion 追回。

- **PDL 是什麼**：CUDA 的 Programmatic Dependent Launch（Hopper 以後）。
  - 它是同一條 stream，kernel 順序和資料相依都不變，不是 multi stream。
  - 前一個 kernel 呼叫 `griddepcontrol.launch_dependents`，下一個 kernel 就可以先被派到空出來的 SM 上，做不依賴前一個輸出的事，例如 launch、設定 shared memory、GEMM 先載 weight。真正要讀前一個 kernel 的輸出之前，再呼叫 `griddepcontrol.wait` 等它寫完。
  - 省下的是 launch 延遲和最後一波 CTA 的尾巴，所以 B200 trace 裡每個 kernel 的 duration 會包含等待的時間。
- **MI355X 不能用**：
  - 這次 image 的 ROCm 10.0，`hip_runtime_api.h` 只有從 CUDA 搬來的 graph edge enum（`hipGraphDependencyTypeProgrammatic`）。
  - 沒有 device 端的 trigger / wait 函式，也沒有 stream 的 launch attribute。
  - MI355X 每個 kernel 邊界約 4.3 µs，推測來自 CP dispatch 加上 kernel 之間的 barrier 和 cache release / acquire（8 個 XCD 的 L2 不共用），這一點沒有量過。
  - 在 MI355X 上要消掉 kernel 邊界，方法是 fusion，或 persistent kernel（monokernel）。這次 conc16 / 32 的設定 monokernel 是關的。

### 6.1 哪些小 kernel 可以合成一個（Triton 或 aiter）

- **條件**：
  - 合併後的 kernel 必須能在一個 CTA 裡拿到需要的整塊資料，也就是「每個 token、每個 head 各算各的」。需要整個 grid 都做完才能做下一步的，就不能合，例如 GEMM 的輸出還沒算完整個 row 就要做 topk。
  - 合成一個 Triton kernel 本身也有約 4.3 µs 的邊界成本，所以 N 個合成 1 個，最多省 (N − 1) × 4.3 µs，再扣掉合併後多做的工作。

| 每層的 kernel 群（bs16，µs） | 能不能合 | 做法 |
|---|---|---|
| KDA：`compare_scalar` 4.4 + `causal_conv1d_update` 5.0 + delta rule 26.1 + `layer_norm_gated` 4.3 | 可以，3 個併進 delta rule | conv1d 是每個 channel 自己做，可以放進 delta kernel 開頭（B200 的 `kda_decode_mtp` 就是這樣）；gated norm 是每個 head 的 128 維，delta kernel 結束時整個 head 都在 register 裡，可以直接做 |
| router GEMM 8.0 + bf16→fp32 4.4 + `grouped_topk` 5.3 + append shared 4.2 + sort P0 / P23 8.7 | 可以，6 個變 1–2 個 | N=288 很小，一個 CTA 可以算完一個 token 的 288 個 logit 再做 topk；sort 要所有 token 的 expert 計數，M ≤ 120 時一個 CTA 做得完（1110 計畫 4.1 / 4.2） |
| `mhc_post` + 下一個 `mhc_pre_gemm_sqrsum`（每層 2 個邊界） | 可以，aiter 已經有 | `mhc_fused_post_pre`，只要放寬 sglang 的 16 token 門檻（第 8 節第 10 項） |
| DSA：`dynamic_per_group_scaled_quant` × 3（每個約 4.2） | 可以 | 併進前一個 kernel 的 epilogue：mHC pre、`fused_qk_rmsnorm`、sparse MLA reduce |
| indexer：LayerNorm 4.6 + Hadamard 4.4 + copy 4.3 + hadamard_quant 4.7 + kpool 寫入 6.7 | 大部分可以 | 都是每個 token 自己做（128 維 Hadamard），可以合成 1–2 個 |
| CK / hipBLASLt 的 GEMM | 不行直接合 | 要先換成 Triton 或 FlyDSL GEMM，才能把前後的小 op 放進 prologue / epilogue |

- **粗估**：
  - KDA+MoE 層每層約可拿掉 7 個 kernel：KDA 3 個、router 到 sort 4 個，約 30 µs。mHC 合併實測只省 1–2 µs。合計最多約 32 µs × 31 層 ≈ 1.0 ms。
  - DSA+MoE 層再拿掉約 8 個，約 0.35 ms。
  - 合併後多做的工作還沒算，實際可能只有一半到七成。
- **風險**：每個合併都要改 sglang 的呼叫點，數值不一定 bitwise 相同，要跑 GSM8K。

所以不用 multi stream 的話，要追的是：**(a) 把幾個連續的小 kernel 合成一個（每拿掉一個約 4.3 µs），(b) 把明顯沒調好的 GEMM 和 KDA kernel 拉到 B200 的水準。**

## 7. ReplaySSM 在兩邊的效果

| | MI355X bs6 | B200 bs6 | MI355X bs16 | B200 bs16 |
|---|---|---|---|---|
| KDA delta kernel，關 → 開（µs） | 16.85 → 14.76 | 16.59 → 9.47 | 31.41 → 26.11 | 27.10 → 14.54 |
| KDA+MoE 層 wall，關 → 開（µs） | 207.7 → 201.1 | 172.3 → 158.3 | 295.0 → 280.9 | 223.5 → 198.3 |
| 45 層，關 → 開（ms） | 10.79 → 10.51 | 8.48 → 8.00 | 15.11 → 14.46 | 10.93 → 10.09 |

- bs6 的「關」用的是 11/08 的 trace；bs16 的開和關是今天在同一台機器上跑的。
- ReplaySSM 把 B200 的 KDA kernel 時間砍掉 43–46%，MI355X 只砍掉 12–17%。
- ReplaySSM 在 tail 多了 `kda_replayssm_exact_fold_kernel`（accept 之後把 state 補上去），每次呼叫的時間如下。commit 這邊 MI355X 反而比較快，所以要追的是 verify 裡的 KDA kernel，不是 fold。

  | | bs6 | bs16 |
  |---|---|---|
  | MI355X（µs / 次） | 80 | 214 |
  | B200（µs / 次） | 126 | 293 |
- 所以 MI355X 的 Triton KDA kernel 在 ReplaySSM 模式下還有很大空間。開了 ReplaySSM 之後，verify 不必寫每個 draft 位置的 fp32 state；如果 kernel 還在做多餘的 state 讀寫，或 tile 沒有針對「只讀一次 state」調整，就會是現在這樣。要讀 Triton kernel 確認。

## 8. 建議（不用 multi stream，依 bs16 能省的時間排序，括號是 bs6）

| 順位 | 項目 | 現況（MI355X / B200，µs） | 預估每 cycle 省 | 備註 |
|---|---|---|---|---|
| 1 | **ReplaySSM 模式的 KDA delta kernel**：確認 verify 不再寫 intermediate state，再調 tile | bs16 26.1 / 14.5，bs6 14.8 / 9.4 | 0.3–0.4 ms（0.15–0.18） | 第 7 節。B200 的 conv 也併在同一個 kernel |
| 2 | **KDA core fusion**：`compare_scalar` + `causal_conv1d_update` 併進 delta kernel；`layer_norm_gated` 併進 delta kernel 的 epilogue 或 o_proj 的 prologue | 3 個 kernel 約 13.5 µs | 0.3–0.45 ms（同） | 每層省 2–3 個 kernel × 34 層 |
| 3 | **bs16 的 KDA in-proj**：M=96 不要走 hipBLASLt Cijk，改走 bs6 用的 FlyDSL hgemm，或補 tuned CSV | 23.8 / 13.0（bs6 FlyDSL 14.9） | 約 0.3 ms（—） | 每層 8–9 µs × 34 層 |
| 4 | **MoE router 到 sort 合併**：router GEMM、bf16→fp32、grouped_topk、append shared 合成一個 kernel（見 1110 計畫 4.1）；bs16 也走 fused `sort_quant` | 6 個 kernel 30.5 / 18.3 | 0.4–0.55 ms（0.25） | 42 層 |
| 5 | **MoE GEMM1 補 M=96 的 tune** | 75.2 / 60.6 | 最多 0.6 ms（—） | CSV 只有 64 和 128；bs6 已經打平 |
| 6 | **DSA 的 MLA quant 併進前一個 kernel**：q_a/kv_a 前用 mHC 的 FP8 輸出或 `fused_rms_fp8_group_quant`，q_b 前併進 `fused_qk_rmsnorm`，o_proj 前併進 sparse MLA reduce；**W_UK / W_UV 換更好的 bmm** | 3 個 quant 12.7；W_UK + W_UV 22.1 / 7.3 | 約 0.3 ms（0.3） | 11 層 |
| 7 | **allreduce**：bs16（786 KB）custom 2-stage 13.6–14.8 µs，B200 2-shot 6–7 µs；microbench 是 11.9 µs | 28.4 / 13.0 每層 | 最多 0.6 ms | 已排除 QR / FP8 AR（handoff 第 5 節）；要做低延遲 2-stage 或跟 mHC post 合併 |
| 8 | **indexer logits `next_n=6` layout**（1110 計畫已實作，env gate） | 65.7 / 37.8 | 約 0.15–0.25 ms | 1110 實測 bs16 −0.8% |
| 9 | **k-pool topk** | 40.0 / 15.0 | 最多 0.27 ms | 1110 實測 v2-A register topk 在這個 stack 看不出收益，要重新查原因 |
| 10 | **mHC post + 下一個 pre 合併**：把 sglang 的 `_MHC_FUSED_BOUNDARY_MAX_TOKENS=16` 放到約 192，讓 aiter `mhc_fused_post_pre` 生效（1110 計畫第 3 節） | M=36：12.6 → 10.6；M=96：14.1 → 13.2 | 約 0.08 ms（0.18） | 每個 cycle 約 89 個層間邊界；只改一個門檻，對 bs6 最划算 |

- 第 1–3 項都在 KDA 層，彼此獨立，加起來 bs16 約 0.9–1.1 ms（5–6%），bs6 約 0.5–0.6 ms（4%）。**這三項做完，P90 Int 有機會在 conc16 和 conc32 都到 +5%。**
- 第 4、6、10 項主要是減少 kernel 數，bs6 也有效。第 10 項最便宜，可以先做。
- 第 5、7 項只在 bs16 有效。

## 9. 檔案

| 內容 | 路徑 |
|---|---|
| MI355X trace（ReplaySSM 開：bs6、bs16；關：bs16） | `/shared_nfs/jacchang/prof_1111/mi/{repon,repoff}/` |
| B200 trace（同上） | `/shared_nfs/jacchang/prof_1111/b200/{repon,repoff}/`，原檔在 dgx-029 `/mnt/home/jacchang/prof_1111/b200/` |
| 每層 wall 和逐 kernel 表（工具輸出） | `/shared_nfs/jacchang/prof_1111/ana/*.seq.txt` |
| 11/08 的 trace（ReplaySSM 關，bs6 對照用） | `/shared_nfs/jacchang/prof_1108/{mi,b200}/` |
| profile 腳本 | `/shared_nfs/jacchang/prof_1111/prof_rep.sh`（MI355X）、`tools/b200_prof_rep_1111.sh`（B200） |
| 新工具（未 commit） | `tools/decode_layer_wall.py`：`decode_layer_wall.py [--seq] LABEL=trace.json.gz ...` |

## 10. 11/11 下午追問的回答

### 10.1 KDA in-proj 與 tuned CSV

- **in-proj 是什麼**：KDA 層的第一個 projection（input projection）。mHC pre 輸出的 hidden（4096）一次投影成 q、k、v（每個 rank 各 2048）、beta（16）、f_a（128）、g_a（128），合計 N=6416、K=4096 的 BF16 GEMM（`fused_qkvbfg_a_proj`）。
- **CSV 有涵蓋，但 M=96 選錯了**：aiter #5982 的 `glm53_flash_bf16_tuned_gemm.csv` 有 M=1–248。image（285579b62）和 aiter upstream main（11/11 `6efb333ad`）這幾列完全一樣，也沒有其他 open PR 在改。
- M=96、112、120、128 當時選的是 torch（hipBLASLt），比旁邊 M=104 的 FlyDSL 還慢。11/11 在 node 152 用 `csrc/gemm_a16w16/gemm_a16w16_tune.py` 重 tune，每個 M 都是 FlyDSL splitK 4 勝：

| M（verify row） | 16 | 32 | 48 | 64 | 72 | 88 | **96** | 112 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| 現在 CSV（µs） | 12.62 | 14.07 | 15.45 | 16.99 | 18.53 | 19.08 | **24.28**（torch） | 21.63（torch） | 21.69（torch） |
| 重 tune（µs） | 12.07 | 12.46 | 13.20 | 13.91 | 15.76 | 16.54 | **16.74** | 17.68 | 18.32 |

- o_proj（N=4096、K=2048）在 GLM-5.3-Flash 自己的 CSV 裡沒有 M=8–128 的 row。GLM-5 的 `glm5_bf16_tuned_gemm.csv` 有同一個 shape 的 M=24、32、48、60、64、84、128，是 6.8–9.7 µs。bs6 trace 走 Triton（8.8 µs），原因不是缺 row，而是 sglang 在 K ≤ 2048、M ≤ 64 時一律走 Triton（11.1 的 B 修掉了）。重 tune 後 M=18–66 是 6.2–7.7 µs，M=96 是 8.6 µs，跟 bs16 現在一樣。
  - 兩個 CSV 有同一個 key 時，aiter 會報錯，並把較慢的 row 從 image 的檔案刪掉。opt2 跑的時候它刪了 `glm5_bf16_tuned_gemm.csv` 的 7 個 row。原檔備份在 `/shared_nfs/jacchang/csv_backup/glm5_bf16_tuned_gemm.csv.image285579b62`，lane 跑完會用 git checkout 還原。之後的 run 改用 `/shared_nfs/jacchang/tune1111/glm5_bf16_dedup.csv`，不再動 image 的檔案。
- verify row 都是 6 的倍數（36、54、66…），CSV 只有 8 的倍數，查不到時會往上補。M=18–114 的 6 的倍數正在 tune（`/shared_nfs/jacchang/tune1111/round3_*`）。
- 結論：可以自己開 aiter PR 更新這兩個 shape。

### 10.2 in-proj 能不能跟 f_b / g_b 合併

不行。f_b / g_b（`fused_fg_b_proj`，(2, 128 → 2048) 的 batched GEMM）的輸入是 in-proj 輸出的 f_a / g_a。low-rank 的第二段要等第一段整個算完。要合只能把兩段 weight 先乘起來，每層 weight 從約 3 MB 變 34 MB，不划算。f_b / g_b 本身可以補 small-M 的 tuned row：aiter #7049（已關閉）在 16 × 2048 × 128 從 5.0 降到 2.5 µs。

### 10.3 MoE sort：P23 是什麼，為什麼 bs16 沒走 `sort_quant`

- Opus sort 分兩個 kernel：P0 數每個 expert 的 token 數，P23 是 phase 2+3（prefix sum 加 scatter）。
- block_size 16 時，aiter 的 `_aux_uses_opus` 規則是 `3 × routed_rows ≥ 4 × num_experts` 就改走 Opus。這條是在 kimi-k3（896 experts、topk 16）上定的。
  - GLM-5.3-Flash（289 experts、topk 9）在 rows ≥ 43 就切過去。bs6 是 36 row，所以走 fused；bs16 是 96 row，所以走 Opus。
- microbench（HIP graph，`tools/bench_moe_sort_1111.py`）兩種 sort 的 routing 結果完全相同。fused 在 GLM 上一直快到 rows 約 160：

| rows | 36 | 54 | 96 | 120 | 144 | 168 |
|---|---|---|---|---|---|---|
| fused（µs） | 4.32 | 5.04 | 5.68 | 6.11 | 6.43 | 6.73 |
| Opus（µs） | 6.16 | 6.22 | 6.42 | 6.59 | 6.58 | 6.66 |

- 其他 6 種 shape 的 crossover（token 數）：896/16 約 50、513/11 約 78、385/9 約 108、256/8 約 160、257/9 約 225、129/5 約 850，跟 `routed_rows ∝ num_experts` 對不上。
- 新規則是兩個條件都成立才走 Opus：`tokens × E ≥ 40000` 且 `routed × E ≥ 400000`。在 7 種 shape 上離較快的 sort 都不超過 0.4 µs；舊規則在 289/9 最多慢 1.2 µs，在 129/5 慢 2.2 µs。
  - 改動在 `~/PR/wt-aiter-sort-crossover`（branch `jacob/moe-bm16-sort-crossover`，以 upstream main 為 base，未 commit）。

### 10.4 MoE GEMM 的 M=96

不需要 tune 96，tune 了也用不到。aiter fused MoE 用 `get_padded_M` 把 token 數補到下一個 2 的冪（32768 以下），所以 96 一定查 128 那一列。要讓 96 有自己的設定，得先改 aiter 的 tier，那就要先證明 128 的設定在 96 不好，目前沒有證據。

### 10.5 `mhc_pre_gemm_sqrsum` 和 `mhc_pre_big_fuse_rmsnorm` 能不能合

不能直接合，因為中間有一次整個 K 的加總：

- 第一個 kernel 是 [M, 16384] × [16384, 24] 的 GEMM 加平方和。M 很小，所以靠 split-K 把 16384 拆給很多 CTA。
- 第二個 kernel 要每個 token 的完整加總，才能做 sinkhorn、把 4 條 stream 加權相加、再做 RMSNorm。
- 合成一個 kernel 只有兩種做法：
  - 最後一個做完的 CTA 負責收尾：收尾的工作量是每個 token 讀 4 × 4096 個值，只剩幾個 CTA 在做，會更慢。
  - grid barrier：就是 monokernel 的作法，代價見 10.10。
- 可行的是跟別的 kernel 合：
  - 上一層的 `mhc_post` 是逐元素運算，可以併進下一個 `mhc_pre_gemm_sqrsum` 的 A 讀取（aiter `mhc_fused_post_pre`，3 個 kernel 變 2 個），目前被 sglang 的 16 token 門檻擋住。1110 實測 M=36 是 12.6 → 10.6 µs，但 sglang 的註解寫「24 token 打平」，兩者不一致，要在 decode 裡實測。
  - big_fuse 的輸出直接接下一個 GEMM，也可以併進那個 GEMM 的 A 讀取，但要改 FlyDSL / hipBLASLt GEMM，工程很大。

### 10.6 DSA 的 quant + CK fp8 blockscale

- **找到一個 sglang 的回歸**：sglang #31105（8 月，Kimi per-channel FP8 修正）把 `forward_mla_rocm.py` 的條件從 `weight.dtype == fp8` 改成 `_is_block_scale_fp8(proj)`。
  - 這個函式只看 `weight_scale`。block FP8 走 `Fp8LinearMethod`，scale 是放在 `weight_scale_inv`。GLM-5.3-Flash（Quark checkpoint 的 MLA projection 也是這條）和 DeepSeek 的 block FP8 因此都被判成「不是 block FP8」。
  - 影響：q_b 前面本來是 `fused_rms_fp8_group_quant`（RMSNorm 加 quant 一個 kernel），變成 `fused_qk_rmsnorm` 加獨立的 `dynamic_per_group_scaled_quant`。o_proj 前面的 `fused_flatten_fp8_group_quant` 也一樣走不到。upstream main 還沒修。
  - 修法：先讀 fp32 的 `weight_scale_inv`，不是 fp32（MXFP8 的 e8m0）就不算；沒有再讀 `weight_scale`。單元測試加了兩個 case，11 個全部通過。
- **能不能跟後面的 `fused_qk_rmsnorm` 合**：RMSNorm 要整個 row（q 1536、kv 512）。GEMM 的 N 是拆給多個 CTA 算的，所以不能把 RMSNorm 併進 GEMM 的 epilogue。要省的是 quant，方向是併進前一個 kernel：
  - q_b 前面的 quant：上面的修正會把它併回 RMSNorm。
  - q_a / kv_a 前面的 quant：輸入是 `mhc_pre_big_fuse_rmsnorm` 的輸出，要讓 mHC pre 直接輸出 FP8。1110 確認過 sglang 和 aiter 都還沒有這條路。
  - 另一條路是學 B200 用 BF16 GEMM（TgvGemm 6.85 µs），但 checkpoint 是 FP8 weight，要在 load 時 dequant，數值會變。

### 10.7 indexer logits + k-pool topk（MI355X 約 105 µs，B200 約 53 µs）

- **logits（65.7 對 37.8 µs）**：sglang 把 verify 的 q 攤平成 6 × bs 個 row，用 `next_n=1` 呼叫 aiter，同一個 request 的 6 個 row 各讀一次整段 k-pool。B200 的 deep_gemm 也是 NextN=1，但它的 L2 和 kernel 比較能吃下重複讀取。
  - 1110 已經實作 `next_n=6` 的 layout（同一個 kernel、logits bitwise 相同），bs16 microbench 從 72 降到 49 µs。這次把它移到 `~/PR/wt-glm53-1111-opt`，拿掉 env，在 HIP 的 target verify 直接啟用。
  - 再往下要改 aiter gluon kernel，讓 K tile 只載一次給 6 個 row 用（1110 計畫 6.1 步驟 2）。
- **topk（40.0 對 15.0 µs）**：B200 用 `topk_persistent_cluster` 加 `topk_main`，靠 thread block cluster 共用 shared memory 做多 CTA 的 radix select。MI355X 沒有 cluster，`kpool_topk_transform<512>` 是每個 row 一個 CTA，自己掃過約 3.2 萬個 pooled column。1110 的 v2-A register topk 在 stack 上看不出收益，還沒找到原因。

### 10.8 KDA Triton kernel 在 ReplaySSM 模式（確認結果）

- ReplaySSM 開的時候，`CACHE_INTERMEDIATE_STATES` 是 False，verify 確實沒有寫 intermediate state，第 7 節的懷疑不成立。
- 現在的 launch 設定已經是最好的：sweep BV 4–128 × num_warps 1 / 2 / 4，bs6 現在 13.46 µs、最好 12.92；bs16 現在 21.12、最好 20.52。
- kernel 的瓶頸是時間軸上的序列：每個 CTA 只有 1 個 warp，T=6 步依序做，每一步都要等 q、k、v、g、beta 的 load 回來。B200 的 CUTLASS kernel 把 conv 和整段 T 的 load 排在一起做。
- `fused_kda_conv_gating_verify`（conv1d 併進 recurrence，sglang 已有）在 ReplaySSM ring 下輸出和 ring 都 bitwise 相同，但只在部分 bs 有收益：

| bs | 3 | 6 | 9 | 16 | 20 |
|---|---|---|---|---|---|
| 現在（conv + delta，µs） | 14.04 | 18.13 | 23.01 | 26.86 | 34.69 |
| fused（µs） | 14.58 | 16.12 | 23.65 | 26.52（BV 32：25.45） | 61.67（超出 BV 調參範圍） |

  每 cycle 只省 0.03–0.07 ms，所以這次沒有開。要追上 B200，得重寫 kernel：一次把整段 T 的 q / k / v / g 載進來，再做 recurrence。

### 10.9 「CTA 拿到整塊資料」是什麼意思

- CTA（AMD 叫 workgroup）是同一個 CU 上的一群 thread。同一個 CTA 裡可以用 LDS 交換資料、用 barrier 同步；**不同 CTA 之間在同一個 kernel 裡沒辦法互相等**，除非用 global memory 的 atomic 自己做 grid barrier。
- 所以把 A、B 兩個 kernel 合成一個時，每個 CTA 做 B 的那一份工作，只能用它自己算出來的 A 結果（或自己重算）。
- 可以合的例子：conv1d → delta rule。delta rule 處理 head h 的 CTA，只需要 head h 那幾個 channel 的 conv 結果，自己算就好。
- 不能合的例子：
  - mHC pre：平方和是很多 CTA 各算一段 K 再加起來，下一步要完整的和。
  - GEMM → RMSNorm：一個 row 的輸出被拆給多個 CTA 算，RMSNorm 要整個 row。

### 10.10 「半 monokernel」的看法

先量兩件事（GPU 7 單獨跑，HIP graph，40 步平均）：

| 每一步的成本（µs） | 32 CTA | 64 CTA | 128 CTA | 256 CTA |
|---|---|---|---|---|
| kernel 邊界（graph 裡連續的空 kernel） | 1.84 | 1.85 | 1.85 | 1.86 |
| persistent kernel 裡的 grid barrier（atomic 加 spin） | 1.88 | 3.23 | 4.30 | 7.65 |

| 同一串 kernel（µs / 個） | 沒開 profiler | torch profiler 量到的 duration |
|---|---|---|
| 空 kernel（32 或 256 CTA） | 1.85 | 4.22 |
| 96 × 4096 bf16 copy（bs16 的一份 hidden） | 1.92 | 4.25 |
| 4096 × 4096 copy | 11.37 | 11.34 |
| 16384 × 4096 copy | 39.08 | 38.97 |

- **profiler 會把短 kernel 撐到約 4.2 µs**。第 6 節說「MI355X 最小的 kernel 要 4.2–4.5 µs」、「每個 kernel 邊界約 4.3 µs」，這是 profiler 造成的；沒開 profiler 時，kernel 邊界約 1.85 µs。10 µs 以上的 kernel 不受影響。
  - 所以 trace 裡每拿掉一個 4.2 µs 的小 kernel，實際只省約 2 µs。大 kernel（in-proj、indexer logits）的改善則是準的。
  - 一個 cycle（verify 加 draft）約 1,400 個 kernel，其中 bs16 有 530 個、bs6 有 650 個落在 4.0–4.8 µs。所以 MI355X 的 trace cycle 可能比實際多算 0.6–1.5 ms。B200 的 profiler 有沒有同樣的問題還沒量。
- **grid barrier 不比 kernel 邊界便宜**：32 CTA 打平，CTA 越多越貴，因為所有 CTA 都在搶同一個 counter。分層的 barrier 可以再便宜一點，但最多也只是打平。
- **所以 monokernel 的收益不是來自省掉 kernel 邊界**，而是來自 phase 之間不必經過 global memory、可以預先載 weight，以及跨 phase 的 overlap。你之前 PR 71 / 97 / 98 / 100 在 conc8 + MTP 沒有收益，跟這個量測一致。
- **「半 monokernel」的建議**：不要用 grid barrier 把很多 phase 串起來。只合併「不需要 grid 同步」的相鄰 kernel，也就是 10.9 說的每個 CTA 自己算得完的那種。每合併一個，實際省約 2 µs，不會有 barrier 的成本。MoE GEMM、sparse MLA、indexer 這些大 kernel 照舊分開，你的想法是對的。
  - 照 6.1 的表，verify 裡 KDA+MoE 每層約可拿掉 7 個、DSA+MoE 每層約 8 個，合計約 300 個小 kernel，實際省約 0.6 ms（bs16 一個 cycle 約 17 ms 的 3–4%）。
  - 值得做的是 KDA 那串（conv1d、delta rule、gated norm）、router 到 sort 那串，以及 DSA indexer 的 LayerNorm、Hadamard、quant、k-pool 寫入那串。

## 11. 11/11 的優化結果

### 11.1 做了什麼

| # | 改動 | 在哪裡 | 對應第 8 節 |
|---|---|---|---|
| A | KDA in-proj 和 o_proj 重 tune（M=16–128，含 6 的倍數） | `/shared_nfs/jacchang/tune1111/r123/glm53_flash_bf16_tuned_gemm.csv`（265 row，比 image 多 55 row 或換掉較慢的 row） | 3 |
| B | sglang 的 BF16 GEMM 在 K ≤ 2048、M ≤ 64 時一律走 Triton，不管 tuned CSV 有沒有 row。改成有 tuned row 就走 tgemm | `unquant.py` `_prefer_triton_a16w16` | 3 |
| C | MoE sort 在 block 16 時改用新的 crossover 規則，GLM 到 154 row 都走 fused `sort_quant` | aiter `fused_moe.py` `_aux_uses_opus`（`~/PR/wt-aiter-sort-crossover`） | 4 |
| D | 修 `_is_block_scale_fp8` 的回歸，DSA 層的 q_b 前面變回 `fused_rms_fp8_group_quant` | `deepseek_common/utils.py` 加兩個單元測試 | 6 |
| E | KDA / GDN verify 時不再每層做一次 `compare_scalar`（只在 pp stable rows 開的時候才選 index） | `kda_backend.py`、`gdn_backend.py` | 2 |
| F | indexer logits 用 `next_n=6` layout，同一個 request 的 6 個 row 一起算 | `paged_mqa_logits.py`、`dsa_indexer_kpool.py`、`kpool_plan.py` | 8 |
| G | `_MHC_FUSED_BOUNDARY_MAX_TOKENS` 從 16 改成 192，attention 後的 `mhc_post` 和 FFN 前的 `mhc_pre_gemm_sqrsum` 合成一個 kernel | `glm5_next.py` | 10 |

sglang 改動都在 `~/PR/wt-glm53-1111-opt`（base `09111fe9a3`，未 commit），aiter 改動在 container 裡用 `tools/patch_aiter_sort_crossover.sh` 套上。

### 11.2 trace 上的結果（ISL 128K，MTP5 golden AL，ReplaySSM 開）

| ms | bs6 45 層 | bs6 cycle | bs16 45 層 | bs16 cycle |
|---|---|---|---|---|
| MI355X base | 10.51 | 13.34 | 14.46 | 18.14 |
| MI355X opt（A–G） | 9.93（−5.5%） | 12.77（−4.3%） | 13.58（−6.1%） | 17.15（−5.4%） |
| MI355X 最新（第 12 節 round 5，A–J） | 9.50（−9.6%） | 12.38（−7.2%） | 13.30（−8.0%） | 16.93（−6.7%） |
| B200 | 8.00 | 11.02 | 10.09 | 14.3 |
| opt 比 B200 多 | +1.93（+24%） | +1.75（+16%） | +3.49（+35%） | +2.85（+20%） |
| 最新比 B200 多 | +1.50（+19%） | +1.36（+12%） | +3.21（+32%） | +2.63（+18%） |
| base 到最新，跟 B200 的差距縮小了 | 40% | 41% | 27% | 32% |

- 45 層是 2 個 cycle 的平均。MI355X 的 cycle 是乾淨的那個（c1）。B200 的數字來自第 0 和第 3 節，bs16 的 cycle 已經扣掉 rank 不同步的等待（第 2 節）。
- 這是 trace 上的數字。依 10.10，MI355X 的 profiler 會把短 kernel 撐到約 4.2 µs，B200 的 profiler 有沒有同樣的問題還沒量，所以 MI355X 這邊的差距可能被高估。沒開 profiler 的實際差距要看 AgentX 的 chunk gap 和 P90 Int。
- bs16 的差距縮得比較少：bs16 剩下的大項是 KDA delta kernel、allreduce、MoE GEMM1、indexer 和 k-pool topk（第 4.1、4.2、4.5 節），這次只動到 indexer logits。

| 每層的 kernel（trace µs） | bs6 base → opt | B200 bs6 | bs16 base → opt | B200 bs16 | 來自 |
|---|---|---|---|---|---|
| KDA in-proj | 14.9 → 13.7 | 10.68 | 23.8 → 17.3 | 12.97 | A |
| KDA o_proj | 8.8（Triton）→ 6.3 | 5.98 | 8.6 → 8.3 | 6.56 | A、B |
| `compare_scalar` | 4.4 → 0 | 1.76 | 4.4 → 0 | 1.72 | E |
| MoE sort | 4.4 → 4.4（本來就是 fused） | 0.17 | 8.7（Opus 兩個 kernel）→ 5.4 | 0.81 | C |
| DSA q_b 前的 RMSNorm + quant | 8.5 → 4.4 | 沒拆出來 | 8.7 → 4.2 | 2.26 | D |
| indexer logits | 27.6 → 22.1 | 17.7 | 65.7 → 46.3 | 37.77 | F |
| attention 後的 `mhc_post` + FFN 前的 `mhc_pre_gemm_sqrsum` | 8.8 → 5.3 | 5.79 | 9.2 → 7.6 | 6.73 | G |

- B200 的數字是第 4 節的實際佔用（扣掉 PDL 重疊）。
- B200 沒有獨立的 sort kernel（sort 在 routing kernel 裡），表裡的 0.17 / 0.81 是 nvfp4 quant。
- B200 的 q_b 是 BF16 GEMM，不用 quant，2.26 是兩個 RMSNorm。bs6 的 DSA 層在 4.4 節只有類別合計，沒有逐 kernel 列出。
- B200 的 mHC 那一列是 `mhc_post` 加 `hc_prenorm_gemm`，兩個還是分開的 kernel。
- 依 10.10，trace 裡 4.2 µs 左右的小 kernel 實際只有約 2 µs，所以 C、D、E、G 的實際收益比 trace 少。A、F 是大 kernel，trace 準。
- topk（`sglang::` 那個 kernel）bs6 在 34–38 µs 之間跳，跟這些改動無關。

### 11.3 沒開 profiler 的結果和 GSM8K

ISL 1024 / OSL 1024，MTP5 golden AL（3.56），conc × 5 個 prompt。ISL 短所以沒有 prefill 卡住的問題，量到的是純 decode；但 indexer（F）在 1K context 幾乎不花時間，所以這裡看不到 F 的收益。

| | conc6 base | conc6 opt | conc16 base | conc16 opt |
|---|---|---|---|---|
| median TPOT（ms） | 3.61 | 3.49（−3.3%） | 5.08 | 4.95（−2.6%） |
| mean TPOT（ms） | 3.67 | 3.60 | 5.10 | 4.95 |
| 換算成 cycle（median TPOT × 3.56，ms） | 12.85 | 12.42（−0.43） | 18.09 | 17.62（−0.46） |
| output throughput（tok/s） | 1076 | 1088 | 2205 | 2262（+2.6%） |

- trace 上扣掉 F 之後，bs16 是 −0.67 ms，實際是 −0.46 ms，差的部分就是 10.10 說的小 kernel 被 profiler 撐大。
- 在 128K context，F 在 trace 上 bs16 再省 0.21 ms、bs6 再省 0.06 ms（大 kernel，算準的）。所以 128K 的實際收益估計是 bs16 約 −0.67 ms（−3.7%）、bs6 約 −0.49 ms（−3.8%）。這是估計，沒有實測。
- GSM8K（sgl-eval，1319 題，thinking，MTP5 真實 accept）：**96.66%**，原本是 95.83%，沒有掉。

### 11.4 第 8 節各項的狀態

| 順位 | 項目 | 狀態 |
|---|---|---|
| 1 | ReplaySSM 的 KDA delta kernel | 查完：沒有寫 intermediate state，tile 已經是最好的。剩下的差距是 kernel 本身的設計（10.8），要重寫 |
| 2 | KDA core fusion | `compare_scalar` 拿掉了（E）。conv1d 併進 delta rule 在 bs ≤ 8 開了（12.1 J）。gated norm 做不了（12.5） |
| 3 | KDA in-proj | 做了（A、B）。bs16 每層 −6.5 µs |
| 4 | MoE router 到 sort | sort 做了（C）。router 合併做了（12.1 H、I）：GEMM 到 topk 從 4 個 kernel 變 2 個 |
| 5 | MoE GEMM1 的 M=96 | 不用做：aiter 把 96 補到 128 查表（10.4） |
| 6 | DSA quant | q_b 做了（D）。q_a / kv_a 要 mHC 直接輸出 FP8，還沒做 |
| 7 | allreduce | 沒做 |
| 8 | indexer `next_n=6` | 做了（F）。bs16 每層 −19 µs |
| 9 | k-pool topk | 沒做 |
| 10 | mHC post + pre | 做了（G），只合到 attention 後那一個邊界 |

### 11.5 檔案和環境

| 內容 | 路徑 |
|---|---|
| opt 的 trace（最後一版是 opt3） | `/shared_nfs/jacchang/prof_1111/mi/{opt1b,opt2,opt2mhc,opt3}/` |
| 沒開 profiler 的 bench 和 GSM8K log | `/shared_nfs/jacchang/prof_1111/perf1111/{perfopt,perfbase,gsmopt}/` |
| tune 的輸入和輸出 | `/shared_nfs/jacchang/tune1111/` |
| 新工具（未 commit） | `tools/`：`bench_kda_verify_1111.py`、`bench_moe_sort_1111.py`、`bench_grid_barrier_1111.py`、`bench_kernel_floor_1111.py`、`merge_tuned_csv.py`、`patch_aiter_sort_crossover.sh`、`prof_opt1111.sh`、`bench_opt1111.sh`、`chain_bench1111.sh` |
| `decode_layer_wall.py` | 加了 `mhc_fused_post_pre_gemm_sqrsum` 當層的起點，G 開的時候也能切層 |

- container `jacchang_pr48875` 的 aiter 現在還套著 C 的 patch。還原用 `docker exec jacchang_pr48875 bash /home/jacchang/SGLang-benchmarks/tools/patch_aiter_sort_crossover.sh revert`。
- `glm5_bf16_tuned_gemm.csv` 已經還原成 image 的版本。

## 12. 11/11 晚上：router 合併和 KDA 小 kernel 合併

### 12.1 做了什麼

| # | 改動 | 在哪裡 |
|---|---|---|
| H | aiter 上單一 group 的 sigmoid routing（GLM-5.3-Flash 是 1 group）預設走 sglang 的 Triton router。原本是 fp32 cast、aiter `biased_grouped_topk`、`_fused_append_shared_experts` 三個 kernel，現在一個 kernel 做完，shared expert 那一欄也一起寫 | `topk.py` `_aiter_uses_jit_router` |
| H' | H 漏了 CUDA graph 的 padded row：改成 router 直接寫 aiter 要的 padded row（routed id 0、shared id 288、weight 0），後面兩個 mask kernel 就不用跑。`test_smallm_router_gfx950` 因此從 fail 變 pass | `moe_fused_gate.py` 加 `padded_routed_id`、`topk.py` |
| I | router GEMM 改用現成的 split-K GEMV（`router_gemv_hip`，本來只給 DeepSeek-V4.1），Triton router 直接加總 fp32 partials。logits 不再先轉 bf16，所以選的 expert 和 fp32 reference 一樣 | `deepseek_v2_hip_moe.py` `router_max_tokens`、`router_gemv_hip.py`（上限 96 row）、`topk.py` |
| J | gfx950 + ReplaySSM 的 KDA verify 改用現成的 conv1d + delta rule 合併 kernel（`fused_kda_conv_gating_verify`，原本 HIP 只在 batch 1 開），V tile 依 batch 選 8 / 16 / 32 | `kda_backend.py`、`fused_kda_conv_recurrent_verify.py` |

H 在 round 4（`wt-glm53-1111-opt`），H' + I + J 在 round 5（`wt-glm53-1111-kda`）。

### 12.2 microbench（GPU 7，HIP graph，沒開 profiler）

| | 36 row（bs6） | 96 row（bs16） |
|---|---|---|
| router 後段：aiter 三個 kernel → Triton router（H） | 9.2 → 4.7 µs | 9.5 → 4.7 µs |
| router GEMM + router：hipBLASLt bf16 → split-K GEMV + partials（I） | 11.1 → 9.0 µs | 11.3 → 10.6 µs |
| 跟 fp32 reference 選的 expert 一樣的 row（I 之前 → 之後） | 36 → 36 | 93 → 96 |
| KDA conv1d + delta rule → 合併 kernel（J） | 18.2 → 16.4 µs | 27.0 → 25.3 µs |

- J 在 bs 1–16 都比較快，輸出、intermediate window、ReplaySSM 的 4 個 ring 都和原本 bit 一樣。
- 單元測試：router 相關 78 pass。KDA 的 `test_kda_fused_verify_backend` 有 10 個 fail，沒改之前也是同樣 10 個（測試用 4 個 draft token，HIP 的條件只收 6 或 8）。

### 12.3 trace 上的結果（ISL 128K，MTP5 golden AL，ReplaySSM 開）

| | bs6 opt3 | bs6 round 4 | bs6 round 5 | bs16 opt3 | bs16 round 4 | bs16 round 5 |
|---|---|---|---|---|---|---|
| 45 層（ms，2 個 cycle 平均） | 9.93 | 9.72 | 9.50 | 13.58 | 13.31 | 13.30 |
| 乾淨的 cycle（c1，ms） | 12.77 | 12.61 | 12.38 | 17.15 | 16.90 | 16.93 |

| 每層的 kernel（trace µs） | bs6 round 4 → 5 | B200 bs6 | bs16 round 4 → 5 | B200 bs16 |
|---|---|---|---|---|
| router GEMM（hipBLASLt → split-K GEMV） | 7.9 → 4.6 | 4.89 | 8.0 → 6.5 | 6.19 |
| Triton router | 5.9 → 6.5 | 8.49 | 5.9 → 6.1 | 5.77 |
| conv1d + delta rule → 合併 kernel | 4.4 + 15.5 → 19.1 | 9.42 | 5.0 + 26.1 → 32.5 | 14.52 |

- B200 的 router GEMM 是 nvjet splitK 加 splitKreduce。Triton router 那一列，B200 是兩個 routing kernel，shared expert 另外在 side stream。KDA 那一列是 `cutlass_kda_decode_mtp`，conv 也在裡面。
- router 這段已經追平 B200，KDA 合併 kernel 還是 B200 的 2 倍。

- round 4 的 router 那段從 4 個 kernel 變成 2 個（GEMM + Triton router），沒有多出 mask kernel。
- J 在 bs16 的 trace 上反而慢 1.3 µs，跟 microbench 相反。bs16 的 delta kernel 在 trace 是 26.1 µs，microbench 只有 20.8 µs，表示實際跑的時候 state 讀取比 microbench 慢，合併 kernel（V tile 32、CTA 比較少）受影響更大。所以另外跑了只在 bs ≤ 8 開 J 的版本（12.4）。

### 12.4 沒開 profiler 的結果和 GSM8K

ISL 1024 / OSL 1024，MTP5 golden AL（3.56），conc × 5 個 prompt。

| TPOT（ms，median / mean） | conc6 | conc16 |
|---|---|---|
| base | 3.61 / 3.67 | 5.08 / 5.10 |
| opt3（A–G） | 3.49 / 3.60 | 4.95 / 4.95 |
| round 4（+H），跑兩次 | 3.43 / 3.55、3.44 / 3.54 | 4.97 / 4.99、4.88 / 4.98 |
| round 5（+H' I J），跑兩次 | 3.40 / 3.56、3.38 / 3.52 | 4.89 / 4.90、4.85 / 4.94 |
| round 5，J 只在 bs ≤ 8 開（`wt-glm53-1111-kda8`） | 3.41 / 3.53 | 4.87 / 4.89 |

- 同一個樹跑兩次，conc16 的 median 可以差 0.09 ms，所以 0.05 ms 以下的差別看不出來。
- round 5 比 round 4：conc6 約 −0.04 ms，conc16 約 −0.05 ms（兩次平均）。比 base：conc6 −0.22 ms（−6%），conc16 −0.21 ms（−4%）。
- J 在 bs16 開或不開，沒開 profiler 量不出差別，trace 上開了反而慢。所以建議用 kda8 的版本：J 只在 bs ≤ 8 開。
- GSM8K（sgl-eval，1319 題）：opt3 96.66%，round 4 97.04%，round 5 **97.27%**，沒有掉。

### 12.5 看過但不做的

| 項目 | 原因 |
|---|---|
| gated norm 併進 delta rule kernel 的 epilogue | delta kernel 的 V tile 只有 8–16，一個 CTA 拿不到一個 head 的 128 個 V。原 code 也註明 V tile 128 慢 2 倍 |
| f_b / g_b 的 batched GEMM | 實際 3.4–3.8 µs，已經接近一個 kernel 的下限。改成一個 block-diagonal GEMM 反而 4.2–4.4 µs |
| router GEMM 改 aiter Triton `gemm_a16w16` | 36 row 5.7 µs、96 row 18 µs，不如 split-K GEMV |
| f_b 併進 delta rule kernel | ReplaySSM 的 ring 要存 g，delta kernel 每個 V tile 都要重算一次 128×128 的 f_b，改動大，今天沒時間驗證 |

### 12.6 檔案

| 內容 | 路徑 |
|---|---|
| round 4 / 5 的 trace | `/shared_nfs/jacchang/prof_1111/mi/{prof4,prof5}/` |
| 沒開 profiler 的 bench 和 GSM8K | `/shared_nfs/jacchang/prof_1111/perf1111/{perf4,gsm4,perf5,gsm5,perf4b,perf5b,perf5c}/` |
| 新工具（未 commit） | `tools/`：`bench_router_1111.py`、`bench_router_gemm_1111.py`、`bench_fgb_1111.py`、`check_router_partials_1111.py`、`chain_opt1111.sh`、`wait_chain1111.sh`、`wait_then1111.sh` |
| sglang 樹 | `~/PR/wt-glm53-1111-opt`（到 H）、`~/PR/wt-glm53-1111-kda`（到 J）、`~/PR/wt-glm53-1111-kda8`（J 只在 bs ≤ 8） |

## 13. 11/12：AgentX 重跑，和 ReplaySSM KDA verify 的新 kernel（最後用 FlyDSL）

撰寫：2026-11-12 11:40，12:20 更新 FlyDSL（台灣時間）。

### 13.1 AgentX（node 152，kda8 樹 vs 同一天的 base，ISL 約 128K，MTP5 golden AL，ReplaySSM 開）

| P90 Int | opt（kda8） | 同一天 base | 11/10 CSV | B200 |
|---|---|---|---|---|
| conc16 | **231.1** | 222.1 | 224.8 | 256.8 |
| conc32 | **147.1** | 146.2 | 148.4 | 164.4 |

| 其他指標（opt / base） | conc16 | conc32 |
|---|---|---|
| P50 Int | 322.1 / 301.9（+6.7%） | 250.1 / 233.4（+7.2%） |
| mean Int | 299.0 / 276.1（+8.3%） | 212.0 / 202.5（+4.7%） |
| TTFT p90（s） | 0.591 / 0.541 | 0.736 / 0.749 |
| output tok/s/GPU | 182 / 181 | 341 / 338 |

- 同一天 base 和 11/10 CSV 差 −1.2%（conc16）和 −1.5%（conc32），這是 run 之間的雜訊大小。
- conc16 的 P90 Int +4.1%，B200 的領先從 14% 縮到 11%。
- conc32 的 P50 和 mean 都進步 5–7%，P90 只有 +0.6%。用 `agentic_tail.py` 拆開決定 P90 的 P85–P95 那群 request（opt / base）：

  | P85–P95 | opt | base |
  |---|---|---|
  | TPOT（ms） | 6.89 | 6.91 |
  | decode step 部分（ms） | **5.00** | 5.23（−4.4%） |
  | stall 部分（ms） | **1.89** | 1.68（+0.21） |
  | median chunk gap（ms） | 17.03 | 17.80 |
  | 它們 decode 期間碰到的 prefill（k token） | 133 | 119 |
  | decode batch p10 / p50 / p90 | 7 / 15 / 19 | 8 / 16 / 19 |

  - decode step 確實快了 4.4%，但這次 run 的 tail request 剛好多碰到 12% 的 prefill，stall 多了 0.21 ms，把省下的 0.23 ms 抵掉了。opt 沒有改 prefill，所以 stall 的差別是 run 之間的變動。
  - 把 stall 拿掉後的 P90 Int 上限：opt 192.9，base 186.2（+3.6%），這才是 decode 優化的效果。
  - P85–P95 的 stall 占 27%，conc32 的 P90 有一部分卡在 prefill，decode 變快只能拉動另外 73%。
  - 這次 AgentX 用的是 kda8 樹，還沒有新的 KDA kernel（bs16 每個 cycle 再省 0.70 ms，約 4%）。
- 結果在 `/shared_nfs/jacchang/agentic_1112_opt/{opt,base}/results/`。

### 13.2 ReplaySSM KDA verify 為什麼在 MI355X 沒變快

- B200 的 `cutlass_kda_decode_mtp` 原本卡在寫 per-step state，ReplaySSM 拿掉這些寫入，所以時間減半。
- MI355X 的 Triton kernel 不是卡在記憶體。bs16 讀 16 MB state 只要約 4 µs，kernel 卻要 25 µs。
- 真正的瓶頸是 VALU 和延遲：
  - Triton 把 K = 128 分在 32 個 lane 上，每個 token 的兩個 K 內積都要跨 lane 做 DPP reduction。
  - 每個 SIMD 只有約 1 個 wave，延遲蓋不掉。
  - 每個 V tile 都重算一次 conv、gate、l2norm。
- 拿掉 state 寫入對它幫助不大。

### 13.3 新的 KDA verify kernel（`kda_verify_gfx950`）

一個 kernel 做完 conv1d、gate、beta、l2norm 和 delta rule，同時寫 ReplaySSM 的 4 個 ring 和 intermediate conv window。

先用 HIP 寫到 v5，再照同樣的設計用 FlyDSL 重寫。兩個一樣快（FlyDSL 在 bs16 以上還略快），FlyDSL 比較好維護，所以**最後用 FlyDSL**。HIP v5 只留在開發目錄當備份。FlyDSL 版的細節在 13.3.1。

| 設計 | 內容 |
|---|---|
| grid | 每個（request、v head、V tile）一個 256-thread block。V tile 在 bs ≤ 4 用 32，其餘用 64 |
| 準備階段 | 一個 head 的 q、k、v conv 和 gate 在 block 內只算一次，經 LDS 共用 |
| recurrence | 每個 V 欄由 8 個 lane 分 K（每 lane 16 個 K），內積用 3 次 DPP 加總；state 放在 register，用 `v_pk_fma_f32` |
| l2norm | 平方和在準備階段用 wave reduction 算好，norm 在 recurrence 裡乘上，不再改寫 q、k |
| global load | 全部無條件發出（越界的位址夾到合法範圍），state 排在最後，準備階段不用等 state |

| µs（GPU 0，HIP graph，40 組 slot 輪流，沒開 profiler） | bs1 | bs4 | bs6 | bs8 | bs12 | bs16 | bs32 | bs64 |
|---|---|---|---|---|---|---|---|---|
| conv1d + delta rule（pair，kda8 在 bs > 8 用這個） | 12.65 | 14.72 | 18.26 | 20.03 | 24.72 | 27.22 | 39.78 | 64.93 |
| Triton 合併 kernel（J，kda8 在 bs ≤ 8 用這個） | 11.25 | 14.25 | 16.29 | 18.74 | 23.15 | 25.48 | 91.95 | 172.37 |
| HIP kernel（v5，備份） | 6.06 | 6.52 | 7.29 | 8.43 | 11.24 | 12.37 | 19.59 | 32.50 |
| **FlyDSL kernel（最後用這個）** | **5.91** | **6.46** | **7.44** | **8.39** | **11.14** | **12.28** | **19.26** | **31.92** |
| B200 trace（`cutlass_kda_decode_mtp`） | | | 9.42 | | | 14.52 | | |

- HIP 和 FlyDSL 是同一次 sweep，每個 bs 取較快的 V tile。
- 4 個 ring 和 intermediate conv window 在 bs 1–64 都跟 pair bit 一樣。輸出 o 最大差 3e-5，因為 reduction 順序不同，l2norm 也改用 `v_rsq_f32`。HIP 和 FlyDSL 的誤差一樣。
- FlyDSL 比 kda8 實際用的路徑快：bs6 −8.9 µs，bs16 −14.9 µs，bs32 −20.5 µs。
- 34 層 KDA 換算下來，每個 cycle 約省 0.30 ms（bs6）和 0.51 ms（bs16）。trace 的實際數字在 13.4。
- HIP 的開發過程（都是 microbench，bs6 / bs16，µs）：

  | 版本 | bs6 | bs16 |
  |---|---|---|
  | v1：Triton 的 layout 照搬成 HIP | 11.55 | 18.03 |
  | v2：8 lane 分 K + DPP；l2norm 併進準備階段 | 9.20 | 14.18 |
  | v3：`v_pk_fma_f32`、`v_rsq_f32` | 8.58 | 13.16 |
  | v5：global load 不放在分支裡 | 7.29 | 12.36 |

- 踩到的坑：
  - hipcc 預設會把 mul + add 合成 FMA，ring 因此差 1 個 bf16 ulp。conv 那段要加 `#pragma clang fp contract(off)`。
  - 分支裡的 load 會被編譯器在每個分支結尾插 `vmcnt` wait，變成好幾次串行的 global 延遲。
- HIP 版的單元測試：`test_fused_kda_conv_recurrent_verify` 25 pass。`test_kda_fused_verify_backend` 的 10 個 fail 在只有 kda8 patch 的樹上一模一樣。

#### 13.3.1 FlyDSL 版

- 設計和 HIP v5 一樣（grid、8 lane 分 K + DPP、state 放 register、load 不放在分支裡）。寫法照 sglang 裡現成的 Kimi-K3 `kda_flydsl`：先 import aiter，用 `_run_compiled` 執行。
- 編譯選項：`amdgpu-expert-scheduling-mode` 和 kernarg preload（跟 aiter 的 `tensor_shim` 一樣）。不設 `waves_per_eu`，設 3 或 4 會 spill 到 scratch，慢很多。
- 踩到的坑：V tile 64 一開始用了 340 VGPR + 84 AGPR，比 HIP 慢。
  - 原因：編譯器把每個 token 的 q 內積（輸出 o）全部延到最後才算，所以 6–8 份 state 快照和 q 都一直佔著 register。
  - `sched_barrier` 沒用（ISA 裡看不到它）。
  - 解法：在 token t 的 state 更新前放一個 inline asm 的 `v_mov_b32`，讓它依賴 token t−2 的輸出。這樣編譯器必須先算完 t−2 的內積。V tile 64 降到 158 VGPR，V tile 32 約 80 VGPR（V tile 32 的落後量用 0）。
  - 落後 3 個 token 在 bs64 會因為 VGPR 變多（204）、occupancy 下降而變慢。落後 2 個整體最好。
- buffer store 的 mask 在預設 num_records 下不安全，所以 store 都用 `if` 判斷（slot ≥ 0）。
- 單元測試（kdafly 樹）：`test_fused_kda_conv_recurrent_verify` 31 pass、1 skip（CUDA 才有的 green context 測試）。新加的 `test_gfx950_ring_matches_unfused` 有 6 個 case：bs1 / 6 / 16（V tile 32 和 64）、T = 8、GQA（4 個 q head、8 個 v head）、負的 slot。ring 全部 bitwise 比對。pre-commit 通過。
- 在 server 走的 dispatch 路徑上量（`bench_kda_preload_1112.py`，auto）：bs6 7.39、bs16 12.24、bs64 31.79 µs。
- 整合：`kda_backend` 在 gfx950 + ReplaySSM、draft token 6 或 8、FlyDSL 和 aiter 可以 import 時，所有 batch 都走合併路徑。`fused_kda_conv_gating_verify` 再交給 FlyDSL kernel。import 失敗就回到原本的 Triton 路徑。

#### 13.3.2 rocprof 分析（11/16，node 152）

目的：送 PR A 前確認這個 kernel 還有沒有簡單的優化。結論：沒有，PR A 照 `508c4cc3a3` 送。

| decode batch（開 profiler） | kernel µs fp32 / bf16 | 頻寬 fp32（TB/s） | VALUBusy fp32 |
|---|---|---|---|
| bs1 | 6.20 / 6.04 | 0.37 | 2.7% |
| bs6 | 7.78 / 7.56 | 1.30 | 10.5% |
| bs16 | 12.84 / 11.40 | 2.05 | 22.9% |
| bs32 | 21.58 / 19.62 | 2.43 | 34.6% |
| bs64 | 37.08 / 31.96 | 2.82 | 50.2% |

- 不是 memory-bound：bs16 讀 19.9 MB、寫 5.3 MB，約 HBM 上限的 1/4。
- recurrence 已經是 packed FP32（v64 有 288 個 `v_pk_fma_f32`），指令數接近最少。
- bs16 一個 wave 約 21k cycle（ATT）：開頭等 kernarg 和 slot index ~2.3k，發 37 個 load ~3.5k（25 個是 2 byte 的 `buffer_load_ushort`），prep 計算和最多 48 個 2 byte store ~5.4k，recurrence ~6.5–7.3k。同一個 CU 的 wave 在 370 cycle 內全部開始，kernel 時間約等於較慢 CTA 的 wave 壽命。

| 試過的改法（microbench，GPU 1，對 PR A 版本） | 結果 |
|---|---|
| index 改用 vector load + `sched_barrier`，4 個 index pointer 移到 kernarg preload 區 | ISA 順序對了，輸出 bitwise 一樣；−0.3% 到 −3.1%（bs6 fp32 +2.5%），baseline 重跑本身 ±1.5% |
| `waves_per_eu=4` | spill 17 個 VGPR，慢 12–20% |
| `waves_per_eu=3` | −1% 到 −3%，跟上一列差不多 |

- 開頭的延遲沒省到時間：load 都擠在同一條 memory pipeline，bs16 時同一個 CU 的另一個 CTA 會補上 stall。
- 剩下的方向都要改 thread 配置：2 byte 存取改成一個 lane 處理 2 個 channel、不讓用不到的 wave 發 v/a load、bs ≥ 32 用 V tile 128、ring/iw store 平均分給兩個 V tile。全部做完估計 kernel 快 10% 左右。
- 這個 kernel 在 bs16 trace 是 34 × 12.1 µs ≈ 0.41 ms，占 17.09 ms cycle 的 2.4%，所以 kernel 快 10% 換算 E2E 只有約 0.24%。
- rocprofv3 1.3.5（ROCm 10 nightly）會因為 LLVM option 重複註冊而 deadlock，連純 torch 也一樣。做法：從 FlyDSL cache 取出 hsaco，用 C++ HIP launcher 跑，在 ROCm 7.0 container（rocprofv3 1.0）裡量。每次 `--pmc` 只放一組，TCC counter 一組不超過 3 個，否則會卡住。
- 資料：`/shared_nfs/jacchang/kdaprof1116/`（`prof70/`、`att70_bs16_fp32/`、`hsaco*/`）。原型 tree：`~/PR/wt-glm53-1116-kdaopt`（未 commit）。工具（未 commit）：`tools/` 的 `kda_dump_hsaco_1116.py`、`kda_launch_spec_1116.py`、`kda_hsaco_launch_1116.cpp`、`prof_kda_hsaco_1116.sh`、`kda_prof70_summary_1116.py`、`kda_att_phases_1116.py`、`ab_kda_verify_1116.py`。

### 13.4 GSM8K 和 trace（node 247，GPU 4–7）

| GSM8K（1319 題，thinking，temperature 1.0） | 分數 | stop / truncated | log 裡的平均 accept len |
|---|---|---|---|
| 11/11 第 5 輪（kda8） | 97.27% | | |
| kdav2（HIP v5） | 97.50% | 99.85% / 0.15% | 3.83 |
| kdafly（FlyDSL）第 1 次 | 96.44% | 99.24% / 0.76% | 3.69 |
| kdafly（FlyDSL）第 2 次 | 96.82% | 99.62% / 0.38% | 3.82 |

- FlyDSL 兩次平均 96.63%，HIP 一次 97.50%。歷次分數在 95.83–97.50% 之間（第 11.3、12.4 節），都在這個範圍裡。
- 第 2 次的 accept len 3.82 跟 HIP 的 3.83 一樣。kernel 算錯的話，target 的輸出會跟 draft 對不上，accept len 會掉。
- 結論：沒有掉準確率，第 1 次偏低是 temperature 1.0 抽樣的雜訊。

| trace（ISL 128K，MTP5 golden AL，2 個 cycle 平均） | bs6 第 5 輪 | bs6 HIP v5 | bs6 FlyDSL | bs16 第 5 輪 | bs16 HIP v5 | bs16 FlyDSL |
|---|---|---|---|---|---|---|
| KDA verify kernel（µs） | 19.24 | 8.63 | 8.85 | 32.15 | 12.40 | 12.09 |
| KDA+MoE 層的 wall（µs） | 177.4 | 167.3 | 167.9 | 257.0 | 236.3 | 236.6 |
| 45 層（ms） | 9.50 | 9.20 | 9.21 | 13.30 | 12.59 | 12.61 |
| cycle wall（ms） | 13.05 | 12.27 | 12.30 | 17.47 | 17.03 | 17.09 |
| B200 KDA kernel（µs） | 9.42 | | | 14.52 | | |

- 45 層省了 0.30 ms（bs6）和 0.70 ms（bs16），跟 34 層 × kernel 省下的時間對得上。HIP 和 FlyDSL 在 trace 上一樣快。
- KDA verify kernel 在 trace 上已經比 B200 快（bs6 8.9 vs 9.4，bs16 12.1 vs 14.5）。
- cycle wall 的變化和 45 層不一致，因為 tail 有雜訊：bs6 的 tail 少了 0.47 ms，bs16 的 tail 多了 0.3 ms（rank 之間等待），都跟這個 kernel 無關。
- trace 在 `/shared_nfs/jacchang/kda1112/traces/{kdav5,kdafly}_c{6,16}.trace.json.gz`。

### 13.5 檔案

| 內容 | 路徑 |
|---|---|
| sglang 樹（最後用的） | `~/PR/wt-glm53-1112-kdafly`（kda8 + FlyDSL kernel + 單元測試） |
| sglang 樹（HIP 備份） | `~/PR/wt-glm53-1112-kdav2`（kda8 + HIP v5） |
| kernel | `python/sglang/kernels/ops/attention/linear/kda_verify_gfx950/`（`kernel.py`、`__init__.py`） |
| FlyDSL 開發版（有 `KDA_FLY_*` 調參 env）和 sweep 結果 | `/shared_nfs/jacchang/kda1112_dev/kda_verify_fly.py`、`sweep_final.txt`、`bench_kdafly_tree.txt` |
| HIP 開發版（有 `KDA_DEV_*` 分段 macro） | `/shared_nfs/jacchang/kda1112_dev/kda_verify_v{2,3,4,5}.hip` |
| 新工具（未 commit） | `tools/`：`bench_kda_preload_1112.py`、`kda_fly_dev_1112.py`、`kda_hip_dev_1112.py`、`kda_hip_ring_diff_1112.py`、`kda_kernel_meta_1112.py`、`chain_kdav2_1112.sh` |

## 14. 11/12 下午：allreduce、MoE GEMM 依 M tune、k-pool topk

bs16 的 FlyDSL trace 跟 B200 比，每個 cycle 差最多的是 allreduce 0.72 ms、MoE GEMM 0.61 ms、DSA MLA proj/quant 0.34 ms、KDA proj 0.29 ms、topk 0.27 ms。這次看了其中三項。trace 都是 ISL 128K、MTP5 golden AL、ReplaySSM 開，node 152 GPU 0–3，sglang 樹是 `wt-glm53-1112-kdafly`。

### 14.1 結果

| trace（2 個 cycle 平均） | bs6 FlyDSL | bs6 + topk v2 | bs6 + topk v2 + MoE exact M | bs16 FlyDSL | bs16 + topk v2 | bs16 + topk v2 + MoE exact M |
|---|---|---|---|---|---|---|
| k-pool topk kernel（µs） | 38.0 | 17.3 | | 39.9 | 18.0 | |
| DSA+MoE 層的 wall（µs） | 332.9 | 310.1 | 311.7 | 442.7 | 419.0 | 419.8 |
| KDA+MoE 層的 wall（µs） | 167.9 | 167.7 | 166.5 | 236.6 | 236.1 | 236.6 |
| 45 層（ms） | 9.21 | 8.95 | 8.93 | 12.61 | 12.34 | 12.36 |
| cycle wall（ms） | 12.30 | 11.96 | 11.94 | 17.08 | 16.68 | 16.74 |

| GSM8K（1319 題，thinking，temperature 1.0） | 分數 | truncated | accept len |
|---|---|---|---|
| kdafly（第 13.4 節，兩次） | 96.44% / 96.82% | 0.76% / 0.38% | 3.69 / 3.82 |
| kdafly + topk v2 | 97.12% | 0.38% | 3.79 |
| kdafly + topk v2 + MoE exact M | 96.74% | 0.61% | 3.81 |

- topk v2 每個 DSA 層省約 21 µs，11 層共 0.23–0.26 ms，cycle 快 2.3–2.7%。GSM8K 沒掉。
- MoE exact M 沒有效果，差異在雜訊內（見 14.3）。

### 14.2 k-pool topk：改用 v2 kernel

- v1 `kpool_topk_transform<512>` 每個 row 只用 1 個 1024-thread CTA 做 radix select。logits 分布越集中，要重掃整個 row 的次數越多，所以時間跟資料有關。
- v2（`topk_transform_kpool_v2`，DSV4 的 topk v2 kernel，有 HIP code）在 HIP 上本來就能跑，只是 `model_hook` 對 DSA model 在 HIP 預設把 `SGLANG_OPT_USE_TOPK_V2` 關掉，`kpool_fp8_index.py` 也只在 CUDA 開。這次把 `_kpool_topk_v2_enabled()` 改成 CUDA 或 HIP，並在啟動時設 `SGLANG_OPT_USE_TOPK_V2=1`。

| microbench（32K 長度，topk 2048，沒開 profiler，µs） | v1 | v2 |
|---|---|---|
| relu 分布 | 18.2–18.9 | 10.7–10.8 |
| narrow 分布 | 110 | 27 |

- index 集合跟 v1 一樣（narrow 分布因為同分只有 0.99 一樣）。trace 上 v1 是 34–40 µs，表示真實 logits 比 relu 分布更難。
- 還沒做：不加新 env 的開法。建議在 `model_hook` 的 HIP 分支，`index_kpool > 1` 的 model 沒設 env 時預設開 v2，這就是這次測的設定。

### 14.3 MoE GEMM：M = 6, 12, …, 288 的 tune

- verify 的 M 是 decode batch × 6，aiter 查 config 時會把 M 補到 2 的次方（M=36 用 64 的 config，M=96 用 128 的）。
- 在 node 247 GPU 0–3 用 `gemm_moe_tune.py --mxfp4-flydsl --mxfp4-search-mode full --fast-scan` tune 了 48 個 M，結果在 `/shared_nfs/jacchang/moe1112/glm53_flash_fp4_m6_tuned_fmoe.csv`（新檔，沒改舊 CSV）。
- aiter patch `tools/patch_aiter_moe_exact_m.py`：CSV 有實際 M 的 row 就用它，沒有就跟原本一樣用 2 的次方的 M。trace 確認 kernel 名稱換成新的 config。
- tuner 表上看起來快很多（M=36 是 108 µs，舊的 64 config 是 135 µs），但舊的數字是在 M=64 量的，不是在 M=36。實際 trace 上 stage1 + stage2：

| 每次 MoE（µs） | bs6 原本 | bs6 exact M | bs16 原本 | bs16 exact M |
|---|---|---|---|---|
| sort + stage1 + stage2 | 76.2 | 74.9 | 118.9 | 118.3 |

- 結論：2 的次方的 config 在這些 M 已經差不多是最好的，不用這個 patch。`jacchang_pr48875` 的 aiter 已經還原。
- 估計：bs16 有 864 個 (token, expert)，289 個 expert 大部分都會被讀到（每個約 3.2 MB，全部約 0.9 GB）。stage1 + stage2 是 112 µs，換算約 8 TB/s，已經接近 HBM 頻寬上限，所以換 tile 沒用。要縮小跟 B200 的差距，得減少讀的 weight 或讓 stage1 / stage2 重疊，不是 tune。

### 14.4 allreduce：1-stage vs 2-stage，和跟 mHC 合併

| TP4 allreduce（hidden 4096 bf16，沒開 profiler，µs） | 1-stage | 2-stage | 目前選的 |
|---|---|---|---|
| 12 rows | 7.3 | 5.9 | 1-stage |
| 36 rows（bs6 verify） | 12.2 | 7.2 | 2-stage |
| 96 rows（bs16 verify） | 22.7 | 11.9 | 2-stage |
| 288 rows | 52.0 | 25.8 | 2-stage |

- aiter 用 160 KB 當門檻，verify 的 36 / 96 rows 已經走 2-stage。只有 12 rows 選錯，差 1.4 µs，影響很小。
- 96 rows 的 11.9 µs 約是 5 µs 固定延遲加 7 µs 傳輸，傳輸已經接近 3 條 XGMI link 的上限。
- mHC 的 kernel（沒開 profiler，36 / 96 rows）：`hc_post` 2.3 / 2.7 µs，`hc_pre` 8.8 / 9.4 µs。現在的合併版 post_pre 是 10.16 / 12.84 µs，分開跑是 11.15 / 12.07 µs，96 rows 合併版反而慢一點。
- 把 allreduce + `mhc_post` + 下一層 `mhc_pre_gemm_sqrsum` 合成一個 kernel，估計每個 cycle 省 0.3–0.45 ms（每層省 1–2 個 kernel 邊界和一次讀寫 hidden），但要改 aiter custom allreduce，工作量大，還沒開始。

### 14.5 檔案

| 內容 | 路徑 |
|---|---|
| topk v2 trace | `/shared_nfs/jacchang/topk1112/traces/topkv2_c{6,16}.trace.json.gz` |
| topk v2 + MoE exact M trace | `/shared_nfs/jacchang/moe1112/topkv2moe_c{6,16}.trace.json.gz` |
| MoE tune 輸入、輸出、log | `/shared_nfs/jacchang/moe1112/glm53_flash_fp4_m6_{untuned,tuned,profile}_fmoe.csv`、`tune.log` |
| 新工具（未 commit） | `tools/`：`bench_kpool_topk_1112.py`、`bench_mhc_1112.py`、`tune_moe_m6_1112.sh`、`patch_aiter_moe_exact_m.py`、`prof_opt1112.sh`、`bench_opt1112.sh`（多一個 MoE CSV 參數） |

## 15. 各個優化在哪個 branch（11/12 push）

全部在自己的 fork，不是 PR。sglang 的 branch 都以 `09111fe9a3`（第 1 節的 1106 PR stack，10 個 open PR 合在一起）為 base。

### 15.1 sglang（[Jacob0226/sglang](https://github.com/Jacob0226/sglang)）

| branch | 內容 | 本機 worktree |
|---|---|---|
| [`jacob/glm53-flash-1112-idxfix`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1112-idxfix)（`d4220d038e`，11/12 21:40 push） | kdafly 再加上第 16.3 節：DSA indexer double Hadamard 的修正，和 key LN + k-pool 寫入 + query quant 合成一個 launch | `~/PR/wt-glm53-1112-idxfix` |
| [`jacob/glm53-flash-1112-kdafly`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1112-kdafly) | 11/12 下午**最後的樹**（有第 16.2 節的 bug）：B、D、E、F、G（第 11 節）、H'、I、J（J 只在 bs ≤ 8，第 12 節）、FlyDSL KDA verify kernel 和單元測試（第 13 節）、k-pool topk v2 在 HIP 可用（第 14.2 節，要設 `SGLANG_OPT_USE_TOPK_V2=1`） | `~/PR/wt-glm53-1112-kdafly` |
| [`jacob/glm53-flash-1112-kdav2`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1112-kdav2) | 備份：同上，但 KDA verify 是 HIP v5，沒有 topk v2 的改動 | `~/PR/wt-glm53-1112-kdav2` |
| [`jacob/glm53-flash-1111-kda8`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1111-kda8) | round 5：到 J，J 只在 bs ≤ 8 | `~/PR/wt-glm53-1111-kda8` |
| [`jacob/glm53-flash-1111-kda`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1111-kda) | round 5：到 J，所有 bs 都開 J | `~/PR/wt-glm53-1111-kda` |
| [`jacob/glm53-flash-1111-opt`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1111-opt) | round 4：B、D、E、F、G、H | `~/PR/wt-glm53-1111-opt` |
| [`jacob/glm53-flash-1111-defg`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1111-defg) | 中間版：D、E、F、G | `~/PR/wt-glm53-1111-mhc` |
| [`jacob/glm53-flash-1110-idx-exp`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1110-idx-exp) | 實驗：F 的第一版（有 env） | `~/PR/wt-glm53-1110-idx` |
| [`jacob/glm53-flash-1110-idx-topk-exp`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1110-idx-topk-exp) | 實驗：F 的第一版 + `kpool_topk_transform.cuh` 的修改 | `~/PR/wt-glm53-1110-idx-topk` |
| [`jacob/glm53-flash-1108-opt-exp`](https://github.com/Jacob0226/sglang/tree/jacob/glm53-flash-1108-opt-exp) | 實驗：k-pool topk、speculative `topk1`、sharded verify（AgentX） | `~/PR/wt-glm53-1108-opt` |

每棵樹裡沒 commit 的只有 `python/sglang/bench_serving.py`（給 GLM.sh 用的 shim）。

### 15.2 aiter（[Jacob0226/aiter](https://github.com/Jacob0226/aiter)）

| branch | 內容 |
|---|---|
| [`jacob/glm53-flash-mi355x-1112`](https://github.com/Jacob0226/aiter/tree/jacob/glm53-flash-mi355x-1112) | **跑 benchmark 時的 aiter**：base `285579b62`（`rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261106` 裡的 aiter）+ C（MoE sort crossover）+ A（`glm53_flash_bf16_tuned_gemm.csv` 換成 tune1111 r123，`glm5_bf16_tuned_gemm.csv` 拿掉被新 tune 取代的 7 個 o_proj row）+ ROCm/aiter#6435 的 `glm53flash_fp4_tuned_fmoe.csv`。只改了 Python 和 CSV，checkout 到 `/sgl-workspace/aiter` 不用重新 build。本機 `~/PR/wt-aiter-glm53-1112` |
| [`jacob/moe-bm16-sort-crossover`](https://github.com/Jacob0226/aiter/tree/jacob/moe-bm16-sort-crossover) | C，以 upstream main 為 base。11/16 決定不開 PR（第 17.2.1 節），留著當參考。本機 `~/PR/wt-aiter-sort-crossover` |
| [`jacob/glm53-flash-kda-fused-gemm-tp4`](https://github.com/Jacob0226/aiter/tree/jacob/glm53-flash-kda-fused-gemm-tp4) | TP4 fused KDA in-proj 的 tune（之前就 push 過） |
| [`jacob/glm53-flash-a8w8-blockscale-bpreshuffle`](https://github.com/Jacob0226/aiter/tree/jacob/glm53-flash-a8w8-blockscale-bpreshuffle) | TP4 a8w8 blockscale bpreshuffle GEMM config |

### 15.3 沒放進 git 的

- 第 14.3 節的 MoE M=6..288 tune 和 exact-M patch：沒效果，沒 push。CSV 在 `/shared_nfs/jacchang/moe1112/`，patch 是 `tools/patch_aiter_moe_exact_m.py`。
- `~/SGLang-benchmarks/tools/` 裡 11/11–11/12 的新工具都還沒 commit。
- 用 branch 重現最後的結果：image `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261106`，aiter checkout `jacob/glm53-flash-mi355x-1112`，`PYTHONPATH` 指到 `jacob/glm53-flash-1112-kdafly` 的 `python/`，env 照 `tools/prof_opt1112.sh` 再加 `SGLANG_OPT_USE_TOPK_V2=1`。用了這個 aiter branch，應該就不用再另外給 `BF16_CSV`、`AITER_CONFIG_FMOE` 裡的 aiter6435 CSV 和 sort patch（還沒實際用這個 branch 從頭跑過一次）。

## 16. 11/12 晚上：allreduce + mHC 合併（停損）、DSA indexer 的 double Hadamard bug 和合併、AgentX、router 合併（停損）

### 16.1 allreduce + `mhc_post` + 下一層 `mhc_pre_gemm_sqrsum` 合成一個 kernel：不做

在 aiter worktree 寫了 `fused_ar_mhc_post_pre`（2-stage allreduce 的 reduce-scatter，接 `mhc_post` 和下一層 pre GEMM 的 partial，pull + LDS 版，最多 128 rows）。停損條件是 96 rows 要到約 14 µs，結果差很遠。

| node 152 GPU 4–7，CUDA graph，每次呼叫不同的輸入（µs） | 36 rows | 96 rows | 128 rows |
|---|---|---|---|
| 合併版 | 18.63 | 26.12 | 29.49 |
| 現在 MoE 後面的路徑（AR + `mhc_post` + pre GEMM） | 19.15 | 26.14 | 28.68 |
| 現在 attention 後面的路徑（AR + `mhc_fused_post_pre`） | 17.55 | 25.28 | 27.64 |

- 下限約是 AR + 5–8 µs：要多一次跨 GPU 的來回（reduce 完要再拉別人的結果）、要先讀 fn，而且 HBM 上的 residual 讀寫只靠 64 個 CTA（aiter `Signal` 最多 80 個 block）。
- 36 rows 只省 0.5 µs，96 rows 沒省，不接進 model。kernel 留在 `~/PR/wt-aiter-glm53-1112`（沒 commit），備份和 ISA 在 `/shared_nfs/jacchang/ar1112/`。

### 16.2 DSA indexer 的 bug：verify 時 query 被轉了兩次 Hadamard

- PR [#44417](https://github.com/sgl-project/sglang/pull/44417)（stacked x projection）的 `_get_q_k_fused_x_proj` 一定會對 query 做 `rotate_activation`。
- PR [#44661](https://github.com/sgl-project/sglang/pull/44661)（`SGLANG_DSA_FUSE_HADAMARD_QUANT`）把 Hadamard 移進 `fused_hadamard_act_quant`，只改了 `_get_q_k_bf16`。兩個 PR 疊在一起（`62d2711d65` 解衝突時）沒有改到 `_get_q_k_fused_x_proj`。
- 結果：GLM-5.3-Flash 的 target verify（MTP 開時的每一步 decode）query 轉了兩次，等於沒轉，key 有轉（k-pool 寫入時）。indexer 的分數變成 q·Hk，不是 Hq·Hk。trace 上看得到 `fast_hadamard_transform` 和 `_hadamard_quant_kernel` 都在跑。
- 為什麼沒發現：GSM8K 的 context 短，topk 2048 會選到全部 token，indexer 錯了也沒差。AgentX 用 golden AL，效能數字不受影響。`test_dsa_kpool_fused_x_proj_gfx950.py` 的 indexer 沒設 `fuse_hadamard_quant`，所以沒測到這個組合。
- 影響範圍：開了 `SGLANG_DSA_FUSE_HADAMARD_QUANT=1` 的所有 opt run（`GLM.sh`、`prof_opt111x.sh`、`agentx_opt111x.sh`）。兩個 PR 在 upstream 都還沒 merge，#44417 最新的 head `380ee8c9bc` 也還是無條件 rotate。**11/17 決定把 #44661 從 stack 拿掉，env 也不要設**（第 17.3 節），所以不再需要這個判斷。

| `tools/check_idx_double_hadamard_1112.py`（16 rows，65536 個 key，top 2048） | q dequant 跟正確值的 cosine | top-2048 重疊 |
|---|---|---|
| 修正後 | 1.00000 | 1.0000 |
| 原本（轉兩次） | 0.10320 | 0.0545 |

### 16.3 修正，和把 indexer verify 的 5 個 kernel 併成 1 個

新的 branch `jacob/glm53-flash-1112-idxfix`（從 kdafly 開，worktree `~/PR/wt-glm53-1112-idxfix`，commit `d4220d038e`，已 push）：

- 修正：`fuse_hadamard_quant` 開時，`_get_q_k_fused_x_proj` 不再 rotate query。這也少掉一個 `fast_hadamard_transform`。
- `fused_hadamard_act_quant` 直接吃有 stride 的 `head_weights`（`proj[:, 2d:]`），少掉一個 `.contiguous()` copy。
- `kpool_write_tail_and_maybe_compress` 加兩個可選參數：`key_norm` 在 kernel 裡做 key 的 LayerNorm（讀 fp32 的 `proj[:, :d]`），`query_quant` 讓同一個 grid 後面的 program 做 query 的 Hadamard + fp8 quant。HIP 上 `alt_stream` 是 None，verify 只有一條 stream，所以兩件獨立的工作可以放在同一個 launch。
- verify 每個 DSA 層從 `_layer_norm_fwd_1pass` → `fast_hadamard_transform` → `_kpool_write_tail_and_maybe_compress` → copy → `_hadamard_quant_kernel` 5 個 kernel 變成 1 個。

| microbench（node 152 GPU 4，CUDA graph，沒開 profiler，µs） | 原本 5 個 kernel | 只修 bug（3 個） | 合併成 1 個 |
|---|---|---|---|
| decode batch 6 | 15.49 | 10.96 | 6.54 |
| decode batch 16 | 16.20 | 11.12 | 6.67 |

- 每個 cycle 11 個 DSA 層，估計省約 0.1 ms（bs16 約 0.6%）。
- 單元測試：`test_dsa_kpool_fused_x_proj_gfx950.py`（新增 `fuse_hadamard_quant` 開和關兩種，以及 LN + query quant 合併 launch 的比對）、`test_triton_hadamard_quant.py`（新增 strided weights）、`test_dsa_kpool_multi_pool.py` 都過。新增的 `fuse_hadamard_quant=True` case 在舊的 code 上會 fail。

### 16.4 長 context 準確度：needle

`tools/needle_idxfix_1112.sh`（node 152 GPU 4–7，`jacchang_base1112`，server 參數和 env 跟 AgentX 一樣）：4000 組 key → 6 位數 code 的表，約 41.8K token，問 20 個 key，thinking 關（工具送 `enable_thinking: False`，`max_tokens` 1024），temperature 0。

| sglang 樹 | 4000 組（約 41.8K token） | 20 題花的時間 | 12000 組（約 125.4K token） | 20 題花的時間 |
|---|---|---|---|---|
| kdafly（有 bug） | 2/20 | 61 s | 0/20 | 87 s |
| idxfix | 20/20 | 18 s | 20/20 | 25 s |

- 125K 那組在 node 152 GPU 0–3（`jacchang_pr48875`）跑，結果在 `/shared_nfs/jacchang/needle1112_long/`。

有 bug 的版本常常回答相鄰或不存在的 key（例如問 K28766 回答 K28765），回答也長很多。也就是說，之前所有開了 `SGLANG_DSA_FUSE_HADAMARD_QUANT=1` 的 GLM-5.3-Flash run，在長 context 下幾乎找不到資料。

### 16.5 trace（ISL 128K，MTP5 golden AL，ReplaySSM 開，同一個 container 和 GPU 4–7）

| 2 個 cycle 平均 | bs6 kdafly | bs6 idxfix | bs16 kdafly | bs16 idxfix | B200 bs6 / bs16 |
|---|---|---|---|---|---|
| DSA+MoE 層的 wall（µs） | 310.8 | 289.1 | 421.1 | 384.9 | |
| KDA+MoE 層的 wall（µs） | 167.9 | 160.1 | 237.2 | 215.7 | |
| MoE gemm1 + gemm2（µs，KDA+MoE 層） | 72.0 | 64.1 | 114.7 | 94.0 | |
| 45 層（ms） | 8.97 | 8.48 | 12.40 | 11.33 | 9.58 / 12.0（kernel 加總） |
| cycle wall（ms） | 12.42 | 11.57 | 16.50 | 15.54 | 11.02 / 14.3 |

- indexer 那段鏈：每個 DSA 層 24.6 µs（5 個 kernel）變成 11.2 µs（1 個 kernel），跟 microbench 一致。
- 大部分的進步來自 MoE：修好之後 attention 是對的，hidden state 不再是雜訊，router 選的 expert 比較集中，MoE 要讀的 weight 變少。bs16 每個 MoE 層少約 20 µs，42 層約 0.85 ms。
- B200 本來就沒有這個 bug（fused x projection 只在 gfx95 開），所以第 14.3 節「MoE 已經接近 HBM 頻寬」和之前跟 B200 比的 MoE 差距，有一部分是這個 bug 造成的。
- cycle 跟 B200 的差距：bs6 從 8.5%（第 14.1 節的 11.96 ms）變 5.0%，bs16 從 16.6%（16.68 ms）變 8.7%。這次 kdafly 的 bs6 cycle 是 12.42 ms，比第 14.1 節多出來的部分都在 tail，45 層的時間一樣。
- tail 的 idle 在不同 run 之間會在 0.17–0.69 ms 之間跳（只有 2 個 cycle，CPU 排程的雜訊），跟改動無關。比較時看 45 層和 tail 的 busy：tail busy 是 kdafly 3.89 ms、idxfix 3.49 ms（MTP draft 的 MoE 也一起變快）。

### 16.6 AgentX（ISL 約 128K，MTP5 golden AL，ReplaySSM 開，1 小時）

| P90 Int | idxfix（修好） | kdafly（有 bug） | kda8（有 bug） | 同一天 base（有 bug） | B200 |
|---|---|---|---|---|---|
| conc16 | **242.6**（GPU 0–3）/ 246.3（GPU 4–7） | 250.7 | 231.4 | 222.1 | 256.8 |
| conc32 | **150.0**（GPU 4–7） | 159.1 | 147.1 | 146.2 | 164.4 |

- 修好之後 AgentX 反而比有 bug 的 kdafly 慢：conc16 −3.3%（同樣 GPU 0–3、同一個 container），conc32 −5.7%。兩次 conc16 差 1.5%，是 GPU 組和 run 之間的雜訊。
- 決定 P90 的 P85–P95 那群 request，step 的時間 conc16 3.49 vs 3.31 ms，conc32 4.93 vs 4.68 ms，都多約 5%；stall 差不多。
- 每個 decode batch 的 ms/token（p50，conc32）：bs 1–2 一樣（2.42 vs 2.39），bs 5–8 多 4.6%，bs 13–16 多 6.6%。差距跟著 batch 變大，比較像 MoE：有 bug 時 hidden state 是錯的，在真的文字上 router 可能擠到比較少的 expert，要讀的 weight 比較少。跟第 16.5 節（合成的 prompt，修好之後 MoE 比較快）方向相反，還沒驗證。
- **base 也有這個 bug**：base 的樹 `wt-glm53-1106-prstack` 有 `62d2711d65`，`common.sh` 也開了 `SGLANG_DSA_FUSE_HADAMARD_QUANT=1`。所以 1106 PR stack 之後 MI355X 所有的 AgentX 數字（11/10 CSV 的 224.8 / 148.4、base、kda8、kdafly）都是 indexer 錯的狀態。B200 沒有這個 bug，所以真正能跟 B200 比的只有 idxfix：conc16 差 5.5%（242.6 vs 256.8），conc32 差 8.8%（150.0 vs 164.4）。
- 正確的 base 在 11/13 跑了，見第 16.9 節。
- 結果在 `/shared_nfs/jacchang/agentic_1112b/{idx,idx0,opt}/`。kdafly conc16 的 `profile_export.jsonl` 第 628 行有 NUL，分析用的是 `/shared_nfs/jacchang/idx1112/bench-Agentic-kdafly-topkv2-c16-filtered/`（原檔沒動）。

### 16.7 router GEMV + top-k 合成一個 launch：不做

想法：verify 只有一條 stream，router 的 split-K GEMV（144 個 CTA）寫完 partial 後，用 grid-wide counter 等全部寫完，再由前幾個 CTA 做 top-k，每個 MoE 層少一次 launch（每個 cycle 42 層）。跨 XCD 的 L2 不 coherent，所以 partial 用 write-through store（`sc0 sc1`），counter 用 release / acquire（ISA 上是 `buffer_wbl2 sc1` 和 `buffer_inv sc1`）。停損條件：36 和 96 rows 都要比現在快 1.5 µs 以上。

`tools/bench_router_fused_1112.py`（node 152 GPU 0，CUDA graph，沒開 profiler，ids 和 weight 都跟現在的路徑一致）：

| µs | 6 rows | 16 rows | 36 rows | 96 rows |
|---|---|---|---|---|
| 現在（GEMV + `moe_fused_gate`） | 7.56 | 7.53 | 8.75 | 10.54 |
| GEMV + 簡化的 route kernel（1 row / 1 warp） | 6.72 | 6.73 | 7.97 | 9.84 |
| 合成一個 launch（最好的設定） | 8.87 | 9.39 | 11.46 | 15.49 |

- 合成一個 launch 在所有 row 數和設定（每個 CTA 1/2/4 rows，1/4 warps，打包 key）都比較慢。在 CUDA graph 裡兩個小 kernel 的 launch 本來就很便宜，grid barrier、write-through store 和 `.cv` load 的代價比省下的 launch 大。
- 把 max 和最小 id 打包成一個 int64 key（每步只做一次 reduction）也比較慢。
- 簡化的 route kernel 只快 0.7–0.8 µs，每個 cycle 約 34 µs（0.2%），不值得改共用的 `moe_fused_gate`。
- 完整的表在 `/shared_nfs/jacchang/idx1112/router_fused_bench.txt`。
- trace 上這兩個 kernel 是 4.8 + 6.2 = 11.0 µs，microbench 只有 7.5 µs。差別可能來自 profiler 的開銷，或 microbench 的 router weight（2.4 MB）一直留在 cache 裡；不管哪個，用 trace 上的 kernel 時間估「少一次 launch 省多少」都會高估。

### 16.8 第 3 點其他的候選：看過，沒做

| 候選 | 沒做的原因 |
|---|---|
| `set_mla_kv_buffer` 併進 indexer 那次 launch | 寫 MLA KV cache 在 attention backend 裡，indexer 之後才呼叫；要把它提前，得改 model、indexer 和 backend 之間的介面，每個 cycle 只省 11 次 launch（約 20 µs） |
| `layer_norm_gated` 併進 KDA verify kernel | V tile 64 時一個 head 分在不同 CTA（不同 XCD），要跨 CTA 做 norm；改成 V tile 128 state register 加倍 |
| activation quant 併進 `big_fuse` | 要改 opus 的 C++ kernel，工作量大，每層只省一次小 kernel |
| weight prefetch（在 allreduce 等 XGMI 的時候先把下一層的 weight 拉進 MALL） | 要在 aiter custom AR 裡加 prefetch，而且要先證明 MALL 命中能讓下一個 GEMM 變快；還沒有實驗數據，風險高 |

### 16.9 11/13：AgentX，修好的 idxfix vs 正確的 base（同一天、同一個 node）

node 152，兩條 lane 同時跑：idxfix 在 GPU 0–3（`tools/agentx_opt1112c.sh`），base 在 GPU 4–7（`tools/agentx_base1112c.sh`：base 的樹 `09111fe9a3` 加 `SGLANG_DSA_FUSE_HADAMARD_QUANT=0`，所以 indexer 是對的，Hadamard 和 quant 不合併）。ISL 約 128K，MTP5 golden AL，ReplaySSM 開，1 小時。

| P90 Int | idxfix | 正確的 base | 提升 | B200 | 跟 B200 差 |
|---|---|---|---|---|---|
| conc16 | **248.8** | 207.1 | +20.1% | 256.8 | 3.1% |
| conc32 | **152.1** | 137.8 | +10.4% | 164.4 | 7.5% |

- 決定 P90 的 P85–P95 那群 request，step 的時間 conc16 3.50 vs 4.16 ms，conc32 4.90 vs 5.49 ms；stall 差不多（conc32 1.76 vs 1.83 ms）。conc32 的 stall 佔 26%，是剩下差距的主要來源。
- 每個 decode batch 的 ms/token（p50）：bs 1–2 少 9–10%，bs 5–8 少 13–14%，bs 9–12 少 3–10%。
- 跟 11/12 的 idxfix 比：conc16 242.6 → 248.8，conc32 150.0 → 152.1，是 GPU 組和 run 之間的雜訊範圍。
- base 的 Hadamard 和 quant 是分開的 kernel，所以這 20% 裡有一部分是 #44661 的合併和 indexer 5 個 kernel 併成 1 個（第 16.3 節），不是只有第 14、16 節的優化。
- 結果在 `/shared_nfs/jacchang/agentic_1112b/idx1113/` 和 `/shared_nfs/jacchang/agentic_1112c/bhq0/`，分析在 `/shared_nfs/jacchang/idx1112/agentic_tail_1113.txt`。

## 17. 11/13 總整理：11/10–11/13 所有的優化、對 P90 Int 的貢獻、要開幾個 PR

撰寫：2026-11-13 15:10（台灣時間）。conc64 正確 base 的結果 16:58 補上。

### 17.1 AgentX 總表（ISL 約 128K，golden AL，1 小時）

| P90 Int | CSV stock 1106 | CSV pr1106（有 bug） | 正確的 base（16.9） | 11/13 FP32 state | 11/13 BF16 state | B200 |
|---|---|---|---|---|---|---|
| conc1（mono，MTP5） | 430.6 | 449.2 | | **448.4**（node 153；node 247 也是 448.4） | **463.7**（+3.4%，node 247，mono 改 BF16，第 17.6 節） | 421.9 |
| conc4（mono，MTP3） | 403.1 | 421.6 | | **427.4**（node 247：413.1） | **446.5**（+8.1%，跟 node 247 的 413.1 比） | 393.5 |
| conc8（mono，MTP3） | 293.9 | 321.4 | | **323.6**（node 247：326.7） | **346.0**（+5.9%，跟 node 247 的 326.7 比） | 324.6 |
| conc16（MTP5，ReplaySSM） | 158.8 | 224.8 | 207.1 | **246.9**（上午 248.8） | 243.9（−1.2%） | 256.8 |
| conc32（MTP5，ReplaySSM） | 91.4 | 148.4 | 137.8 | **153.4**（上午 152.1） | **158.0**（+3.0%） | 164.4 |
| conc64（MTP5，ReplaySSM，HiCache 216） | 41.0 | 64.7 | 64.3（node 153） | **66.8**（+3.8%） | **70.9**（+6.1%） | 59.4 |

- 11/13 FP32 / BF16 是 node 152 同時跑（GPU 0–3 是 `d4220d038e`，GPU 4–7 是 `508c4cc3a3`）。conc4 / 8 是 node 153 的 `jacchang_mono1113`，mono 是 `25723f1`。
- CSV pr1106 和 11/10 之後的 base 都有第 16.2 節的 bug。能跟 B200 比、能算「我們的優化」的，只有跟正確的 base 比。
- conc1–8 的 target 步驟大多在 monokernel 裡，FP32 state 那一欄沒動 monokernel，所以跟 pr1106 差不多（第 17.5 節）。BF16 state 那一欄的 monokernel 改成吃 BF16 state（第 17.6 節），conc1–8 都超過 B200：+9.9% / +13.5% / +6.6%。
- 跟 B200 的差距：conc16 3.9%，conc32 6.7%（BF16 state 3.9%）。conc64 已經超過 B200：FP32 +12.5%，BF16 +19.4%。
- conc64 的其他指標（FP32 / BF16 / CSV pr1106 / B200）：P50 Int 108.0 / 113.3 / 103.6 / 120.5，TTFT p90 1.10 / 0.98 / 1.06 / 2.60 s，Throughput/GPU 94825 / 96378 / 93504 / 92890，prefix cache hit 96.3% / 96.6%，各 3 個空回應 error。
- conc64 正確 base（`/shared_nfs/jacchang/agentic_1112c/bhq0c64/`）：P50 Int 99.5，TTFT p90 1.02 s，Throughput/GPU 91825，profiled 9830 個 request。dropped 的數量跟 FP32 / BF16 一樣（warmup 707、InvalidInferenceResultError 約 512），結果可以用。

### 17.2 每個優化估計能加多少 P90 Int

- 方法：每項用 bs16 trace 上 45 層省下的時間，依比例分掉 conc16 實測的 +20.1% 和 conc32 的 +10.4%（idxfix 對正確的 base，第 16.9 節）。
- 這是估計。C、D、E、G 是 4 µs 左右的小 kernel，trace 會高估（第 10.10 節），實際貢獻大概只有表上的一半，多出來的部分應該歸給大 kernel。
- 11/16 發現的問題：conc16 那一欄也是用 bs16 trace 分的，但 conc16 的 P90 request 跑在 decode batch 4–9（中心 bs6，第 1 節），bs16 只代表 conc32。conc16 那一欄應該改用 bs6 trace（第 11 節的逐 kernel 表有 bs6 的數字）重新分配，還沒做。只在大 batch 才有效的項目（例如 in-proj 在 M=96 的部分）在 conc16 可能被高估。

| # | 優化 | 節 | bs16 每個 cycle 省（trace ms） | conc16 | conc32 | 準確度檢查 |
|---|---|---|---|---|---|---|
| 1 | FlyDSL KDA verify kernel（取代 J 和原本的 conv1d + delta rule） | 13 | 0.70 | +6.2% | +3.2% | GSM8K 96.4 / 96.8%，ring bitwise |
| 2 | MoE router：aiter 單 group 走 Triton router、padded row、split-K GEMV（H、H'、I） | 12 | 0.30 | +2.7% | +1.4% | GSM8K 97.0%，選的 expert 跟 fp32 一樣 |
| 3 | k-pool topk v2 在 HIP 上開 | 14.2 | 0.27 | +2.4% | +1.2% | GSM8K 97.1%，index 集合一樣 |
| 4 | KDA in-proj / o_proj 重 tune（A），tgemm 有 tuned row 就不走 Triton（B） | 11 | 0.23 | +2.0% | +1.1% | GSM8K 96.7% |
| 5 | indexer logits `next_n=6` layout（F） | 11 | 0.21 | +1.9% | +1.0% | 同上 |
| 6 | KDA / GDN verify 不再每層 `compare_scalar`（E） | 11 | 0.15 | +1.3% | +0.7% | 同上 |
| 7 | indexer verify 的 key LN、k-pool 寫入、query quant 合成一個 launch | 16.3 | 0.15 | +1.3% | +0.7% | needle 20/20（32K / 125K） |
| 8 | aiter MoE sort BM16 crossover（C）。**11/16 重估後不開 PR**（第 17.2.1 節） | 11 | 0.14（trace 高估約 4 倍） | 約 +0.15%（重估） | 約 +0.2%（重估） | 同 4 |
| 9 | mHC post + pre 合併的上限 16 → 192 token（G） | 11 | 0.07 | +0.6% | +0.3% | 同 4 |
| 10 | DSA q_b 前的 RMSNorm + quant 回歸修正（D） | 11 | 0.05 | +0.4% | +0.2% | 同 4 |
| | 合計（實測） | | 2.27 | **+20.1%** | **+10.4%** | |
| 11 | double Hadamard 修正 | 16.2 | 不算速度 | | | needle 0/20 → 20/20 |
| 12 | FlyDSL KDA verify 讀 BF16 state pool（接 #49036） | 17.4 | 0.06 | −1.2%（實測） | +3.0%（實測） | needle 20/20、GSM8K、bitwise |
| 13 | monokernel 吃 BF16 state（sgl-amd-mono，接 #12） | 17.6 | 不在這張 trace 上 | conc1 / 4 / 8：+3.4% / +8.1% / +5.9%（實測，含 sglang 那邊的 BF16） | | bitwise、needle 20/20、GSM8K |

- conc32 的收益比較少，是因為決定 P90 的 request 有 26% 的時間卡在 prefill（第 16.9 節），這些優化只動 decode。
- conc64（實測）：正確 base 64.3，FP32 state 66.8（+3.8%），BF16 state 70.9（+10.3%）。FP32 比預估的 +5–10% 低，可能是 conc64 的 P90 request 被 prefill stall 卡得更多（還沒有拆 tail request 驗證）。BF16 state 在 conc64 多拿 +6.1%，是 #12 收益最大的 conc。base 是在 node 153 GPU 0–3 跑的（`jacchang_mono1113`，sort patch 先還原，跟第 16.9 節一樣），FP32 / BF16 在 node 152，node 之間有約 1–3% 的雜訊。
- 看過、沒有效果或停損的：MoE exact-M tune（14.3）、allreduce + mHC 合併（16.1）、router GEMV + top-k 合成一個 launch（16.7）、gated norm 併進 KDA kernel（12.5、16.8）。這些都不開 PR。
- #8 重估之後，表上各項加起來比實測總數少約 1.0% / 0.4%。實測總數不變，這部分應該分回其他項，還沒重新分配。

#### 17.2.1 #8（MoE sort crossover）重新確認（11/16）

結論：kernel 本身快 7–19%，但對 P90 Int 的影響約 0.15–0.2%，在 node 之間 1–3% 的雜訊以下。不開 PR，branch `jacob/moe-bm16-sort-crossover` 留著當參考。

microbench（node 153 GPU 7，`tools/bench_moe_sort_1111.py`，HIP graph，fused 和 Opus 的 routing 結果完全相同）。舊規則在 43 rows 以上走 Opus，新規則到 154 rows 都走 fused：

| rows | 6–36 | 48 | 60 | 72 | 96 | 120 | 144 | 156 以上 |
|---|---|---|---|---|---|---|---|---|
| fused（µs） | 3.2–4.4 | 4.75 | 5.20 | 5.28 | 5.74 | 6.12 | 6.48 | 6.65–6.85 |
| Opus（µs） | 6.0–9.3 | 6.21 | 6.48 | 6.34 | 6.44 | 6.67 | 6.59 | 6.61–6.90 |
| 新規則省多少 | 0（兩邊都走 fused） | 1.46 | 1.28 | 1.06 | 0.70 | 0.55 | 0.11 | 0（兩邊都走 Opus） |

換算成 P90 Int：rows = 6 × decode batch，decode batch 依 CUDA graph 補到 capture size（1–8，之後 10–32 的偶數），每個 decode step 有 42 個 MoE 層。

| | P90 request 的 decode batch（第 1 節） | 補到 capture size 後的 rows | 每個 step 平均省 | cycle（trace） | 佔比 |
|---|---|---|---|---|---|
| conc16 | 4–9 | 24–60，只有 48、60 有省 | 約 19 µs | bs6 約 12.4 ms | 約 0.15% |
| conc32 | 11–20 | 72–120 | 約 32 µs | bs16 約 16.9 ms | 約 0.2% |

- 第 17.2 節原本的 +1.2% / +0.6% 是用 trace 的 8.7 → 5.4 µs 算的。Opus 是兩個短 kernel，trace 把每個都撐到約 4.2 µs（第 10.10 節），所以高估了約 4 倍。
- 所有 decode step 的 decode batch 分布（11/13 BF16 state run 的 server log，每一行 `Decode batch` 約代表同樣多的 step）：conc16 有 98% 的 step 在 decode batch 1–6（1：31%、2：29%、3：20%、4：11%、5：5%、6：2%）；conc32 是 1–8 佔 86%、10–12 佔 10%；conc64 有 84% 在 8 以上。tail request 跑在比平均更大的 decode batch，所以 P90 要用第 1 節的範圍，不能用這個平均分布。

### 17.3 要開幾個 PR

依子系統合成 5 個：sglang 3 個、aiter 1 個、sgl-amd-mono 1 個（E，11/13 晚上加的，第 17.6 節）。每個 PR 只碰一個子系統，各自找一個 reviewer。都要建在 1106 PR stack 上，**不含 #44661**（9 個 open PR，`wt-glm53-1117-prstack` `6fd8d4b50a`；跟第 17.8 節的 `309b924c75` 內容一樣）。

| PR | repo | 內容（第 17.2 節的 #） | 改動行數 | conc16 / conc32 | 依賴 |
|---|---|---|---|---|---|
| A：KDA verify on gfx950（[sgl-project/sglang#51126](https://github.com/sgl-project/sglang/pull/51126)，11/17 開 draft（#50880），11/18 重開成連號，建在 main `6c9a1938d0` 上） | sglang | FlyDSL KDA verify kernel 和 dispatch（#1）、拿掉 `compare_scalar`（#6）、讀 BF16 state（#12）、單元測試 | 約 870 | +7.5% / +3.9% | ReplaySSM；#12 要等 #49036 |
| B：DSA indexer on gfx950（[sgl-project/sglang#51127](https://github.com/sgl-project/sglang/pull/51127)，11/17 開 draft（#50936），11/18 重開成連號，疊在 #44417 上，第 17.8 節） | sglang | `next_n=6` logits（#5）、k-pool topk v2 預設開（#3）、indexer verify 合成一個 launch（#7）、q_b quant 回歸修正（#10）、單元測試。#44661 拿掉後不需要 double Hadamard 修正（#11） | +446 / −26 | +6.0% / +3.1% | #44417 |
| C：MoE router 和 GEMM dispatch on HIP（[sgl-project/sglang#51128](https://github.com/sgl-project/sglang/pull/51128)，11/17 開 draft（#51003），11/18 重開成連號，建在 main `9158db2b97` 上，第 17.9 節） | sglang | Triton router、padded row、split-K GEMV（#2）、tgemm 有 tuned row 就用（#4 的 B）、mHC 上限（#9） | 約 190 | +3.3% / +1.7% | 無 |
| D：GLM-5.3-Flash TP4 on gfx950（[ROCm/aiter#8006](https://github.com/ROCm/aiter/pull/8006)，11/16 開，同天轉 ready for review） | aiter | BF16 GEMM tuned CSV（#4 的 A）：KDA in-proj / o_proj 補 6 的倍數的 M，in-proj 在 M=84–128 從 torch native 換成 FlyDSL hgemm。#8 不放（第 17.2.1 節） | 只改 CSV | +2.0% / +1.1% | 無 |
| E：glm53_flash 吃 BF16 KDA state（[sgl-project/sgl-amd-mono#141](https://github.com/sgl-project/sgl-amd-mono/pull/141)，11/16 開 draft） | sgl-amd-mono | monokernel 的 BF16 state（#13）、`check_layer.py --check-state-bf16` | +193 / −48 | conc1 / 4 / 8：+3.4% / +8.1% / +5.9% | 要開 BF16 state 才有用：#49036 和 A 的 #12 |

- A–D 加起來 +18.8% / +9.8%，比實測的 +20.1% / +10.4% 少，是因為 #8 重估後拿掉了，多出來的部分還沒分回各項（第 17.2 節）。#4 的 o_proj 部分要 C 和 D 都 merge 才有效，表上把 #4 算在 D。
- D 的 tuned CSV 在 kernel 層級（node 152，同一個 process 裡交替跑新舊 CSV，cold weight，兩張 GPU 平均）：有改到的 M，in-proj 快 4–26%（M=96 從 19.50 到 17.21 µs），o_proj 快 4–14%，沒有任何 M 變慢；M ≤ 16 和 M > 128 查到的 row 完全一樣。代價是 split-K 讓相對 fp32 的誤差從 1.66e-3 變成 2.45e-3–3.15e-3，跟 CSV 裡原本的 split-K 4 row 同一個水準。o_proj 在 M ≤ 64 要加上 C 的 #4 B 才走得到 tuned kernel。工具：`tools/bench_kda_proj_ab_inproc_1116.py`，結果在 `/shared_nfs/jacchang/kdaproj1116/cold_gpu{0,1}.csv`。
- B 不放 #44661（11/17 改）。#44661 的 `fused_hadamard_act_quant` 會再轉一次 #44417 已經轉過的 query，造成 double Hadamard，所以從 stack 拿掉，env `SGLANG_DSA_FUSE_HADAMARD_QUANT` 也不要設。沒有 #44661，double Hadamard 修正（#11）就不需要了。合成一個 launch（#7）改用 `kpool_fp8_index.py` 本來就有的 `_hadamard_quantize_fp8`，不用 #44661 的 code，結果跟 baseline 的 `act_quant(rotate_activation(q))` bit 完全一樣。B 只依賴 #44417。
- topk v2 預設開放在 B：`model_hook` 的 HIP 分支裡，`index_kpool > 1` 的 model 預設開，不用新 env，明確設定的 `SGLANG_OPT_USE_TOPK_V2` 還是優先。
- 順序：A、C、D 互不依賴，可以同時開；B 疊在 #44417 上。
- A 最大，大部分是新的 kernel（`kernel.py` 455 行、`__init__.py` 174 行），對既有 code 的改動不多。如果 reviewer 覺得太大，可以把 #6 和 #12 拆出去，A 剩約 760 行。

### 17.4 FlyDSL KDA verify 讀 BF16 state（接 #49036）

- branch `jacob/glm53-flash-1113-kdabf16`（`~/PR/wt-glm53-1113-kdabf16`，11/13 已 push 到 Jacob0226/sglang）：`476dc821f0` 是 cherry-pick 的 #49036，`508c4cc3a3` 是我們的改動。開 `--mamba-ssm-dtype bfloat16` 時，verify 讀 state 時轉成 fp32，計算還是 fp32。verify 不寫 state，ReplaySSM 的 fold kernel 原本就跟 dtype 無關。
- 單元測試新加 `test_gfx950_ring_bf16_state_matches_fp32`：輸出的有效 row、`win` 和 4 個 ring 都跟 FP32 state pool 放同一組 BF16 值時的結果 bit 一樣。needle 32K / 125K 20/20，GSM8K 沒掉，trace 確認 server 走到新的 kernel。

| verify microbench（µs） | bs1 | bs6 | bs8 | bs16 | bs32 | bs64 |
|---|---|---|---|---|---|---|
| FP32 state | 5.96 | 7.53 | 8.37 | 12.85 | 20.45 | 32.85 |
| BF16 state | 5.66 | 7.25 | 7.58 | 10.99 | 17.56 | 28.29 |

- #49036 的 PR body 寫 conc64 +25%，主要是因為它的 base verify kernel 在 64 個 request 要 93.45 µs（BF16 42.95 µs）。我們的 FlyDSL verify 用 FP32 state 就只要 32.85 µs，所以這部分的收益大多已經拿到了。conc64 的 state slot 也沒有用完（11/10 的 conc64：1353 個 slot，usage 最高 22%），slot 變多不是收益來源。
- 結果在 `/shared_nfs/jacchang/kdabf16_1113/`、`/shared_nfs/jacchang/idx1112/kda_bf16_state_bench_1113.txt`、`/shared_nfs/jacchang/agentic_1113/`。

### 17.5 conc4 / conc8：monokernel 跟 pr1106 比

| | conc4 這次 | conc4 pr1106 | conc8 這次 | conc8 pr1106 |
|---|---|---|---|---|
| P90 Int | 427.4 | 421.6 | 323.6 | 321.4 |
| P50 Int | 578.0 | 572.1 | 476.2 | 473.7 |
| TTFT p90（s） | 0.37 | 0.37 | 0.39 | 0.38 |
| Throughput/GPU | 9830 | 9744 | 16259 | 16012 |

- monokernel 只處理 target 不超過 16 rows 的步驟。MTP draft、prefill 和 conc8 超過 16 rows 的 verify 走一般的 SGLang forward，這些地方有第 17.2 節的改動，所以有可能稍微快一點。不過 +0.7–1.4% 還在單次 run 的雜訊裡。
- conc1（MTP5，bs ≤ 2 整步都在 mono 裡）：node 153 和 node 247 都是 448.4，pr1106 是 449.2，一樣。P50 Int 485.4，TTFT p90 0.35 s，Throughput/GPU 6866，2 個空回應 error。
- 第一次在新容器跑的時候 server abort：image 內建的 `module_moe_mxfp4_aux.so` 沒有因為 aiter patch 重新 build，缺 `aux_scatter_H4096_TOPK9_NT1`。換成 `/shared_nfs/jacchang/module_moe_mxfp4_aux_sort1113.so` 之後就正常了。

### 17.6 monokernel 吃 BF16 KDA state（sgl-amd-mono）

撰寫：2026-11-13 17:00（台灣時間）。

- 改之前：開 `--mamba-ssm-dtype bfloat16` 時，`glm53_flash/spec.py` 回報「needs the KDA state in f32」，monokernel 整個不開。monokernel 和 BF16 state 只能二選一。
- branch `jacob/glm53-flash-mono-bf16state-1113`（`/shared_nfs/jacchang/exp/sgl-amd-mono-bf16state-1113`，base `25723f1`，commit `97a0a5d`），已經 push 到 `Jacob0226/sgl-amd-mono`，PR 下週開（第 17.3 節的 E）。5 個檔案，+193 / −48。

| 檔案 | 改動 |
|---|---|
| `gfx950/kernel.py` | `FlashBuild.state_bf16`：decode 讀 BF16 state、轉 FP32 算、寫回時 round to nearest even；verify 整串 draft 用 FP32 carry 在暫存器裡算，每個 draft token 的 verify scratch 寫成 BF16。計算一律 FP32 |
| `spec.py` | state dtype 接受 FP32 或 BF16（conv 還是 BF16），依 `mamba2_state_dtype` build 對應的版本；verify scratch 的 dtype 要跟 state 一樣 |
| `gfx950/runner.py` | launch 時檢查 state 和 verify scratch 的 dtype 跟 build 的版本一致 |
| `test/kernels/glm53_flash/check_layer.py` | `--check-state-bf16`（加 `--verify L` 測 verify） |
| `test/unit/test_glm53_flash_spec.py` | BF16 state 不再被擋，FP16 還是擋 |

| 驗證關卡 | 結果 |
|---|---|
| 1. bitwise：BF16 pool 對放同一組 BF16 值的 FP32 pool | 12 組全部相同：decode 2 / 4 / 8 / 12 / 16 rows，verify MTP5 8 / 12 / 16 rows、MTP3 4 / 8 / 12 / 16 rows。輸出、conv state、寫回的 state（FP32 round 成 BF16 後比）、verify scratch |
| 2. FP32 不變 | 跟 `25723f1` bitwise 相同（decode 2 / 8 / 12 / 16，verify 8×MTP5、12×MTP5、8×MTP3、16×MTP3，4 個 rank） |
| 3. server log | `monokernel glm53_flash_hip on gfx950 runs target_verify steps of batch size [1, 2] x 6 tokens`（MTP5）、`[1, 2, 3, 4] x 4 tokens`（MTP3） |
| 4. needle 32K / 125K | MTP5、MTP3 都是 20/20、20/20 |
| 5. GSM8K（sgl-eval，thinking，temp 1.0，4096 tokens，MTP3，4 threads，全部走 mono） | 見下表 |
| lint、unit test | ruff 0.15.1 / isort 7.0.0 乾淨；`test_glm53_flash_*` 29 passed、3 skipped |

| GSM8K | seed 0 | seed 1 | seed 2 | 平均 | accept length 平均 |
|---|---|---|---|---|---|
| FP32 state（mono `25723f1`） | 96.51% | 97.12% | 97.42% | 97.02% | 3.179 |
| BF16 state（mono `97a0a5d`） | 97.04% | 96.74% | 97.27% | 97.02% | 3.169（−0.3%） |

- seed 0 的差 +0.53 點超過 0.5 點的門檻，所以加跑 seed 1、2。三個 seed 的差 +0.53 / −0.38 / −0.15 點，方向不一致，平均一樣，是 temperature 1.0 的雜訊。

AgentX（node 247，GPU 0–3 FP32 state：sglang `d4220d038e` + mono `25723f1`；GPU 4–7 BF16 state：sglang `508c4cc3a3` + mono `97a0a5d`；兩條 lane 同時跑，每個 conc 1 小時，golden AL；P90 用 adj ITL p90，跟第 17.1 節一樣）：

| conc | FP32 P90 Int | BF16 P90 Int | 差 | B200 | CSV pr1106 | P50 Int（FP32 / BF16） | TTFT p90 s（FP32 / BF16） |
|---|---|---|---|---|---|---|---|
| 1（MTP5） | 448.4 | **463.7** | **+3.4%** | 421.9 | 449.2 | 486.9 / 503.6 | 0.37 / 0.37 |
| 4（MTP3） | 413.1 | **446.5** | **+8.1%** | 393.5 | 421.6 | 575.0 / 598.8 | 0.35 / 0.35 |
| 8（MTP3） | 326.7 | **346.0** | **+5.9%** | 324.6 | 321.4 | 479.2 / 505.3 | 0.38 / 0.39 |

- 兩邊的 decode batch 分布幾乎一樣（conc8：bs 1–2 佔 89.6% / 91.2%），差距不是 batch 大小造成的。去掉 prefill stall 的 P90 Int 也是同樣的差距（conc4 431.2 → 470.1，conc8 348.8 → 371.0）。
- 收益來源：每層每個 request 的 KDA state 在 TP4 是 16 heads × 128 × 128，FP32 1 MiB、BF16 0.5 MiB。decode 每步讀寫一次，MTP3 的 verify 讀一次、寫 4 個 draft token 的 scratch。decode batch 小的時候這些記憶體流量佔 step 時間的比例高。state slot 也從 1477 個變成 2860 個（MTP3），這點對 P90 有沒有貢獻沒有拆開驗證。
- 這些數字**不全是 monokernel 的功勞**：BF16 lane 的 sglang（`508c4cc3a3`）讓 prefill、MTP draft 和超過 16 rows 的 verify 也吃 BF16 state。要拆開的話，要多跑一組「BF16 state、不開 monokernel」。
- conc4 的 FP32 這次是 413.1，比 node 153 的 427.4 低 3.3%。同一台 node、同時跑的兩條 lane 互相比才準。
- 結果在 `/shared_nfs/jacchang/agentic_1113_mono/`（`fp32/`、`bf16/`），驗證在 `/shared_nfs/jacchang/mono_bf16_gate_1113/`、`/shared_nfs/jacchang/mono_bf16_needle_1113/`。工具：`tools/mono_bf16_gate_1113.sh`、`tools/mono_fp32_dump_1113.py`、`tools/mono_needle_bf16_1113.sh`、`tools/agentx_mono_lane_1113.sh`。
- 第一次跑的時候，FP32 lane 和 BF16 needle 都遇到第 17.5 節的 `module_moe_mxfp4_aux.so` abort（容器重建時漏換）。換掉之後重跑，上面的數字都是重跑的結果；失敗的 log 移到 `failed_no_aux_so/`。

### 17.7 11/16：新 image 上 BF16 state 的 on / off（[sgl-project/sgl-amd-mono#141](https://github.com/sgl-project/sgl-amd-mono/pull/141)）

撰寫：2026-11-16 21:30（台灣時間）。

#### 17.7.0 AgentX 每個 conc 的 MTP 設定（11/18 補，不要再弄錯）

| conc | MTP | monokernel |
|---|---|---|
| 1 | MTP5 | 開 |
| 4、8 | MTP3 | 開（MTP3 一個 request 4 rows，4 個 request 剛好 16 rows） |
| 16、32、64 | **MTP5** | 不開 |

- control point 要用 conc16 以上的設定，也就是 **MTP5**。11/18 我用 conc16 **MTP3** 跑 control，是錯的。
- 跑任何 AgentX 之前先對這張表。

#### 17.7.1 結論

- **P90 Int**：conc1 +3.8%（一輪），conc4 −0.5%（兩輪平均，−3.7% / +2.7%），conc8 +2.8%（−0.1% / +5.9%）。同一個設定跑兩輪就差 1.8–8.5%，跟增益一樣大，所以 conc4 / 8 的 P90 Int 這次看不出確定的增益。
- **decode step 本身每個點都變快**：server log 裡 decode batch 1–2 的每 token 時間（p50）在 5 組裡都少 2.8–4.1%，decode batch 3–4 少 2.6–4.9%。P50 Int 5 組全部上升（+1.4% 到 +8.3%，conc8 兩輪平均 +7.8%）。
- **P90 沒跟著動的原因**：conc8 設定 P90 的 request（TPOT 的 P85–P95）有 15–17% 的時間在等 prefill（stall）。扣掉 stall 後，on 比 off 高 +5.5% / +9.0%。這個 image 的 TTFT 比之前長（TTFT p90 0.62–0.73 s；1106 曲線 0.42–0.49 s；週五 0.35–0.37 s），stall 稀釋了 decode 的收益。（11/19 更正：不是 prefill 變慢，是新的預設 `--hybrid-cache-checkpoints-level 2` 讓重算的 token 多了 62%，見第 17.10 節。）
- **Throughput/GPU**：conc1 +2.0%，conc4 +2.1%（兩輪平均），conc8 −0.6%。conc8 的 throughput 幾乎都是 input token，主要看 prefill，所以 decode 變快反映不出來。
- **準確度**：GSM8K 3 個 seed 平均 97.09% → 97.17%。RULER2 NIAH 300K 88.81% → 87.12%，640K（每個 task 10 題）80.42% → 77.54%。兩邊是同一組題目，答對答錯的翻轉兩邊都有（300K 4 比 2，640K 3 比 2），`mk_niah_basic` / `mv_niah_basic` 兩個長度兩邊都是滿分。差距集中在 `mv_niah_hard`，而且兩個長度都是 on 比較低，題數太少分不出是不是雜訊，要再跑（第 17.7.4 節）。
- **要注意的一點**：真實 acceptance 下，BF16 state 的 accept length 比 FP32 低：GSM8K 3.200 → 3.163（−1.2%，3 個 seed 的 on 都比 off 低），RULER 300K 3.391 → 3.311。上午 v1 兩邊都是 BF16 state（一邊沒開 monokernel）時是 3.178 / 3.182，所以這是 BF16 state 本身的精度造成的，不是 monokernel 算錯。AgentX 用 golden AL，所以上面的速度沒有反映這 1% 左右的損失。
- **跟週五（第 17.6 節）比**：週五是 +3.4% / +8.1% / +5.9%。不過週五兩條 lane 的 SGLang 不一樣（BF16 lane 多了 #49036 和 FlyDSL KDA verify 讀 BF16 state），而且當時的 base 是 AMD PR stack，prefill 快很多。今天兩邊的 SGLang 一樣，只差 state dtype 和 mono tree。

#### 17.7.2 設定

- node `crsuse2-m2m-v3-153`（job 290395），container `jacchang_mono1116`，image `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261115`（sglang `cbd4439f3d`，aiter `285579b62`，image 內建 sgl-amd-mono `7113024`，`glm53_flash` 跟 main `572ddf1` 一樣，已經包含 #71 / #97 / #98 / #100）。
- 兩邊同一個 SGLang：`cbd4439f3d` + #49036（`~/PR/wt-sglang-img1115-49036`，`1695988c28`）。#49036 不是必要的：沒有它的話，HIP prefill 只接受 FP32 state，BF16 時會改走 Triton。
- off：FP32 state + image 內建的 mono `7113024`（`PYTHONPATH=$SGL/python`）。on：`--mamba-ssm-dtype bfloat16` + PR 的 mono `e83543d`（`PYTHONPATH=~/PR/wt-mono-bf16state:$SGL/python`）。其他 flag 完全一樣：`--kv-cache-dtype bfloat16 --enable-amd-monokernel`，TileLang DSA，ReplaySSM 關（mono 的 spec 需要），golden AL（MTP5 3.56 / MTP3 2.99），每個 conc 3600 s。
- 第 1 輪：off 在 GPU 0–3、on 在 GPU 4–7，同時跑 conc1（MTP5）、conc4、conc8（MTP3）。第 2 輪：GPU 對調，跑 conc4、conc8。
- mono 涵蓋的步驟（server log）：MTP5 是 `target_verify steps of batch size [1, 2] x 6 tokens`，MTP3 是 `[1, 2, 3, 4] x 4 tokens`。conc8 超過 16 rows 的 verify 走 SGLang。
- state slot：MTP3 off 1460、on 2827；MTP5 off 1195、on 2314。

#### 17.7.3 AgentX

| conc（MTP） | 輪 | P90 Int off → on | P50 Int | P90 E2E-norm Int | P90 TTFT（s） | Throughput/GPU | 完成的 request |
|---|---|---|---|---|---|---|---|
| 1（MTP5） | 1 | 429.4 → **445.5（+3.8%）** | 467.4 → 482.7（+3.3%） | 211.9 → 224.4（+5.9%） | 0.729 → 0.746 | 6,569 → 6,704（+2.0%） | 306 → 309 |
| 4（MTP3） | 1 | 389.1 → 374.9（−3.7%） | 547.7 → 555.6（+1.4%） | 205.0 → 180.8 | 0.614 → 0.657 | 9,122 → 9,408（+3.1%） | 698 → 724 |
| 4（MTP3） | 2 | 396.2 → 406.8（+2.7%） | 543.0 → 565.0（+4.1%） | 198.1 → 200.9 | 0.623 → 0.636 | 9,134 → 9,236（+1.1%） | 703 → 709 |
| 4 平均 | | 392.6 → **390.8（−0.5%）** | 545.4 → 560.3（+2.7%） | 201.5 → 190.8 | 0.619 → 0.646 | 9,128 → 9,322（+2.1%） | |
| 8（MTP3） | 1 | 270.9 → 270.7（−0.1%） | 419.8 → 450.8（+7.4%） | 175.7 → 181.1 | 0.646 → 0.630 | 15,706 → 15,650（−0.4%） | 1,406 → 1,403 |
| 8（MTP3） | 2 | 261.3 → 276.7（+5.9%） | 414.3 → 448.8（+8.3%） | 172.7 → 181.4 | 0.668 → 0.657 | 15,746 → 15,599（−0.9%） | 1,410 → 1,399 |
| 8 平均 | | 266.1 → **273.7（+2.8%）** | 417.1 → 449.8（+7.8%） | 174.2 → 181.3（+4.1%） | 0.657 → 0.644 | 15,726 → 15,625（−0.6%） | |

| server log 每 token 時間 p50（ms），off → on | conc1 | conc4 第 1 輪 | conc4 第 2 輪 | conc8 第 1 輪 | conc8 第 2 輪 |
|---|---|---|---|---|---|
| decode batch 1–2 | 2.15 → 2.09（−2.8%） | 1.80 → 1.75（−2.8%） | 1.80 → 1.74（−3.3%） | 1.95 → 1.87（−4.1%） | 1.94 → 1.88（−3.1%） |
| decode batch 3–4 | | 3.07 → 2.92（−4.9%） | 3.03 → 2.95（−2.6%） | 3.08 → 2.93（−4.9%） | 3.12 → 2.98（−4.5%） |
| P90 Int，扣掉 stall | 436.9 → 452.0（+3.5%） | 411.1 → 404.8（−1.5%） | 415.1 → 429.5（+3.5%） | 312.9 → 330.0（+5.5%） | 302.9 → 330.0（+9.0%） |
| 設定 P90 的 request：step / stall（ms） | 2.33 / 0.02 → 2.24 / 0.02 | 2.46 / 0.13 → 2.52 / 0.18 | 2.46 / 0.08 → 2.35 / 0.11 | 3.15 / 0.64 → 3.10 / 0.63 | 3.29 / 0.58 → 3.08 / 0.58 |
| 設定 P90 的 request：平均 decode batch | 1.24 → 1.24 | 1.70 → 2.06 | 1.64 → 1.66 | 3.08 → 3.25 | 3.23 → 3.05 |

- conc4 第 1 輪的 −3.7% 是 timeline 的差別：on 設定 P90 的 request 剛好跑在比較大的 decode batch（平均 2.06，off 1.70），同一個 decode batch 下 on 還是比較快。兩邊都完成的 695 個 turn 比，結果一樣是 −3.6%，TTFT 比值的中位數 1.001，所以不是 prefill 變慢。第 2 輪 GPU 對調後變成 +2.7%。
- 同一個設定兩輪的差距：conc4 off 389.1 / 396.2（1.8%），on 374.9 / 406.8（8.5%）；conc8 off 270.9 / 261.3（3.6%），on 270.7 / 276.7（2.2%）。GPU 組合沒有固定偏向哪邊。
- accept length（server log）兩邊一樣：MTP3 2.991，MTP5 3.559 / 3.562，就是 golden AL。
- 兩邊都有大約 4.5% 的 `InvalidInferenceResultError`（No responses with actual content），數量差不多（例如 conc8 63 / 63、60 / 62）。週五的 run 也一樣（conc4 39，conc8 61），是 dataset 本身的空回應，不是 regression。
- 跟 1106 stock 曲線（同樣 FP32 + mono，舊 image）比，今天的 off：conc1 429.4 對 430.6，一樣；conc4 392.6 對 403.1（−2.6%）；conc8 266.1 對 293.9（−9.5%），TTFT p90 0.657 對 0.418 s。新 image 的 TTFT 比較長，conc 越高越明顯。可能的原因之一：1106 曲線用 `--cuda-graph-backend-prefill disabled`（breakable prefill graph 要有 sglang#48875 才會 replay），今天的 recipe 用 `breakable`，而 #48875 不在 `cbd4439f3d` 裡。這點還沒驗證。（11/19 更正：這個推測不對。prefill kernel 和 breakable graph 都沒有差，原因是 sglang#49886 的 `--hybrid-cache-checkpoints-level 2` 讓 prefix 命中率從 96.6% 掉到 94.5%，見第 17.10 節。）

#### 17.7.4 準確度（真實 acceptance，MTP3，`--max-running-requests 8`，跟 #98 的 long-context server 一樣）

| GSM8K（sgl-eval，thinking，temp 1.0，top-p 0.95，4096 tokens，4 threads） | seed 0 | seed 1 | seed 2 | 平均 | accept length 平均 |
|---|---|---|---|---|---|
| off：FP32 state + mono `7113024` | 96.97% | 97.12% | 97.19% | 97.09% | 3.200 |
| on：BF16 state + mono `e83543d` | 97.19% | 97.50% | 96.82% | 97.17% | 3.163（−1.2%） |

| RULER2 NIAH（thinking，8192 tokens，4 threads） | mk_niah_basic | mk_niah_hard | mv_niah_basic | mv_niah_hard | 平均 | accept length |
|---|---|---|---|---|---|---|
| 300K off（每個 task 20 題） | 100.00% | 90.00% | 100.00% | 65.25% | 88.81% | 3.391 |
| 300K on | 100.00% | 85.00% | 98.75% | 64.72% | 87.12% | 3.311 |
| 640K off（每個 task 10 題） | 100.00% | 80.00% | 100.00% | 41.67% | 80.42% | 3.429 |
| 640K on | 100.00% | 90.00% | 100.00% | 20.16% | 77.54% | 3.581 |

- 300K 逐題比（`tools/ruler_pair_1116.py`）：兩邊 80 題的答案完全一樣。只有 off 對的 4 題：`mk_niah_hard-12317`（on 想到 8192 tokens 被截斷）、`mv_niah_basic-0`（on 0.75）、`mv_niah_hard-6893`、`mv_niah_hard-9428`。只有 on 對的 2 題：`mv_niah_hard-10794`、`mv_niah_hard-12446`。兩邊都錯 8 題。差距落在 `*_hard` 的翻轉和一次截斷，最直接的檢索 `mk_niah_basic` 兩邊都是 20/20。
- 640K 只跑 10 題：node 的 job 22:05 結束，20 題跑不完。兩邊 40 題的答案一樣。只有 off 對的 3 題：`mv_niah_hard-5111`、`-10445`、`-10966`。只有 on 對的 2 題：`mk_niah_hard-6577`、`mv_niah_hard-10524`。兩邊都錯 6 題，截斷的 off 1 題、on 2 題。
- 怎麼看：兩個長度都是 on 低一點（−1.7、−2.9 點），差距幾乎都在 `mv_niah_hard`（300K −0.5 點，640K −21.5 點，640K 每題的權重是 10%）。翻轉兩個方向都有，題數這麼少，統計上分不出來（640K 不一致的題是 3 比 2）。`mk_niah_basic`、`mv_niah_basic` 兩個長度兩邊都是滿分，所以 BF16 state 在 640K 沒有壞掉。不過 `mv_niah_hard` 的方向兩次都一樣，要再跑才能排除 BF16 state 的精度影響（第 17.7.6 節）。
- on 的 300K 第一次在產生 dataset 時就失敗：兩邊同時往 `~/.cache/sgl_eval/ruler2/` 產生同一個 dataset，on 的暫存目錄被 off 搬走（`FileNotFoundError: .../mk_niah_basic.partial/test.jsonl`）。改成 on 用自己的 `HOME` 重跑（`tools/ruler_on_retry_1116.sh`），dataset 跟 off 的完全一樣。640K 兩邊也各用自己的 `HOME`（`tools/ruler640_1116.sh`）。

#### 17.7.5 上午 v1（定義錯的 off，留作參考）

上午第一次跑時，off 是「BF16 state、不開 monokernel」，on 是 BF16 state + PR 的 mono，一輪，同一個 SGLang。這剛好是 PR #141 Not covered 裡「BF16 state 下 monokernel 開 / 關」那一項。

| | off（BF16，沒有 mono） | on（BF16 + PR mono） | 差 |
|---|---|---|---|
| conc1 P90 Int | 312.9 | 443.4 | +41.7% |
| conc1 Throughput/GPU | 5,412 | 6,678 | +23.4% |
| conc4 P90 Int | 253.8 | 371.8 | +46.5% |
| conc4 Throughput/GPU | 8,172 | 9,278 | +13.5% |
| GSM8K seed 0 / 1 / 2 | 97.04% / 96.89% / 96.97% | 97.27% / 96.97% / 97.04% | |
| GSM8K accept length 平均 | 3.178 | 3.182 | |

#### 17.7.6 下一步

- conc4 / conc8 再跑第 3、4 輪，或拉長 window，把兩輪 1.8–8.5% 的雜訊壓下來。
- ~~用 `--cuda-graph-backend-prefill disabled` 重跑 off 的 conc8，確認新 image 的 TTFT 變慢是不是 breakable prefill graph 造成的。~~ 11/19 做完：不是 prefill graph，是 checkpoint level，見第 17.10 節。
- RULER 640K 補到每個 task 20 題，`mv_niah_hard` 多跑幾個 seed。再加一組「BF16 state、monokernel 開 / 關」的 RULER，把 BF16 state 的精度和 monokernel 分開。
- PR #141 要的 control point（超過 16 rows，例如 conc16 MTP3）和第 2 輪的 conc1 還沒跑。
- 真實 acceptance 下 BF16 state 的 accept length 少 1.2%，要在 PR 裡寫出來，說明這是 state dtype（#49036）的取捨，不是 monokernel。

#### 17.7.7 檔案

- AgentX：`/shared_nfs/jacchang/agentic_1116_onoff_v2/{r1,r2}-{off,on}/`，分析在 `/shared_nfs/jacchang/onoff_v2_analysis_1116/`（`tail_r*_conc*.md`、`matched_r*_conc*.txt`）。
- 準確度：`/shared_nfs/jacchang/acc_1116_onoff_v2/`（GSM8K、off 的 300K）、`/shared_nfs/jacchang/acc_1116_onoff_v2_ruler_on/`（on 的 300K）、`/shared_nfs/jacchang/acc_1116_ruler640_{off,on}/`。v1 在 `/shared_nfs/jacchang/agentic_1116_onoff/`、`/shared_nfs/jacchang/acc_1116_onoff/`。
- 工具（`~/SGLang-benchmarks/tools/`）：`agentx_onoff_v2_1116.sh`、`onoff_driver_v2_1116.sh`、`acc_onoff_v2_1116.sh`、`after_agentx_v2_1116.sh`、`ruler_on_retry_1116.sh`、`ruler640_1116.sh`、`monitor_onoff_v3_1116.sh`、`summ_onoff_v2_1116.py`、`matched_turns_1116.py`、`ruler_pair_1116.py`、`agentic_tail.py`。

### 17.8 11/17：PR B（DSA indexer on gfx950）開 draft，[sgl-project/sglang#51127](https://github.com/sgl-project/sglang/pull/51127)（原 #50936）

撰寫：2026-11-17（台灣時間）。全部在 node 247（`crsuse2-m2m-v3-247`，spur job 293646，container `jacchang_prB1117`，1106 image，aiter 有 #6435）。

**#44661 拿掉。** #44661 的 `fused_hadamard_act_quant` 會把 #44417 已經轉過的 query 再轉一次（double Hadamard，第 16 節），所以 baseline 和 PR B 都不含 #44661，也不設 `SGLANG_DSA_FUSE_HADAMARD_QUANT`。第 17.2 節的 #11（double Hadamard 修正）因此不需要。

| | branch / commit | 內容 |
|---|---|---|
| baseline | `glm53-prstack1106-no44661` `309b924c75`（`~/PR/wt-glm53-1117-base-no44661`） | 1106 stack（`09111fe9a3`）revert #44661 的 `62d2711d65` |
| PR B 測試用 | `jacob/glm53-flash-1117-prB-stack` `74d9634911`（`~/PR/wt-glm53-1117-prB`） | baseline + PR B 一個 commit |
| 上游 PR | `jacob/glm53-flash-dsa-indexer-verify-gfx950-v2` `e2b14486a6`（`~/PR/wt-glm53-prB-upstream`，已 push） | #44417 head `380ee8c9bc` + merge main `d08d15985b` + PR B |

PR B 的四個部分（+446 / −26，含 3 個測試檔）：

| 部分 | 改了什麼 |
|---|---|
| logits `next_n=6`（#5） | `aiter_paged_mqa_logits(..., next_n)`；`KPoolWritePlan.pool_context_lens_per_req`（HIP 才配，= 最後一個 draft row 的 pool 長度 + 5）；`_kpool_verify_next_n_layout` 只在 row 能整除成 request、page table 連續時才用 |
| top-k v2 預設（#3） | `_kpool_topk_v2_enabled` 也認 HIP；`model_hook` 的 HIP 分支改成 `SGLANG_OPT_USE_TOPK_V2 = index_kpool > 1`（明確設定的 env 優先）。用 sglang 的 config loader 讀 GLM-5.3-Flash 是 `Glm5NextConfig`、`index_kpool=4`，所以 server 會開 v2 |
| 一個 launch（#7） | `_get_q_k_fused_x_proj(norm_and_rotate=False)` 回傳 fp32 key 和沒轉的 query；`kpool_write_tail_and_maybe_compress` 多了 `key_norm`、`query`、`head_weights`，write kernel 做 key LayerNorm，同一個 grid 多開 `rows × 32` 個 program，一個 head 一個，用 `kpool_fp8_index.py` 原本的 `_hadamard_quantize_fp8` 轉 + quant + 算 head gate。不用 #44661 的 code，跟 `act_quant(rotate_activation(q))` bit 完全一樣。只在單 stream 走（HIP 沒設 `SGLANG_ROCM_USE_MULTI_STREAM` 就沒有 alt stream） |
| q_b quant 回歸（#10） | `_is_block_scale_fp8` 先讀 fp32 `weight_scale_inv` |

- 第一版一個 program 做一個 token row（32 個 head 串行），合起來的 launch 反而比 5 個 kernel 慢（bs1 23.7 vs 21.6 µs），因為每個 program 要串行跑 32 次 Hadamard。改成一個 program 一個 head 後是 7.5 µs。RULER 是用第一版跑的，兩版輸出 bit 完全一樣（同一個單元測試、`check_idx_fused_quant_1117.py`），所以 RULER 結果有效。

**準確度（RULER，13 tasks × 20，同時跑，baseline GPU 0–3、PR B GPU 4–7）**

| | 32K baseline | 32K PR B | 128K baseline | 128K PR B |
|---|---|---|---|---|
| 13 tasks 平均 | 95.26 | 95.38 | 92.09 | 92.92 |
| NIAH 8 個 task 最低 | 95.0（multivalue） | 95.0 | 92.5（multivalue） | 96.25 |
| vt / cwe / fwe | 100 / 100 / 98.33 | 100 / 100 / 100 | 100 / 78 / 86.67 | 100 / 80 / 86.67 |
| qa_1 / qa_2 | 80 / 65 | 80 / 65 | 80 / 60 | 80 / 65 |

單元測試：上游 branch（main + #44417 + PR B）上 19 個全過（`test_dsa_kpool_fused_x_proj_gfx950.py` 7 + 5 subtests、`test_aiter_paged_mqa_logits_next_n.py` 1、`test_fp8_per_channel_detection.py` 11）。

**Kernel（每個 indexer layer，HIP graph replay，µs，GPU 0；GPU 1 每格差 3 µs 以內）**

| decode batch | write + quant baseline（5 個 launch，head gate 用 torch.compile） | PR B（1 個 launch） |
|---|---|---|
| 1 | 19.1 | 7.5 |
| 6 | 20.6 | 7.6 |
| 16 | 21.6 | 7.7 |
| 32 | 22.7 | 7.9 |
| 64 | 26.4 | 11.5 |

| decode batch, context | logits `next_n=1` | `next_n=6` | top-k v1 | top-k v2 |
|---|---|---|---|---|
| 6, 32K | 10.1 | 8.2 | 12.6 | 7.5 |
| 6, 64K | 17.2 | 14.4 | 15.6 | 9.0 |
| 6, 128K | 31.2 | 24.6 | 21.9 | 12.6 |
| 16, 32K | 22.0 | 14.8 | 12.6 | 7.3 |
| 16, 64K | 39.6 | 24.8 | 17.0 | 9.4 |
| 16, 128K | 73.6 | 45.8 | 22.2 | 12.8 |
| 32, 32K | 39.9 | 25.8 | 13.2 | 7.5 |
| 32, 64K | 74.4 | 46.9 | 17.0 | 9.6 |
| 32, 128K | 122.5 | 68.0 | 24.3 | 13.9 |

- bs6 / 128K 每層省 13.0 + 6.6 + 9.3 = 28.9 µs，11 層 0.32 ms；bs16 / 128K 每層省 51.1 µs，11 層 0.56 ms。

**AgentX（golden AL 3.56，1 小時，baseline GPU 0–3、PR B GPU 4–7 同時跑，env 和 server 參數只差 `PYTHONPATH` 的 tree）**

| conc | P90 Int baseline | PR B | 差異 | P50 Int baseline / PR B | P90 ITL ms baseline / PR B | TTFT p90 s baseline / PR B | Throughput/GPU 差異 |
|---|---|---|---|---|---|---|---|
| 16 | 209.1 | 222.7 | **+6.5%** | 287.6 / 303.4 | 4.78 / 4.49 | 0.572 / 0.551 | +0.7% |
| 32 | 137.9 | 145.5 | **+5.6%** | 216.5 / 229.8 | 7.25 / 6.87 | 0.756 / 0.753 | +1.8% |
| 64（HiCache 216） | 61.47 | 62.99 | **+2.5%** | 96.57 / 97.6 | 16.27 / 15.88 | 1.214 / 1.19 | +2.2% |

- 第 17.3 節估 B 是 conc16 / conc32 +6.0% / +3.1%，實測 +6.5% / +5.6%。conc64 batch 大，indexer 佔 step 的比例小，所以只有 +2.5%。
- 這裡的 baseline 是「1106 stack 去掉 #44661」，沒有 A、C、D 和 topk v2 env，所以 conc16 的 209.1 比第 17.1 節的 246.9 低。

**GSM8K（1319，thinking，temperature 1.0）**：baseline 97.27%（truncated 0.30%），PR B 96.59%（truncated 0.76%）。這個 stack 之前在 temperature 1.0 跑過 95.83–97.50%，差異在 noise 內。第一次跑因為 1319 個 thread 超過 open-file 上限失敗，script 加了 `ulimit -n 65536` 後重跑。

**CI**：draft 沒有 `run-ci` label，pr-gate 會 fail，這是正常的。lint 是 `ruff format` 要重排兩個測試檔，已用 ruff 0.15.1 修好並 force push（上游 `e2b14486a6`，測試 tree `3b643332ce`）。跑 benchmark 的 `74d9634911` 跟它們只差測試檔的排版。

**11/18：#51127 改成不疊 #44417。** 原本的 branch 是「#44417 + merge main + PR B」，所以 PR diff 連 #44417 一起顯示（16 個檔、+847）。合成一個 launch（#7）是改在 #44417 加的 `_get_q_k_fused_x_proj` 上，main 沒有這個 function，所以這部分拿出 #51127，等 #44417 merge 後另開 PR。

| | branch | commit | 內容 |
|---|---|---|---|
| #51127（force push） | `jacob/glm53-flash-dsa-indexer-verify-gfx950-v2`（`~/PR/wt-glm53-1118-prB-clean`） | `b9b731cae0`（main `9d3c71cbbe` + 1 commit，8 個檔，+178 / −9；之前是 `e8001564af`） | logits `next_n=6`、top-k v2 預設、q_b quant 修正 |
| #44417 加一個 commit（方案 A） | `jacob/glm53-flash-dsa-kpool-hip` | `2bb679d0f6`（#44417 原 head `380ee8c9bc` + 1 commit，3 個檔，+271 / −17） | 合成一個 launch（#7），#44417 的第 4 項 |
| 原本完整版本的備份 | `jacob/glm53-flash-dsa-indexer-fused-write-quant` | `e2b14486a6` | 只當備份 |
| 測試 tree | `jacob/glm53-flash-1118-prB-lite`（`~/PR/wt-glm53-1118-prB-lite`） | `eab1678a96` = baseline `309b924c75` + #51127 | 重跑 AgentX 用 |

- main 上單元測試 12 個全過（`test_aiter_paged_mqa_logits_next_n.py` 1、`test_fp8_per_channel_detection.py` 11），ruff format / check 乾淨。
- 準確度沿用：測的 build 多了 #7，但 #7 跟沒合的路徑 bit 完全一樣，所以數值等於 #51127。kernel 表只留 logits 和 top-k（bs16 / 128K 每個 verify step 省 0.41 ms）。
- 11/18 決定用方案 A：#7 放進 #44417（主題就是減少 indexer launch，改的 function 和測試檔也是 #44417 的），不重跑 AgentX。
  - #44417：Summary 加第 4 項，Test plan 加「Item 4」：kernel 表（5 個 launch 19.1–26.4 µs → 1 個 7.5–11.5 µs）、RULER / GSM8K（跟 #51127 一起量）、bit 一致。#44417 的單元測試在 `2bb679d0f6` 上 7 個 + 5 subtests 全過，ruff 乾淨。
  - #51127：AgentX 表放回 +6.5% / +5.6%，標明是「this PR + #44417 的 `2bb679d0f6`」一起量的。kernel 層級分開列：bs16 / 128K 每個 verify step，#51127 省 0.41 ms，`2bb679d0f6` 省 0.15 ms。
- 11/18 下午：`e8001564af` 的 CI lint 又 fail（`check-registered-tests`）。upstream 把 suite `stage-*` 改名成 `base-*`，`test_aiter_paged_mqa_logits_next_n.py` 註冊的 `stage-b-test-1-gpu-small-amd-mi35x` 沒有 workflow 會跑，改成 `base-b-test-1-gpu-small-amd-mi35x`。同時 #51127 又跟 main conflict（有 conflict 時 GitHub 不跑 lint）：main 的 #50997（Kimi-K3.1）在 `aiter_paged_mqa_logits` 加了 `persistent_schedule`，跟 PR 的 `next_n` 同一行，兩個參數都留（目前沒有 caller 同時傳）。rebase 到 main `9d3c71cbbe` 後是 `b9b731cae0`，ruff / isort 乾淨、上面 12 個單元測試過、CI lint 過。`~/PR/wt-glm53-prB-upstream` 還停在舊的 `e2b14486a6`，之後改 #51127 用 `wt-glm53-1118-prB-clean`。
- 11/18 14:30：#44417 跟 main conflict（落後 943 個 commit）。照 branch 原本的做法 merge main（不 force push），head `d111fa196a`，MERGEABLE。
  - `kpool_fp8_index.py`：main 刪掉了 CP helper（`all_gather_and_scatter_pool_slots` 等，沒有 caller），只留 #44417 的 `_rotate_quant_query_head`。
  - CI 名稱跟著 main 改：`test_fused_metadata_copy.py`、`test_dsa_kpool_metadata_fusion.py` 改用 `base-a-jit-kernel-unit`，`test_dsa_kpool_fused_x_proj_gfx950.py` 改用 `base-b-test-1-gpu-small-amd-mi35x`。
  - 在 247（1106 image，GPU 7）上，#44417 的 5 個測試檔 71 passed + 61 subtests，ruff check / format 乾淨。
- 11/19：reviewer Raiden-Makoto 在 `_is_block_scale_fp8` 要求修改（CHANGES_REQUESTED），他說的對。fp32、多欄的 `weight_scale_inv` 不代表是 128 寬的 block：`Fp8LinearMethod` 也支援 `[32, 32]`，但 `fused_rms_fp8_group_quant` 寫死 `group_size=128`，下游 block GEMM 又照 `self.weight_block_size` 讀 activation scale（K=512 時給 4 個 scale，GEMM 要 16 個）。
  - 修在 `85c874654d`：有 `weight_scale_inv` 時，必須是 fp32 而且 `proj.quant_method.weight_block_size == [128, 128]`，否則回 False。另外加了 block-32 和「拿不到 block 大小」兩個回 False 的測試。
  - GLM-5.3-Flash 不受影響：Quark config 裡 `q_b_proj` 是 `per_block [128, 128]`，`quark.py` 528 行把它交給 `Fp8LinearMethod`，`weight_block_size` 是 `[128, 128]`。
  - 用真的 `ReplicatedLinear` 加 `Fp8LinearMethod` 驗證：`[32, 32]` 在 `b9b731cae0` 回 True（bug），修完回 False；`[128, 128]` 兩邊都是 True。
  - 247 的 job 295965 已經結束，這次借 153（job 290401）的 GPU 7 跑：`test_fp8_per_channel_detection.py` 加 `test_aiter_paged_mqa_logits_next_n.py` 共 14 passed，ruff 乾淨。不掛 GPU 時，aiter 偵測不到 gfx arch，fnuz 那組測試會 fail，跟這次改動無關。
  - 已在 review thread 回覆，PR body 的 Scope 補上「只接受 128 × 128 block」，單元測試改成 14 passed。
- 11/18 決定保留 #51127 的 `next_n`，不改用 helper（使用者選的）。背景：main 的 #50997 加了 `aiter_paged_mqa_logits_verify`（Q 當成 `[B, next_n]`，`WavePerEU=4`），目前沒有 caller，功能跟 #51127 的 `next_n` 參數重疊。#51127 可以拿掉 `paged_mqa_logits.py` 的 `next_n` 改動，改成在 k-pool verify 呼叫這個 helper（`request_lens` 傳 `pool_context_lens_per_req`、`q_offset=n_real`）。`_kpool_verify_next_n_layout` 和 `pool_context_lens_per_req` 還是要留。`WavePerEU` 不同，所以 logits kernel 數字要重測。

**檔案**

- RULER：`/shared_nfs/jacchang/ruler1117/out_prB/{base,prB}/`；kernel：`/shared_nfs/jacchang/ruler1117/bench_idx_prB_gpu{0,1}.txt`。
- AgentX / GSM8K：`/shared_nfs/jacchang/agentic_1117_prB/`。
- 工具（`~/SGLang-benchmarks/tools/`）：`ruler_ab_prB_1117.sh`、`gsm8k_ab_prB_1117.sh`、`agentx_prB_1117.sh`、`driver_prB_1117.sh`、`driver2_prB_1117.sh`、`bench_idx_prB_1117.py`、`check_idx_fused_quant_1117.py`。

### 17.9 11/17：PR C（MoE router 和 GEMM dispatch on HIP）開 draft，[sgl-project/sglang#51128](https://github.com/sgl-project/sglang/pull/51128)（原 #51003）

撰寫：2026-11-17（台灣時間）。全部在 node 247（spur job 293646，container `jacchang_prB1117`，跟第 17.8 節同一個環境）。

| | branch / commit | 內容 |
|---|---|---|
| baseline | `glm53-prstack1106-no44661` `309b924c75`（`~/PR/wt-glm53-1117-base-no44661`） | 跟 PR B 同一個 baseline |
| PR C 測試用 | `jacob/glm53-flash-1117-prC-stack` `1cab91201b`（`~/PR/wt-glm53-1117-prC`） | baseline + kdafly 樹裡 C 的 6 個檔案（第 12 節 H、H'、I，第 11 節 B、G） |
| 上游 PR | `jacob/glm53-flash-moe-router-gemm-hip-v2` `8e5c62a71c`（`~/PR/wt-glm53-prC-upstream`，已 push；之前是 `f2d9296838` → `f427e7ced0`） | main `da395ae091` + PR C（+307 / −41，含新的測試檔），第 17.9.1 節的修正 |

搬到 main 時改的地方（跟測試用的樹不一樣）：

- main 的 #46693（MiniMax）把 `moe_fused_gate` 的 split-K partials 輸入（`SCORES_SPLIT_K`）拿掉了，DeepSeek-V4.1 改走 `rocm_router_gate`（只支援 sqrtsoftplus、384 experts，GLM 不能用）。PR C 把它加回來。
- main 多了 `route_quant_handoff`（Kimi-K3），`padded_rows_masked` 改成兩邊 OR 起來。
- main 多了 `_gfx950_triton_a16w16_shapes`（明確註冊要走 Triton 的 shape）。tuned row 的檢查只套在一般的 K ≤ 2048 小 M 規則上，註冊的 shape 還是走 Triton。
- `_aiter_uses_jit_router` 多加「有 fused shared expert」的條件。沒加的話 Kimi-K3（沒有 fused shared expert）在 ROCm 會被拉離 radix4，`test_topk_correction_bias_cache.py` 3 個測試會 fail。GLM 有 1 個 fused shared expert，所以 GLM 的路徑不變，測試用的樹沒有重跑。
- 測試用的 stack 有 #44416，它的 HIP router 先吃 ≤ 16 rows，所以 PR C 的 split-K router 在 E2E 只跑到 17–96 rows。main 上 PR C 涵蓋 1–96 rows。

**結果**

| | baseline | PR C |
|---|---|---|
| GSM8K（1319，thinking，temperature 1.0） | 96.66% | 96.59% |
| AgentX conc16 P90 Int（1 小時，PR 用這組） | 209.9 | 217.3（+3.5%） |
| AgentX conc32 P90 Int（1 小時，PR 用這組） | 138.9 | 144.0（+3.7%） |
| P50 Int conc16 / conc32（1 小時） | 285.8 / 216.9 | 301.8 / 228.4（+5.6% / +5.3%） |
| median ITL conc16 / conc32（1 小時，ms） | 3.50 / 4.61 | 3.31 / 4.38（−5.4% / −5.0%） |
| 第一次 30 分鐘的 P90 Int conc16 / conc32（不用） | 194.3 / 125.7 | 203.7 / 126.8（+4.8% / +0.9%） |

- 第一次 AgentX 只跑了 30 分鐘（job 293646 12:14 UTC 結束），這不該自己決定。之後另外借 node 247（job 294587，`amd-aifw-sglang-v2`，台北 11/17 22:48 – 11/18 03:48），重建 `jacchang_prB1117`，conc16 / conc32 各跑 1 小時，PR 換成 1 小時的數字。GSM8K 和 kernel bench 跟時長無關，沒有重跑。
- 30 分鐘那次 conc32 只有 +0.9%，1 小時是 +3.7%，跟 conc16 一致，30 分鐘的 window 太短。第 17.3 節估 C 是 +3.3% / +1.7%，實測 +3.5% / +3.7%。
- reserve-crusoe skill 的 script 裡 partition / account / QoS 還是舊的 `amd-aifw-sglang`，現在要用 `amd-aifw-sglang-v2`（`amd-aifw-sglang-v2-qos`），已 pending 的 hold 也是 v2。
- kernel（HIP graph，GPU 0 / 4 差 0.3 µs 以內）：router GEMM + top-k 15.3–16.1 → 7.7–10.6 µs（6–96 rows）；o_proj Triton → tgemm 在 M=18–60 快 14–36%；mHC post + pre 在 18–192 rows 快 2–15%，288 rows 合起來反而慢（21.35 → 24.40），所以上限停在 192。
- 單元測試（main + PR C）：新的 `test_glm_router_split_k_gfx950.py` 3 個過；router / top-k 相關 9 個檔 116 個過，`test_topk_routing_inputs.py` 的 2 個 fail 在 main 上也 fail。lint：ruff 0.15.1（pre-commit 的 select）、isort 7.0.0 乾淨。

**檔案**

- AgentX 1 小時：`/shared_nfs/jacchang/agentic_1117_prC_1h/`；30 分鐘、GSM8K、kernel：`/shared_nfs/jacchang/agentic_1117_prC/`（`bench.log` 是 kernel）。
- 工具（`~/SGLang-benchmarks/tools/`）：`driver_prC_1117.sh`、`driver_prC_1h_1117.sh`、`agentx_prC_1117.sh`、`bench_prC_1117.sh`（GSM8K 沿用 `gsm8k_ab_prB_1117.sh`）。

#### 17.9.1 11/18：轉 ready 前的修正

自己 review 時找到三個問題，都在 `f427e7ced0` 修掉，PR body 的 Scope 和 Test plan 也更新了：

- **CI lint 失敗**：upstream 把 `stage-*` suite 改名成 `base-*`，新測試註冊的 `stage-b-test-1-gpu-small-amd-mi35x` 沒有 workflow 會跑，`check-registered-tests` 擋下來。改成 `suite="base-b-test-1-gpu-small-amd-mi35x"`，CI lint 已過。
- **router 的影響範圍比 PR 寫的大**：`_aiter_uses_jit_router` 只看 `_use_aiter`，MI300 也會換成 Triton router；而且「single-group sigmoid + fused shared expert」也涵蓋 GLM-5 / 5.2、Kimi-K2.5-MXFP4、GLM-4.7-Flash。改成要 gfx95（MI300 / MI325 維持 aiter `biased_grouped_topk`），加測試確認 MI300 上 gate 會拒絕，Scope 寫明其他 model 也會走新路徑。
- **mHC 上限也會改到 CUDA**：CUDA 上三個 TileLang mHC 環境變數預設都開，GLM-5.3-Flash 一樣走融合路徑，PR 寫的「CUDA is unchanged」不成立。改成 `192 if _use_aiter_gfx95 else 16`。
- 順手：`select_experts` 裡兩份「走 Triton router」的條件抽成一個 `jit_router_writes_shared`，`partials_to_grouped_router` 從它推出來，行為不變。rebase 到 main `b085beba3d`。

驗證在 node 250（spur job 296048，`amd-aifw-sglang-v2`，container `jacchang_prC1118`，image `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261106`，沒有套 aiter#6435）：

| | main `b085beba3d` | + PR C |
|---|---|---|
| GLM-5.2-FP8 GSM8K，seed 0 / 1 / 2（TP8，FP8 KV，Triton DSA，沒有 MTP，`--max-running-requests 64`） | 95.38 / 95.00 / 95.30%（平均 95.23%） | 94.69 / 94.84 / 95.45%（平均 94.99%） |
| GLM-5.2-FP8 真實 router 權重（第 3、40、77 層，6–96 rows，各 20 個 batch）：experts 跟 fp64 參考一致的 rows | 12,686 / 12,960（97.9%） | 12,959 / 12,960（99.99%） |
| 單元測試（9 個檔） | `test_topk_routing_inputs.py` 2 fail | 117 過、同樣 2 fail；新測試檔 4 個過 |

- GLM-5.2-FP8（`GlmMoeDsaForCausalLM`，256 experts，hidden 6144）server log 有「Shared experts fusion optimization enabled」，router 權重是 bf16 `[256, 6144]`（12 個 split），所以 96 rows 以內同時跑到 split-K GEMV 和 Triton router。
- GSM8K 第一次只有 seed 0（95.38 vs 94.69），差距在單次的標準誤（約 0.8 個百分點）內但證據不夠，兩邊各補兩次。三次範圍重疊。
- kernel 檢查：舊路徑跟參考不一致的 rows 全部來自先把 logits 捨入成 bf16；新路徑用 fp32 partials，只差 1 row。
- 檔案：`/shared_nfs/jacchang/prC_ready_1118/`；工具：`driver_prC_ready_1118.sh`、`driver_prC_ready_seeds_1118.sh`、`gsm8k_glm52fp8_prC_1118.sh`、`check_router_glm52_prC_1118.py`。

轉 ready 之後 main 進了 #50707（muse-spark gfx1250），`unquant.py` 衝突，rebase 到 main `da395ae091`，push 成 `8e5c62a71c`：

- main 把 Triton a16w16 / FlyDSL 的 gate 從 `_is_gfx95` 改成 `_rowmajor_ok`（gfx95 或 gfx1250）。保留 main 的 `_rowmajor_ok`；PR C 的 tuned row 檢查（`_has_tuned_a16w16`，查 aiter 的 gfx950 tuned CSV）只在 `_is_gfx95` 時套用，gfx1250 維持 main 的行為。
- `test_unquant_apply_with_addend.py` 在 MI355X 上 6 fail / 3 過，main `da395ae091` 上一模一樣（測 cuBLAS 路徑，ROCm 上本來就 fail），不是 PR C 造成的。lint、registered test 檢查過。
- 驗證用的 node 250 原本的 job 296048 06:12 UTC 被外部取消，另借 job 296304（1 小時），用完已釋放。
- PR 狀態：MERGEABLE，CI lint 過；`pr-test-finish` fail 是因為還沒有 `run-ci` label。

11/19 reviewer（Raiden-Makoto）要求改：GLM 分支的 `router_max_tokens` 沒檢查有沒有 fused shared expert，說 `--disable-shared-experts-fusion` 時 partials 會撞到 `biased_grouped_topk_gpu` 的 assert。

- crash 不成立：沒有 shared slot 時 `partials_to_grouped_router` 是 False，`select_experts` 一開頭就把 partials 加總回 logits。node 153（job 290401，container `jacchang_mono1113`）實測 shared slot 0 / 1、6 / 36 / 96 rows 都沒有 assert，結果跟先加總 logits 位元一致（`tools/check_prC_fusion_off_1119.py`）。
- 但白跑一次 GEMV 加 reduce，所以照建議收緊：GLM 分支要有 global shared slot（`get_model().shared_experts.num_shared_topk_slots > 0`，而且不是 per-rank slot），條件跟 Triton router 一致，Waterfill 也涵蓋。V4.1 不變。新測試 2 個（gate 准入、沒有 shared slot 時 partials 先加總），測試檔共 6 個過；router / top-k 7 個檔 103 過，只有 `test_topk_routing_inputs.py` 2 個 main 上也 fail。commit `3a256f3bbb`，PR body 測試列改成 6 passed。
- 同一天 #51127 也被要求改（`_is_block_scale_fp8` 會把 `[32, 32]` block 當成 128），意見正確，先不改。

### 17.10 11/19：1115 image 的 TTFT 變長，不是 prefill regression，是 `--hybrid-cache-checkpoints-level` 預設改成 2

撰寫：2026-11-19（台灣時間）。第 17.7 節看到 1115 image 的 AgentX TTFT p90 比 1106 曲線長（conc8 0.657 vs 0.418 s），當時猜是 breakable prefill graph。結論：

- **prefill kernel 沒有變慢**：兩個 image 用各自的 SGLang、同一組 GLM.sh 參數跑 fixed-length，TTFT 和 EXTEND 每步的 kernel 時間都在 ±2.5% 內。
- **原因是 prefix 命中率下降**：sglang#49886（`a14fd4e5b4`，11/13 merge，在 1115 的 `cbd4439f3d` 裡，不在 1106 的 `c06e39c6ca` 裡）新增 `--hybrid-cache-checkpoints-level`，預設 2，只在分岔點和 request 結束時存 KDA state checkpoint，每條 path 最多 1 個。radix cache 只能重用到有 checkpoint 的位置，所以 AgentX 每輪要重算的 token 多了 62%。
- **level 3 就是舊行為**：每個 prefill chunk 和每 `--mamba-track-interval` 個 decode token 存一個，沒有上限。1106 的 server args 是 `mamba_max_states_per_path: -1`、沒有 level，等同 level 3。`mamba_path_state_cap_of`（`arg_groups/overrides.py`）在沒設 `--mamba-max-states-per-path` 時，level < 3 回 1、level 3 回 −1（不限），所以**只要 `--hybrid-cache-checkpoints-level 3` 就夠**。`--mamba-max-states-per-path` 已經 deprecated，不用再加。
- 上游 #49886 / #50034（`docs/advanced_features/hybrid_cache_checkpoints.mdx`）在 B300 + HiCache 量到 level 2 比較好，但比的是 level 3 加上每條 path 1 個的上限。我們 MI355X 沒開 HiCache，結果相反。

**fixed-length A/B**（node 151，GPU 0–3 / 4–7 同時跑，MTP3 golden AL，`--kv-cache-dtype bfloat16`，各 image 自己的 SGLang，median TTFT ms）

| ISL | conc | pgoff 1106 | pgoff 1115 | bcg 1106 | bcg 1115 |
|---|---|---|---|---|---|
| 1K | 1 | 89.0 | 88.2 | 50.9 | 52.9 |
| 1K | 4 | 205.7 | 161.9 | 96.4 | 91.6 |
| 1K | 8 | 166.4 | 261.1 | 121.1 | 151.1 |
| 8K | 1 | 219.7 | 219.6 | 215.0 | 219.0 |
| 8K | 4 | 599.4 | 396.2 | 502.6 | 524.1 |
| 8K | 8 | 504.3 | 592.2 | 543.7 | 482.0 |
| 70K | 1 | 1404 | 1413 | 1443 | 1416 |
| 70K | 4 | 3428 | 3453 | 3426 | 3456 |
| 70K | 8 | 6259 | 6327 | 6487 | 6324 |

- pgoff 是 `--cuda-graph-backend-prefill disabled`，bcg 是 `breakable --cuda-graph-max-bs-prefill 2560`。
- conc1 和 70K 每個 conc 都在 ±2.5% 內。1K / 8K 的 conc4 / 8 兩個方向都有大跳動，是 request 跟 decode 交錯的時機不同，不是 image 的差別。
- 1106 的 bcg 在 1K 有 replay（89 → 51 ms），第 17.7 節「1106 沒有 #48875 時 breakable 不會 replay」的說法也不對。
- trace（`PROF_IN_OUT_OVERRIDE="8192:16 70000:16"`，conc1，用 `trace_analysis/diagnostics/extend_step_compare.py` 依 `step[EXTEND ...]` window 比較）：pgoff 8K 每步 191.9 → 189.6 ms（−1.2%），70K 每個 16K chunk 326.4 → 328.7 ms（+0.7%）；bcg 8K +0.5%。沒有哪個 kernel 變慢超過 noise。

**AgentX 確認**（node 151，1115 image，跟第 17.7 節的 off 同一套：`1695988c28`、FP32 state、`--enable-amd-monokernel`、TileLang DSA、ReplaySSM 關，conc8，MTP3，3600 s）

| | P90 Int | P50 Int | TTFT p90 | prefix 命中 | 重算 token | 每 GPU throughput |
|---|---|---|---|---|---|---|
| level 2（預設） | 262.3 | 421.7 | 0.688 s | 94.47% | 13.39M | 15,620 |
| level 3 | 285.1 | 435.4 | 0.461 s | 96.59% | 8.27M | 15,674 |
| 1106 stock 曲線 | 293.9 | 440.2 | 0.418 s | 96.59% | 8.26M | 15,647 |
| 11/16 off r1 / r2 / r3（level 2） | 270.9 / 261.3 / 269.3 | — | 0.646–0.674 s | — | — | — |

- level 3 的命中率和重算 token 跟 1106 一模一樣，P90 Int +8.7%。剩下跟 1106 的 −3% 在 run-to-run 的 1.8–8.5% 範圍內。
- 第 17.7 節 conc4 的舊 log 也一樣：重算 token 4.35M → 7.18M（命中 96.96% → 94.94%）。
- **第 17.7 節 1115 image 上的 on / off P90 數字都是在 level 2 量的**，絕對值偏低。on / off 用同一個 level，比例還是可以用，但跟 1106 曲線或 B200 比之前要用 level 3 重跑。

**改了什麼**

- `Agentic/recipe_glm53flash_fp4_{mi355x,b200}_sglang_mtp.sh`：server 認得這個 flag 時加 `--hybrid-cache-checkpoints-level 3`（用 `launch_server --help` 偵測；1106 image 和舊的 B200 image 沒有這個 flag，本來就是 level 3）。`HYBRID_CACHE_CHECKPOINTS_LEVEL=2` 可以量預設值。在 151 的兩個 container 試過：1115 加上、1106 不加。B200 還沒量，設 level 3 是讓兩邊同一個 checkpoint 規則。
- `BenchFixedLength/GLM.sh`，1115 image 上的兩個問題：
  - sglang#47990 拿掉了 `sglang.bench_serving`，bench cell 一瞬間跑完、沒有資料。改成有 `sglang.benchmark.serving` 就用它。
  - `/start_profile` 變成 admin-only，沒有 `--admin-api-key` 時每個 profile request 都是 403，prof 目錄是空的。bench client 會送 `SGLANG_ADMIN_API_KEY` 的 build（有 `sglang.srt.utils.auth.admin_auth_headers`）在開 profile 時自動設 key 並傳給 server。

**檔案與環境**

- node 151：spur job 297937（`amd-aifw-sglang-v2`，到台北 11/19 24:00），container `jacchang_pre1106` / `jacchang_pre1115`。153 被另一個 session 佔用，沒動。
- fixed-length：`results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.23-rocm10-mi35x-202611{06,15}/{bench,prof}-Fixed-MTP-GoldenAL-prefill1119-{pgoff,bcg}/`。
- AgentX：`/shared_nfs/jacchang/agentic_1119_ckpt/{l2,l3}/`。
- 工具：`tools/prefill_reg_lane_1119.sh`（fixed-length lane）、`tools/agentx_ckpt_level_1119.sh`（AgentX l2 / l3）、`trace_analysis/diagnostics/extend_step_compare.py`。
