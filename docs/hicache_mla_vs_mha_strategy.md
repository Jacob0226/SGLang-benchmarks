# HiCache 在 MLA / MHA / GQA / MQA 下的策略

整理 SGLang HiCache 在不同 attention 架構下：
- KV cache **是否切（TP 分片）vs 全 rank 複製**
- 每架構下的 **per-token KV 大小**
- 適合的 **write policy / prefetch policy**
- 背後的程式碼依據與架構原因
- L1/L2/L3 速度層級與 backend 選擇
- 多 node 共享 L3 何時真的有用

---

## 0. 前置概念：儲存階層的速度（Why RDMA > NVMe SSD）

理解 HiCache 三層架構前，要先搞懂底層儲存的速度差，特別是「為什麼遠端 RAM 比本地 SSD 快」。

### 速度對照表

| 媒介 | 物理形式 | 頻寬（單裝置） | 延遲 |
|------|---------|----------------|------|
| GPU HBM3 (L1) | GPU 內 | 5–8 TB/s | ~10 ns |
| DDR5 RAM (L2) | 主機板 DIMM | ~100 GB/s | ~80 ns |
| USB 3.2 Gen 2 | 線材+控制器 | ~1 GB/s | ms 等級 |
| SATA SSD | SATA 介面 | ~0.5 GB/s | ~100 µs |
| NVMe SSD (PCIe Gen4) | PCIe 直連 | ~7 GB/s | ~50 µs |
| NVMe SSD (PCIe Gen5) | PCIe 直連 | ~14 GB/s | ~20 µs |
| **InfiniBand / RoCE RDMA**（單口） | 400 Gbps NIC | **50 GB/s** | **1–5 µs** |
| **多口 RDMA 聚合**（HGX/MI300 機型） | 多 NIC | **200+ GB/s** | 1–5 µs |

### 為什麼 RDMA 比本地 NVMe SSD 還快？

核心差別：**RDMA 在「動 DRAM」，NVMe 在「動 NAND flash」**。

#### NAND Flash 是什麼？

NAND flash 是一種**非揮發性電子記憶體**（斷電後資料還在），用「浮動閘極（floating gate）」結構儲存電子。SSD、USB 隨身碟、SD 卡都是它。

特性：
- **斷電不失憶**（vs DRAM 斷電就清空）
- **儲存密度高**、單位 GB 便宜
- **讀取要走充放電過程**（µs 等級，比 DRAM 慢 1000 倍）
- **寫入更慢**（要 erase 整個 block 再寫）
- **有壽命限制**（單 cell 寫入次數有限，現代 SSD 用 wear-leveling 分散）

DRAM（用在 RAM）反而是「揮發性」電容式記憶體，靠電容充放電快速讀寫，但斷電就忘。所以「DRAM 快但易失，NAND 慢但持久」。

#### 兩者的瓶頸位置

**NVMe SSD 的瓶頸是「NAND flash 本身」：**
```
請求 → NVMe controller → NAND flash chip 讀取 → 過 PCIe → CPU/GPU
                              ↑
                           瓶頸在這
                          NAND 物理特性：µs 級充放電
                          PCIe Gen5 x4 = 16 GB/s 上限
```
- NAND 內部讀取需要 µs 級時間（無法用更多 lane 加速）
- 單顆 SSD 受 PCIe lane 限制（一般 x4 = 16 GB/s）
- 想突破要 RAID 多顆 SSD，但 IOPS 跟延遲會放大

**RDMA 在「DRAM ↔ DRAM 直接搬」：**
```
GPU/CPU → 本機 RDMA NIC → 光纖/銅纜 → 遠端 NIC → 遠端 DRAM
                                                     ↑
                                              另一台機器的 RAM
                                              本身就 100 GB/s
```
- 兩端都是 DRAM（100 GB/s 級）
- 中間網路：400 Gbps NIC = 50 GB/s 單口，多 NIC 聚合 200+ GB/s
- 完全 bypass CPU/OS kernel（zero-copy）
- 沒有 NAND 那種物理介質讀取延遲

#### 直接類比

| 比喻 | 速度 |
|------|------|
| NVMe SSD | 從鄰居家「**倉庫**」搬書（要走樓梯找，慢） |
| RDMA | 從鄰居家「**書桌上**」直接拿書（書桌東西本來就在手邊） |

鄰居家的「書桌」（DRAM）就是比你家「倉庫」（SSD）快，**這跟距離（網路 vs 本地）關係不大，跟儲存媒介關係很大**。

---

## 1. 各架構的 KV cache 設計概覽

| 架構 | 代表模型 | KV cache 結構 | TP 切分？ | per-rank KV/token |
|------|---------|---------------|-----------|-------------------|
| **MHA**（純多頭） | Llama 1 65B、老 GPT 系 | 每個 head 各有完整 K、V | **切**（按 head） | 大（~300+ KB） |
| **GQA**（分組） | Llama 3 70B / 405B、Qwen3 大型 | KV head 數 << Q head 數 | **切**（按 KV head） | 小（~30–60 KB） |
| **MQA**（單頭） | PaLM、Falcon | 全模型只有 1 個 KV head | **不切**（複製） | 很小 |
| **MLA**（潛向量） | DeepSeek-V3/R1、Kimi-K2、GLM-5 | 壓縮成 latent + RoPE | **不切**（複製） | 中（~70 KB） |
| **NSA**（稀疏 + MLA） | DeepSeek-V3.2、GLM-5-FP8 | MLA latent + sparse index | **不切**（複製） | 與 MLA 相當 |

> **注意**：「per-rank KV/token」是 TP=8 之後每個 rank 實際儲存的量。**MLA 雖然 per-rank 比 GQA 大，但模型總計算量與 attention 設計帶來的 prefill 節省更大**，HiCache 對 MLA 還有額外優化（見後）。

---

## 2. 為什麼 MLA 不切、要全 rank 複製？

### MLA 的核心：把 K/V 壓成共享的 latent 向量

MHA 每個 token 在每層存：
```
K = [k_1, k_2, ..., k_h]  共 h × d_k 維
V = [v_1, v_2, ..., v_h]  共 h × d_v 維
```
TP 把 head 切開很自然：rank i 拿走 (h/tp) 個 head 的 K/V。

MLA 改成：每個 token 只存一個 **共享 latent** `c_kv`（DeepSeek-V3 是 512 維）+ 一個小 RoPE 分量 `k_pe`（64 維）。Attention 算的時候才從 c_kv 投影出每個 head 的 K、V：
```
K_i = W_K_i @ c_kv     ← 每個 head 從同一個 latent 投影
V_i = W_V_i @ c_kv
```

### 結論：c_kv 不屬於任何 head，**沒有「按 head 切」的軸**

技術上你能沿 latent 維度切（512 → 64×8），但這會破壞 W_K_i / W_V_i 的乘法結構，且每層都要 all-gather c_kv，通訊成本反而比複製 70 KB/token 還貴。

DeepSeek 與 SGLang 都選擇：
- **KV cache 全 rank 複製**（每 rank 各存一份完整 c_kv + k_pe）
- **W_K_i / W_V_i / W_O_i 這些投影矩陣按 head 切**（各 rank 算自己負責的 head）
- **每層 attention 結束後 all-reduce 一次** output（這個本來 TP 就要做）

### 程式碼證據（SGLang）

MLA 的 per-token size：
```846:846:PR/sglang/python/sglang/srt/mem_cache/memory_pool_host.py
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num
```
`kv_cache_dim = kv_lora_rank + qk_rope_head_dim`，**沒有除以 tp_size、也沒有 num_kv_heads**。每個 rank 各自 allocate 完整大小。

對比 MHA：
```341:346:PR/sglang/python/sglang/srt/mem_cache/memory_pool_host.py
    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
```
`head_num` 已經是 **per-rank 的 KV head 數**（model runner 構造時已切過 TP），所以 MHA per-rank size 是 `1/tp` 的完整 KV。

---

## 3. 實際數字（per token, fp16/bf16）

| 模型 | 架構 | 全模型 KV/token | TP=8 per-rank KV/token |
|------|------|-----------------|------------------------|
| DeepSeek-V3/R1 (61 layers, lora=512, rope=64) | MLA | 70 KB | **70 KB（複製）** |
| GLM-5-FP8（MLA-style）| MLA + NSA | ~70–90 KB | **~70–90 KB（複製）** |
| Llama 3.1 70B (80 layers, 8 KV head, hd=128) | GQA | 320 KB | **40 KB** |
| Llama 3.1 405B (126 layers, 8 KV head, hd=128) | GQA | 500 KB | **63 KB** |
| Qwen3 32B (64 layers, 8 KV head, hd=128) | GQA | 256 KB | **32 KB** |
| Mistral 7B (32 layers, 8 KV head, hd=128) | GQA | 128 KB | **16 KB** |
| Llama 1 65B (80 layers, 64 KV head, hd=128) | 純 MHA | 2.6 MB | **330 KB** |

### 對 HiCache 的影響

- **`--hicache-size` 換算到 token 數**：用 per-rank KV/token 算。
  - 例如 DeepSeek-R1 上 `--hicache-size 100`（per rank 100 GB）：100e9 / 70e3 ≈ **1.43M tokens** of host cache
  - Llama 70B 上 `--hicache-size 100`：100e9 / 40e3 ≈ **2.5M tokens** of host cache
  - 同樣 GB，GQA 能裝更多 token

---

## 4. HiCache 對 MLA 的專屬優化

### Write-back（L2 → L3）只由 rank 0 執行

```494:512:PR/sglang/python/sglang/srt/managers/cache_controller.py
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
```

```1159:1164:PR/sglang/python/sglang/srt/managers/cache_controller.py
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)
```

效果：
- **MHA/GQA**：8 個 rank 各寫自己那 1/8 → L3 上拼成完整 1 份 KV（每 token 1 份完整資料）
- **MLA**：8 個 rank 都有完整 KV → 但只 rank 0 寫 → L3 上仍是 1 份（如果 8 rank 都寫會冗餘 8 份）

| 架構 | L3 上每 token 的資料 | rank 寫入次數 |
|------|---------------------|---------------|
| MHA / GQA | 1 份完整 KV（由 8 個 rank 拼出） | 8 |
| MLA | 1 份完整 latent（由 rank 0 獨自寫） | 1 |

### Prefetch（L3 → L2）：MLA **沒有**省

每個 rank 的 L2 都需要完整 KV，所以 8 個 rank 都要從 L3 讀同一份 70KB/token，aggregate read 頻寬 = 8 × 70 KB = **560 KB/token**。MHA/GQA 反而 aggregate read 較小（各 rank 讀自己 1/8）。

> **小結**：MLA 在 L3 寫入端省 8 倍，讀取端不省。

---

## 4.5 L3 與 L2 的關係（重點：不是嚴格 superset）

很多人直覺認為「L3 包含 L2 包含 L1」是強制成立的，**只有 L1 ⊆ L2 是強制的**（write-through 機制），L3 ⊆ 或 ⊇ L2 都不一定。

### L1 ⊆ L2 是強制保證

HiCache 對每個 device slot 都會在 host 預留對應 slot（write-through 到 L2），所以 L1 一定是 L2 的子集。這也是為什麼 `prefetch_capacity_limit = 0.8 × (L2 − L1)` 的公式成立。

### L3 跟 L2 的關係完全看 write policy

| Policy | L1→L2 | L2→L3 | L3 ⊇ L2？ |
|--------|-------|-------|-----------|
| `write_through` | 立即 | 立即 | **近似 ⊇**（但 L3 有自己的 eviction） |
| `write_through_selective` | 立即 | 命中次數 ≥ 2 才寫 | ✗（L3 只有「熱點」的子集） |
| `write_back` | 立即 | 只在 L2 evict 時寫 | ✗（L3 是 L2 evict 出來的「殘留」） |

三種模式的關係示意：

```
write_through:           L1 ⊆ L2 ⊆ L3 (近似，L3 自己會 evict)

write_through_selective: L1 ⊆ L2
                          ⊇ hot subset → L3

write_back:              L1 ⊆ L2
                              ↓ 只在 evict 時才寫
                              L3
```

### L3 還可能有「其他 instance 寫進去」的資料

L3 是 **cluster-shared**，所以你的 L2 沒有的資料，可能在 L3：

```75:75:PR/sglang/docs/advanced_features/hicache_design.md
**Cross-instance Sharing**: When data is written back from L2 to L3, only data not already present in L3 is transferred. KV caches stored in L3 can then be shared across all SGLang instances in the cluster (depending on the L3 backend implementation), significantly improving cache hit rates within the same memory budget.
```

所以「L3 ⊇ L2」跟「L2 ⊇ L3」都不一定成立，它們是**不同 scope 的快取層**：
- L1, L2 = instance-private（這個推理實例專有）
- L3 = cluster-shared（叢集共享）

---

## 5. HiCache 完整流程（L1 → L2 → L3）

```
請求進來
   ↓
[1] Local match：查 HiRadixTree
   找出已經在 L1（GPU）+ L2（host RAM）的 prefix tokens
   ↓
[2] 對 L1+L2 都沒有的部分，去 L3 metadata 查還有多少能拿
   storage_hit_count = L3 命中的 token 數
   ↓
[3] 觸發決策：storage_hit_count < prefetch_threshold？
       │
       ├─ 是 → REVOKE prefetch
       │       不去 L3，這部分等等讓 GPU 重算
       │       （理由：避免短命中也付 fixed overhead）
       │
       └─ 否 → 啟動 prefetch（L3 → L2）
               同時 GPU 看 prefetch policy 決定何時開工：
                  ├─ best_effort: GPU 隨時開工，沒拉到就重算
                  ├─ wait_complete: 等 prefetch 全完成才開工
                  └─ timeout: 等到 timeout 或完成才開工
   ↓
[4] Prefill 開始
       對每個 token，需要的 K/V：
         ├─ 在 L1：直接用（GPU 內，5+ TB/s）
         ├─ 在 L2：用 cudaMemcpyAsync / GPU-assisted kernel 搬 GPU
         │         （PCIe 50–100 GB/s，可與 prefill 重疊）
         └─ 都沒有：GPU 重算 Q/K/V projection + FFN
                   （attention 仍然在所有 token 上計算）
   ↓
[5] Decode 階段：新生成的 token
       └─ Write-through 把 K/V 寫回 L2（之後也可能進 L3）

═════════════════════════════════════════════════════════
速度與大小階層：
═════════════════════════════════════════════════════════

   ┌──────────────────────────────────────────────────┐
   │ L1 = GPU HBM       5–8 TB/s     ~10s GB    最快  │
   │ ──────────────  Direct / Kernel I/O ──────────── │
   │ L2 = Host RAM      ~100 GB/s    ~100s GB         │
   │ ──────────────  RDMA / SSD I/O ───────────────── │
   │ L3 = External      RDMA: 25–200 GB/s   TB+       │
   │      Storage       SSD:  7–15 GB/s     最大      │
   └──────────────────────────────────────────────────┘

   單機（per inference instance）私有：L1, L2
   叢集（cross-instance）共享：L3
```

---

## 5.5 Prefetch 與 prefill 的時間關係（有 dependency）

很多人會疑問：「prefetch 從 L3 載入 25K cached KV，跟 prefill 新的 5K tokens，是不是要等 prefetch 完成才能 prefill？」**對，有 dependency**，但具體看 policy：

### 場景：30K prompt，其中 25K 已在 L3、5K 是新的

#### `wait_complete` policy：完全 sequential
```
L3 → L2 prefetch    ████████████        100ms（拉完 25K）
                                 │
                                 ↓ dependency: 必須等 KV 在 L2 才能開始
GPU extend prefill              ████░   30ms（算 5K 新 token + attention）
                                       ↑ 完成
總時間 = 100 + 30 = 130ms
```

#### `best_effort` policy：GPU 不等，部分 overlap
```
L3 → L2 prefetch    ████████████        100ms（被中斷）
GPU 馬上開工           ░░░░░░░░░░░░░░    150ms（重算 22K + extend）
                                       ↑ 完成
總時間 ≈ 150ms（best_effort 等不到，回退到部分重算）
```

#### 對照：完全沒開 L3
```
GPU full prefill 30K  ████████████████████ 200ms
                                          ↑ 完成
```

### 重點：sequential 並不代表「沒用」

即使 wait_complete 是 100+30=130ms，**仍比沒 L3 的 200ms 快 35%**。Prefetch 換來的好處是：
- 省下對 25K cached tokens 的 **FFN 計算**（O(N × d²)）
- 省下對 25K cached tokens 的 **Q/K/V projection**（O(N × d²)）
- Attention 從 O(30K²) = 900M ops 變成 O(5K × 30K) = 150M ops（**6 倍少**）

### L2 → GPU 在 prefill 內是 layer-by-layer overlap

L3 → L2 通常是「prefill 前」一次性完成（因為 L3 慢），但 **L2 → GPU 的 25K KV 載入是 layer-by-layer**，與 prefill 的 layer N 計算重疊：

```5:97:PR/sglang/docs/advanced_features/hicache_design.md
* **Compute-Transfer Overlap**: During the prefill phase, when transferring data from CPU to GPU, HiCache overlaps layers by concurrently loading the KV cache of layer N+1 while computing layer N. This effectively hides data transfer latency.
```

所以 prefill 時間裡，L2 → GPU 的開銷已經被 hide 掉，extend prefill 的時間主要是 compute。

### 「重算 22K」是什麼意思？

回到 best_effort 那個例子：30K prompt，L3 只來得及拉 8K（其餘 22K 沒拉到）。

| 部分 | 在哪裡 | GPU 做什麼 |
|------|--------|-----------|
| 8K（已從 L3 拉到 L2） | L2 → 載入 GPU | 用，不重算 K/V |
| 22K（沒拉到） | 不在 L2 | 重算 K/V projection + FFN |
| 5K（新 token） | 不在任何 cache | 算 K/V + Q + FFN |

**注意：Attention 仍然在 30K 上算**（27K Q × 30K K/V，因為新算的 22K + 5K 是 27K 新的 Q）。「重算 22K」省略的是「**K/V projection + FFN 那部分**」，不是 attention 規模。

完整 prefill 30K（沒 L3）：
- Q/K/V projection: 30K × d²
- FFN: 30K × d_ff × d
- Attention: 30K × 30K

「重算 22K」+「8K 從 L2 載入」+「5K 新」：
- Q/K/V projection: 27K × d²（少 3K，省 10%）
- FFN: 27K × d_ff × d（少 3K，省 10%）
- Attention: 30K × 27K（少 10%，因為 Q 維度只有 27K）

如果 L3 拉到 25K 而不只 8K：
- Q/K/V projection 跟 FFN 只算 5K（**省 83%**）
- Attention: 30K × 5K（**省 83%**）

所以**「拉到越多 cached token、省越多 prefill 計算」**，這就是 L3 prefetch 的真正價值。

---

## 6. Write-back / Prefetch policy 推薦

### Write-back 三種模式

| Policy | L1 → L2 | L2 → L3 | 適合場景 |
|--------|---------|---------|----------|
| `write_through` | 立即 | 立即 | L3 帶寬充足、cache hit rate 優先 |
| `write_through_selective`（**預設**） | 立即 | 命中次數 ≥ 2 才寫 | 多數場景；只備份 hot data |
| `write_back` | 立即 | 只在 L2 evict 時寫 | L3 寫頻寬受限 |

對應的 hit count 邏輯：
```176:177:PR/sglang/python/sglang/srt/mem_cache/hiradix_cache.py
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
```

### Prefetch 三種終止策略

| Policy | 行為 | 適合場景 |
|--------|------|----------|
| `best_effort` | GPU 隨時開工，沒拉到的重算 | TTFT 極端敏感 |
| `wait_complete` | 等 prefetch 全完成 | 最高命中率優先 |
| `timeout`（**production 推薦**） | 設定時間或完成 | SLO 平衡 |

```bash
# 預設值（可用 --hicache-storage-backend-extra-config 覆寫）
prefetch_threshold = 256                   # tokens
prefetch_timeout_base = 1                  # seconds
prefetch_timeout_per_ki_token = 0.25       # seconds per 1K tokens
```

---

## 7. 各架構建議配置

> **threshold 主要由 L3 backend 速度決定，不由架構決定**。下表只列出對該架構特別有意義的調整。

### MLA（DeepSeek-R1/V3、Kimi-K2、GLM-5）

```bash
--enable-hierarchical-cache
--hicache-ratio 2                      # 起點，per-rank L2 = 2× L1
--hicache-size 0                       # 0 = 用 ratio；想固定就設 GB
--hicache-write-policy write_through_selective
--hicache-storage-prefetch-policy timeout
--hicache-storage-backend mooncake     # 或 hf3fs / nixl / file
```

理由：
- 每 rank KV 較大（~70 KB/token，因為複製），L2 不能太小
- L3 寫入成本只是 1×（不是 8×），所以 `write_through_selective` 或 `write_through` 都很划算
- Prefill 計算量大（attention + dense FFN 部分 + MoE expert），L3 prefetch 省下的時間多 → 適合 `timeout` 或 `wait_complete`

### GQA（Llama 3 70B/405B、Qwen3 系列）

```bash
--enable-hierarchical-cache
--hicache-ratio 3                      # GQA per-rank KV 小，L2 可以大
--hicache-size 0
--hicache-write-policy write_through_selective
--hicache-storage-prefetch-policy timeout
--hicache-storage-backend mooncake
```

理由：
- per-rank KV 小（~30–60 KB/token），同樣 GB 能裝更多 token、命中率更高
- L3 寫入是 8 個 rank 各寫 1/8（沒有 MLA 的 once-only 優化），適合 `write_through_selective` 控制 IO 量
- 短 prompt + 大 batch 的場景（chatbot、code completion）特別吃 L2/L3 容量

### MQA（少見大型模型）

策略類似 MLA（KV 也是複製、每 rank 完整一份），但 KV 更小，L2/L3 壓力低。

### 純 MHA（極少現代模型用）

per-rank KV 非常大（~330 KB/token），L2 容量壓力高。建議：
- `--hicache-ratio` 不要拉太高
- 用 `write_back` 而非 `write_through`，避免 L3 IO 過重
- prefetch 只在長 prompt 才有效，threshold 可能要拉到 512+

---

## 7.5 多 node 共享 L3 何時真的有用？常見誤解

### 誤解 1：Mooncake = 一台「專用 cache 機」

**完全相反**。Mooncake / 3FS / NIXL 是「**所有 serving node 都跑的 daemon**」，每台機器**同時做兩件事**：
1. 用自己的 GPU 跑推理 serving
2. 把自己「閒置的 RAM」貢獻給整個 cluster 的共享 cache pool

```
┌─────────────────────────────────────────────────┐
│   Cluster (4 nodes, each with 8x MI355X)        │
│                                                 │
│  Node A          Node B          Node C         │
│  ┌──────┐       ┌──────┐        ┌──────┐        │
│  │8 GPU │       │8 GPU │        │8 GPU │ ← 都在 serving │
│  │ ↕    │       │ ↕    │        │ ↕    │       │
│  │1.5 TB│       │1.5 TB│        │1.5 TB│ ← 閒置 RAM  │
│  │ RAM  │       │ RAM  │        │ RAM  │       │
│  └──┬───┘       └──┬───┘        └──┬───┘       │
│     └──── RDMA ────┴──────────────┘            │
│            ↓                                    │
│     共享 cache pool ~5 TB                       │
└─────────────────────────────────────────────────┘
```

每台機器都繼續用全部 GPU 做推理，**沒有任何節點被「廢掉」**。RAM 只是「順便」貢獻一部分當共享 cache。

### 誤解 2：「每台獨立 serving」會比共享 L3 好

這個直覺**只在一種情況成立**：每台的 L2 + 路由策略已經能讓「同個 prefix 的 request 都落在同一台」。

#### 場景：8 個 node、共用 5K system prompt、每個 request 帶 1K 個別內容、共 10000 個 request

**方案 A（每台獨立 serving、無共享 L3）**：

假設 router 做 sticky routing 或 round-robin。每台分到 1250 個 request：
- 第 1 個 request：5K（system prompt）+ 1K（user）= 6K prefill，system prompt 寫入該 node 自己的 L2
- 後續 1249 個：system prompt 在 L2 → 命中，只 prefill 1K
- 每台 prefill 計算量 = 6K + 1249 × 1K ≈ **1.26M tokens**
- 全 cluster prefill ≈ 8 × 1.26M ≈ **10M tokens**

**方案 B（共享 L3）**：

第 1 個 request 寫 5K system prompt 到 L3（一份）。後續 9999 個全部 L3 命中（不管落在哪台 node）：
- 全 cluster prefill ≈ 5K + 10000 × 1K ≈ **10M tokens**

**結論：在這個 simple workload，兩者差不多**。L2 已經夠了。

### L3 共享真正贏的場景

#### 場景 1：Router 做 load balancing，prefix 不一定落在同一台

```
User A 的 conversation history（50K tokens）
  ↓
  Turn 1 落在 Node 1 → prefill 50K，Node 1 的 L2 緩存
  Turn 2 落在 Node 5 → Node 5 的 L2 沒這資料！
                       ├─ 無 L3：重算 50K
                       └─ 有 L3：從 L3 抓 50K
```

如果這種 cross-node routing 很常見（multi-turn chatbot 用 round-robin），L3 share **省下重複 prefill**，省的量可能是數十倍。

#### 場景 2：L2 容量不夠裝所有 hot prefix

- 假設 hot prefix 累積有 500GB
- 每台 node 的 L2 只配 100GB
- 無 L3：每台只能 cache 一小部分 → eviction 頻繁、hit rate 低
- 有 L3（10TB pool）：所有 prefix 都能進 L3 → hit rate 高很多

#### 場景 3：Node 重啟 / scale-up

- Node 重啟：L1 + L2 全清空
- 無 L3：要花很久重新「暖機」（重算所有熱點 prefix）
- 有 L3：新啟動的 node 從 L3 暖機，幾分鐘就回到 hit rate 高的狀態

#### 場景 4：Multi-tenant / RAG 共用文件

- 多個 user 的 RAG context 重疊（例如同公司內部文件）
- 沒共享：每台 node 自己 cache 一份這些文件
- 共享 L3：只存一份，所有 node 都能用，**省下 N × 重複儲存**

### 結論：用不用 L3 取決於 workload

| Workload 特性 | 該不該開 L3 |
|---------------|-------------|
| 單機部署、無跨機通訊 | 不需要 |
| 多機部署 + sticky routing + 共用 prefix 集中 | L2 已夠，**L3 邊際效益小** |
| 多機部署 + load balancing 散 routing | **需要 L3**（cross-node hit） |
| L2 容量裝不下所有 hot prefix | **需要 L3**（容量擴展） |
| 多 tenant / RAG / 動態文件 | **需要 L3**（共享避免冗餘） |
| 頻繁 scale-up/down、希望 warm restart | **需要 L3**（持久化暖機） |

---

## 8. 實務 sanity check checklist

部署前快速確認：

- [ ] `--hicache-size` × tp_size ≤ 機器可用 RAM 的 70%（留 OS / 其他 process）
- [ ] L3 backend 對應的傳輸頻寬已知（SSD ≈ 7–15 GB/s vs RDMA ≈ 25–200 GB/s）
- [ ] `--page-size` 對齊 backend 的 IO granularity（一般 64 是好起點）
- [ ] 開 `--enable-cache-report --enable-metrics` 看實際 hit rate / revoke rate / TTFT
- [ ] 對 MLA：確認看到「rank 0 寫 L3、其他 rank skip」的行為（log 應該明顯）
- [ ] 對 GQA：看 L3 用量是否 ≈ tp_size 個 rank 各 1/tp 的總和

調整方向（觀察後）：
- 命中率低 → 加大 `--hicache-size` 或加大 `--hicache-ratio`，或檢查 L3 是否有清空
- TTFT 拉長太多 → policy 從 `wait_complete` 改 `timeout` 或 `best_effort`
- Revoke 太多（短 prefix 浪費 prefetch） → 拉高 `prefetch_threshold`
- L3 寫入頻寬打滿 → write policy 改 `write_through_selective` 或 `write_back`

---

## 9. 快速參考（cheat sheet）

```
                           ┌───────────────────────────────────────┐
                           │  各 attention 架構在 HiCache 的處理  │
                           └───────────────────────────────────────┘

MLA / MQA  ────► KV 複製到所有 rank
                 ├─ L1/L2 大小：per-rank = 完整 KV（不縮）
                 ├─ L3 寫入：只 rank 0 寫（1 份）  ◄── 專屬優化
                 ├─ L3 讀取：每 rank 都讀（aggregate 8×）
                 └─ Policy 建議：write_through_selective + timeout

MHA / GQA  ────► KV 按 head 切到各 rank
                 ├─ L1/L2 大小：per-rank = 完整 KV / tp_size
                 ├─ L3 寫入：每 rank 寫自己 1/tp（拼成 1 份）
                 ├─ L3 讀取：每 rank 讀自己 1/tp（aggregate 1×）
                 └─ Policy 建議：write_through_selective + timeout
                                 GQA 可拉高 hicache-ratio（KV 小）

通用原則
─────────────
- threshold 由 backend 速度決定，不由架構決定
- best_effort  = 安全（不會比沒 L3 慢）但長 prompt 不最優
- wait_complete = 長 prompt 最優，短 prompt 浪費等待
- timeout       = production 推薦折衷
```

---

## 參考程式碼位置

- Per-token KV size 計算：`PR/sglang/python/sglang/srt/mem_cache/memory_pool_host.py`
  - MLA：line 839–846 (`MLATokenToKVPoolHost.get_size_per_token`)
  - MHA：line 341–346 (`MHATokenToKVPoolHost.get_size_per_token`)
- MLA write-back 優化：`PR/sglang/python/sglang/srt/managers/cache_controller.py` line 494–499, 1159–1164
- Prefetch threshold 邏輯：line 512、1033–1052
- Write policy 邏輯：`PR/sglang/python/sglang/srt/mem_cache/hiradix_cache.py` line 176–177、729、875–897
- 官方文件：`PR/sglang/docs/advanced_features/hicache_design.md`、`hicache_best_practices.md`
