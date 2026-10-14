# MTT S5000 GroupGEMM：FlagTree Triton 内核 vs MATE 官方 API —— 汇编级性能差距分析

日期：2026-10-14　|　环境：MTT S5000 (Mp31, 60 MPC) / MUSA 4.3.5 / flagtree 0.7.0+mthreads3.6 / MATE 0.2.0
对象：BF16 GroupGEMM（`benchmark/test_group_bfgemm_perf.py` 30 shapes），我方 = TLE warp-specialize 内核（tle_ws 家族），对方 = `mate.gemm.ragged_m_moe_gemm_16bit`（libmudnn_xmma 手写汇编内核）

---

## 0. 一页结论

| 维度 | FlagTree Triton 内核（我方） | MATE 官方 API（对方） | 差距来源 |
|---|---|---|---|
| **MMA 指令** | `TCE.SQMMA.M128N128K32/64`（经 TLE wgmma 降级，TMEM 硬件累加） | 同族指令（手写 asm 直排） | 指令相同 |
| **计算流水** | 每 k 步 1 mma + 1 wait（1:1 排空），**无指令积压** | 长链 mma 背靠背 + 稀疏 wait（指令积压），mma 单元不饿 | **主要差距 #1** |
| **加载异步** | TME 异步拷贝，6 级槽位管 ✓ | TME 异步 ✓（smem=128KB 元数据） | 持平 |
| **加载→计算重叠** | warp 特化 producer/consumer ✓ | ✓ | 持平 |
| **C 写回重叠** | **每 CTA k 循环完成后串行 unpack+store** | 持久内核边算下个 tile 边写 C | **主要差距 #2** |
| **B 矩阵流量** | BM=256 → 全量 B（shape0: 6.25GB） | 512 行面板 → B 减半（3.12GB） | **主要差距 #3** |
| **实测** | 计算受限段 310-315 TFLOPS（dense 8K-K 探针 342） | 基准面 ~349 TFLOPS | 综合以上三条 |

---

## 1. 指令层证据

### 1.1 我方内核的真实指令流（.llir，编译器后端输入）

来源：tle_ws 最优配置内核（BK=32/S6），`TRITON_CACHE_DIR` 产物 `pp_gemm_kernel.llir`。

```llvm
; 累加器以 <128 x i32> 载具(loop-carried phi)全程驻留 TMEM，循环内零 TMEM↔寄存器往返：
%55 = phi <128 x i32> [ zeroinitializer, %.thread ], [ %67, %54 ]

; 每 k 步：构造 2 个 smem 描述符 + 1 条 mma + 1 条 wait —— 1:1 排空节奏，无积压：
%64 = tail call i32 @llvm.musa.sqmma.desc.half(i16 %63, i32 64, i32 1, i32 3)
%66 = tail call i32 @llvm.musa.sqmma.desc.half(i16 %65, i32 256, i32 2, i32 3)
%67 = tail call <128 x i32> @llvm.musa.sqmma.bfmma.m128n128k32.mma(i32 %64, i32 %66, <128 x i32> %55, ...)
tail call void @llvm.musa.sqmma.wait()
```

关键事实：
- **mma:wait = 1:1**。每条 mma 发射后立即排空，SQMMA 队列深度恒为 1，mma 单元在排空尾部 + 下一次发射之间空转；
- 循环内 `sqmma.ld/sqmma.st`（TMEM 读写）计数 = **0** —— 累加器常驻 TMEM 这点我方与官方一致，不是差距点。

### 1.2 官方编程模型的汇编形态（mcc -S，MUSA C++ API 同路径）

用 `crt/sqmma.h` 的官方 `mma_sync`（内部 = 裸 `__musa_sqmma_m*_mma()`，**不含 wait**）编译：

```asm
; 两条 mma 背靠背，中间没有任何 wait/sync —— 硬件记分板自动链接 R128：
TCE.SQMMA.M128N128K64 R128[8:16][16:1].F32, SL3.x0y0.BF16.ROW, SL2.x0y0.BF16.COL, _, _
TCE.SQMMA.M128N128K64 R128[8:16][16:1].F32, SL3.x0y0.BF16.ROW, SL2.x0y0.BF16.COL, _, _
LSU.ST.B32 R128, ...        ; 消费者读 R128 —— 记分板自动停等，无需显式 wait
```

关键事实（决定性）：
- `TCE.SQMMA` 的**累加器目标是记分板寄存器 R128（TEMP/TMEM）**，指令中不编码累加器地址（隐式 TMEM）——mma 天然可背靠背链接；
- `__musa_sqmma_wait()` 在存在数据依赖（读 R128）时**编译为零指令**——官方模型里 wait 只为"槽位回收安全"服务，不为排序服务；
- **官方 API 的意图就是"连发 N 条 mma + 稀疏 wait"**（`mma_sync` 无内部 wait 是官方头文件设计）。MATE 349 的指令积压 = 官方支持用法。

指令编码（受控差分实测）：`TCE.SQMMA.M128N128K64 = 0x4b810874`、`K32 = 0x0b810874`（仅 bit30 为 K 字段）。完整形状矩阵 `M∈{16..128}×N∈{32..128}×K∈{16,32,64,128}`（IntrinsicsMTGPU.h）。

### 1.3 MATE 内核二进制现状（为什么不能逐指令对比）

libmudnn_xmma.so（1.45GB）内含 25 个大 mtgpu ELF（最大 168MB），**全部 strip**，符号与指令扫描均无法定位目标内核；仅元数据可解：目标 bf16 内核 ~10.5MB 全展开直排码，smem=128KB、shared_reg=167、slot_reg=16、temp_reg=160。工具链无公开 mtgpu 反汇编器（llvm-objdump 全 `<unknown>`）。因此对方证据采用"官方 API 汇编 + 元数据 + 性能反推"三角定位。

---

## 2. 流水（pipelining）对比

| 层次 | 我方 | 对方 | 证据 |
|---|---|---|---|
| TME 加载管 | **6 级**异步槽位（BK=32，实测最优）✓ | smem 128KB ≈ 2 级大槽 | 我方 .llir + smem 预算；对方元数据 |
| 累加器驻留 | TMEM 载具，零往返 ✓ | TMEM（temp_reg=160） | 我方 .llir phi；对方元数据 |
| **MMA 指令积压** | **无**（1:1 wait 排空） | **有**（背靠背 + 稀疏 wait） | 我方 .llir 1:1；对方 = 官方模型 + 349>310 反推 |

我方无法形成积压的三重原因（全部实测）：
1. 硬件无 `wait(N)`（`__musa_sqmma_wait` 无参数，BuiltinsMTGPU.def:2411）；
2. Triton 层等价写法"攒 D 条再 wait"（mode7-DRAIN）**能编译、结果正确、IR 干净**（mma×D : wait=1），但 D≥2 被 flagtree 后端插入 ConvertLayout TMEM 往返，**实测慢 20-35%**；
3. 纯异步循环（循环内零 wait、末尾一次）在 LowerSqmma 崩溃（yield 使用未重写）。

积压缺失的量化：我方所有变型钉在 310-315 TFLOPS（12 种变体实测），MATE 349 —— 差 ~12% 即排空空转。

---

## 3. 异步（async）对比

| 机制 | 我方 | 对方 |
|---|---|---|
| TME 异步 global→smem | ✓（tle.gpu.copy，无 barrier 依赖） | ✓ |
| mma 异步发射 | ✓（TLE wgmma，硬件记分板） | ✓ |
| mbarrier 循环复用 | **✗ 单相位**（第 1 次相位翻转即算错，mini_mb.py 实测边界） | ✓（手写 asm 直控） |
| tle.pipe 嵌套循环（持久化） | **✗**（扁平单循环 only：嵌套=确定性 mbarrier 违例或间歇性竞争） | n/a（手写） |
| 持久 tile 循环 + C 写回重叠 | **✗**（见 §4） | ✓ |

---

## 4. 缓存与流量（实测，方法：解析流量模型 ÷ torch.musa.Event 中位时延）

> 硬件计数器路径：MUPTI（`/usr/local/musa/include/mupti_*.h` + `libmupti.so`）可用但无现成 CLI 工具，需自写采集程序；本文数据用"解析流量 ÷ 实测时间 = 有效带宽"方法，脚本见文末。

### 4.1 DRAM 流量对比（无 L2 折抵的上界模型）

| shape | 我方流量 (BM=256) | MATE 风格流量 (BM=512 行面板) | 我方有效带宽 | MATE 有效带宽 |
|---|---|---|---|---|
| shape0 (K2048/E128/N1536/M254K) | A 1.04 + **B 6.25** + C 0.78 = **8.07GB** | A 1.04 + **B 3.12** + C 0.78 = **4.94GB** | 8.07/5.29ms = **1.53TB/s** | 4.94/4.49ms = **1.10TB/s** |

**我方带宽更高但仍输 15%**：对方少搬 39% 的数据。B 流量 ∝ 行面板宽度，我方 BM=512 被 TMEM 累加器上限（单 acc ≤ 256×256）挡死。

### 4.2 L2 复用（我方可用的补偿手段，实测数据）

PANEL_WIDTH（N 列面板，A 流量 ÷pw 的 L2 复用）逐形状实测：
- s15 (N=4096, A 占比高)：pw2→pw16 = **0.883 → 0.917**；
- s2 (K768)：pw2→pw4 = 0.897 → **0.941**；
- s16 (E=16)：pw2→pw8 = 0.896 → 0.90-0.916。

### 4.3 K 无关固定成本分解（边际 K 测量，s27 同构 K∈{512,1024,2048}）

| | 固定成本（截距） | 边际成本（斜率） |
|---|---|---|
| 我方 | **0.98ms** | 2.94 µs/K |
| MATE | **0.38ms** | 2.82 µs/K |

边际成本几乎相同（mma 效率差距在 §2），**固定成本差 0.6ms ≈ C 写回（0.99GB @1.5TB/s ≈ 0.66ms）未与计算重叠**：对方持久内核边算下个 tile 边写 C；我方每 CTA k 循环后串行 unpack+store。此为我方在短 K/多 tile 形状（0.67-0.78）落后的主因；持久化在 flagtree 当前实现下不可用（§3 表）。

---

## 5. 性能汇总（iters=20 中位，30 shapes 中 ≥0.9x 的 10 个 + 0.89 线 1 个）

| shape | K,E,N | 配置 | speedup |
|---|---|---|---|
| 23 | 128,512,2048 | bk32/s6/pw2 | 1.003 |
| 25 | 64,512,2048 | bk32/s6/pw2 | 0.989 |
| 22 | 2048,512,256 | bk32/s6/pw2 | 0.988 |
| 24 | 2048,512,128 | bn128/bk64/s4/pw1 | 0.955 |
| 2 | 768,128,2048 | bk32/s6/pw4 | 0.941 |
| 20 | 4096,17,7168 | bk32/s6/pw2 | 0.935 |
| 18 | 4096,16,7168 | bk32/s6/pw2 | 0.919 |
| 15 | 192,128,4096 | bk32/s6/pw16 | 0.917 |
| 5 | 192,128,2048 | bk32/s6/pw2 | 0.904 |
| 16 | 4096,16,3072 | bk32/s4/pw8 | ~0.90 |
| 26 | 1024,128,1024 | bk32/s5/pw4 | 0.899 |

剩余差距归因：短 K/多 tile 簇（0.67-0.78）= §4.3 C 写回串行 + §4.1 B 流量；计算受限段 = §2 指令积压。

---

## 6. 复现

```bash
cd /root/ywb/FlagBLAS && source tools/set-env.sh mthreads

# ① 我方内核指令流（mma:wait 1:1 + TMEM 载具 phi）
TRITON_CACHE_DIR=/tmp/tc python3 /root/ywb/hopper_gg/pipe_probe.py --mode 7 --drain 1 --bk 32 --stages 6
grep -nE "phi <128|sqmma\.(bfmma|wait)" /tmp/tc/*/*.llir | head

# ② 官方 API 汇编（背靠背 mma、零 wait）
/usr/local/musa/bin/clang++ -x musa -mtgpu --cuda-gpu-arch=mp_31 -S /root/ywb/hopper_gg/enc/w_yes.cu -o - | grep TCE.SQMMA

# ③ 单相位 barrier 边界（KT≤NS 对 / 翻转即错）
python3 /root/ywb/hopper_gg/mini_mb.py 4 4   # correct
python3 /root/ywb/hopper_gg/mini_mb.py 8 4   # wrong

# ④ 逐形状基准（vs MATE，含正确性校验）
python3 /root/ywb/hopper_gg/triton_gg.py --only 0 --bk 32 --stages 6 --pw 16 --iters 20

# ⑤ 流量/带宽模型
python3 /root/ywb/FlagBLAS/docs/tools/groupgemm_asm_gap_repro.sh traffic 2048 128 1536 254336
```

原始数据与完整实验日志：`/root/ywb/hopper_gg/README.md`（第 5-9 轮 + 附节）。

---

## 7. 增补（2026-10-14，第 10 轮）：三条结论的修正与落地结果

在"仅 FlagTree-Triton"约束下继续优化后的关键修正：

1. **§2 的"指令积压"归因被推翻**。flagtree 的 `pipelineSqmma` pass 对 tl.dot 路径自动做
   MATE 式稀疏 wait（缓存 ttgir 实证：循环内 isAsync sqmma + TMEM carrier、循环外单 wait），
   且 TLE 路径用 split-accumulator 实现 2:1 mma:wait（carrier 契约：`yielded=wait(dot)` 且
   `dot.getC()==iterArg`，纯语法匹配）dense 实测**零增益**（187.2 vs 186.3 TF）。
   wait 节奏不是 310 vs 349 的差距来源。
2. **tl.dot 路径封顶 ~150 TF 的真因**：非 TLE 的 TME load lowering 在每个
   `async_tme_copy` 前无条件插入 `llvm.musa.barrier0`（仅 TLE 显式完成属性可跳过），
   全 CTA 每拍 lockstep 队拍。BK=32（迭代数×2）→ 92 TF，BK=64 → 140 TF，与迭代数成反比。
   `tl.load`(LSU) 路径同样 ~129 TF。持久化 tl.dot 内核因此无法承接计算受限 shape。
3. **§4.3 的 C 写回固定成本**依然无解（TLE 嵌套循环被 TLELowerPipe 直接拒绝；
   manual mbarrier 单相位），短 K 簇（K∈{384,512,768}, N=2048）停在 0.77-0.84。

落地（全部 FlagTree-Triton，无 C 内核）：E=512 且 K>64、N%256==0 的 shape 改路由 tle_ws
（s23 0.958→1.032）；n192 内核 PIPELINE_STAGES 6→5（s6 0.864→0.923）；gs≥256 且 K>512
时 C 写加 evict_first（s22 +0.026）；prune 规则按 shape 簇扩充；修复两个生产 bug
（pack_a int32 地址溢出——M×K>2^31 时非法访存，reroute 前为潜伏 bug；gs=512+stages=6
大 M 间歇挂死——以 (2,4) 限制规避）。

最终结果：30/30 SUCCESS，≥0.9x 的 shape 数 12→**15-17**（run 间有方差），
geomean ~0.92。未达标项集中在短 K 簇（结构性，需 FlagTree 修复嵌套 pipe/mbarrier
或提供 wgmma_wait(N)）与若干 0.87-0.90 临界 shape（310 vs 349 的稳态差距）。
完整实验记录：/root/ywb/hopper_gg/README.md 第 10 轮。

## 8. 第 11 轮（2026-10-14 晚）：epilogue 标量 store 实锤 + pack_a 消除

- **C 写回 0.76ms/2.53ms（30%）是短 K 簇的全部差距**：no-store 运行 1.772ms
  已与 MATE 全程（1.68ms）持平。llir 证据：tle_ws 的 C store 全是标量
  `store bfloat`（#mma musa_sqmma 布局直达 tt.store），对照组 tl.dot 路径为
  `store <8 x bfloat>`。n192 内核同样标量。
- 8 条修复路线逐一排除（set_layout staging OOR / TME store 分区 barrier0 /
  default 分区 drain 被禁 / 手工 mbarrier 与 bar.record 撞号 / debug_barrier
  非 syncthreads.lm / smem 中转 bank 冲突净负 / u32 配对打包净负 /
  tile-pair·双 pipe·双循环相位不支持），详见 /root/ywb/hopper_gg/README.md 第 11 轮。
- **pack_a 消除落地**：nopack TLE WS 内核（A 原位 TME 读）路由
  K∈{384,512,768}&N=2048，簇内 0.759-0.861（原 0.74-0.80），pytest 30 passed。
  OOB 防护：load_base=min(row_start, M-256) + 下移掩码 store 窗口。
- 描述符陷阱：ragged A 的 TensorDescriptor 必须在构造时给真实 block
  （dummy_block + 事后改 .block_shape 会在 TME tail-divisibility 上全错）。
- MATE 自身在 E=512 组崩溃（与 FlagBLAS 无关，控制实验证实），
  完整 30-shape benchmark 暂不可跑。
