# patches / flagtree

FlagBLAS 依赖的 Triton 定制版（flagtree 0.6.1+hcu3.6）以 pip 包形式安装在
site-packages，不属于 FlagBLAS git 仓库。对 Triton 源码的定制改动以 patch 形式
固化在此目录，部署/重建环境时重新应用。

## flagtree-0.6.1-compiler-hcu-per-kernel-itt.patch

**用途**：为 Triton HIPOptions 新增 `use_in_thread_transpose` 编译选项，使
sgemm_nn 可按 shape 控制 in-thread-transpose（ITT）pass 开关（v1.2 per-kernel
ITT 优化，见 benchmark/sgemm_nn_perf_regression_report.md §3.4）。

**目标文件**：`triton/backends/hcu/compiler_hcu.py`（flagtree 0.6.1+hcu3.6）

**应用**（在 flagtree 安装目录下）：

```bash
cd /data/zhangbin/py310/lib/python3.10/site-packages/triton
patch -p1 < /data/zhangbin/cicd/FlagBLAS/patches/flagtree-0.6.1-compiler-hcu-per-kernel-itt.patch
```

**验证**：`grep use_in_thread_transpose backends/hcu/compiler_hcu.py` 应命中
HIPOptions 字段与 compile 中 `itt_enabled = options.use_in_thread_transpose`。

**注意**：该改动会改变 Triton `cache_key`（HIPOptions 字段增加），导致
`/root/.flagblas/config_cache/` 下 config DB 表名漂移；首次运行会重建新表
（运行时自动 benchmark 填充，37 个核心 shape 的 config 值与原表一致）。
