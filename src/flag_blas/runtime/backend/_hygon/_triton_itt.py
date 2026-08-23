"""FlagBLAS 侧对 Triton HCU 后端 per-kernel ITT 编译选项的运行时注入。

背景
----
Triton (flagtree 0.6.1+hcu3.6) 的 in-thread-transpose (ITT) pass 开关由全局 knob
``knobs.hcu.use_in_thread_transpose`` 控制（env ``TRITON_HIP_USE_IN_THREAD_TRANSPOSE``，
动态读取），仅支持进程级开关。Hygon gfx936 上 ITT 对 kernel 的影响按 shape 分化
（大/方 shape 受益、瘦/小 shape 受损），需要 per-kernel 控制。

方案
----
不改 site-packages 源码，在 FlagBLAS 导入时做两处运行时注入：

1. 动态给 ``HIPOptions``（frozen dataclass）增加字段 ``use_in_thread_transpose``
   （默认 None）。Triton 的 ``parse_options`` 按 ``HIPOptions.__dataclass_fields__``
   从 launch kwargs 过滤编译选项，且编译缓存 key（``compute_cache_key`` 用
   ``str(options)``）基于 dataclass 字段序列化——新字段被自动吸收并区分不同 ITT
   变体的编译产物。
2. 包装 ``HIPBackend.make_ttgir``：编译时按 ``options.use_in_thread_transpose``
   临时设置 ``TRITON_HIP_USE_IN_THREAD_TRANSPOSE`` env（knob 每次访问动态读 env），
   使 ``is_in_thread_transpose_enabled()`` 返回 per-kernel 值，编译后恢复 env。

幂等：重复调用安全；若 site-packages 已手工打了同款 patch（字段已存在），跳过注入。
"""

import dataclasses
import os
import threading

_FIELD_NAME = "use_in_thread_transpose"
_ENV_NAME = "TRITON_HIP_USE_IN_THREAD_TRANSPOSE"
# frozen dataclass 重新生成 __init__/__repr__/__eq__/__hash__ 前需要删除的旧方法
# （dataclass 的 _set_new_attribute 不覆盖已有属性）
_DATACLASS_METHODS = (
    "__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__",
    "__match_args__",
)

_installed = False
_install_lock = threading.Lock()
# make_ttgir 编译期间 env 切换的互斥（编译为同步流程，锁仅防多线程编译交错）
_env_lock = threading.Lock()


def install_triton_itt_patch():
    """在 Triton HCU 后端注入 per-kernel ITT 编译选项支持。

    Returns:
        bool: True 表示注入完成（或已支持），False 表示当前环境无 Triton HCU 后端。
    """
    global _installed
    if _installed:
        return True
    with _install_lock:
        if _installed:
            return True
        try:
            from triton.backends.hcu import compiler_hcu as _chcu
        except ImportError:
            return False

        hip_options = _chcu.HIPOptions
        hip_backend = _chcu.HIPBackend

        # 1) HIPOptions 动态加字段（site-packages 已手工 patch 则跳过）
        if _FIELD_NAME not in hip_options.__dataclass_fields__:
            annotations = dict(getattr(hip_options, "__annotations__", {}))
            annotations[_FIELD_NAME] = bool
            hip_options.__annotations__ = annotations
            # 类属性默认值：重新生成 __init__ 时避免 non-default follows default
            setattr(hip_options, _FIELD_NAME, None)
            hip_options.__dataclass_fields__[_FIELD_NAME] = dataclasses.field(default=None)
            for _name in _DATACLASS_METHODS:
                if _name in hip_options.__dict__:
                    delattr(hip_options, _name)
            # 保持原始 @dataclass(frozen=True) 语义（frozen 会生成 __hash__ 与禁赋值 __setattr__）
            dataclasses.dataclass(hip_options, frozen=True)

        # 2) 包装 make_ttgir：编译时按 options 的 ITT 值切换 env
        _make_ttgir = hip_backend.__dict__.get("make_ttgir")
        if _make_ttgir is not None and not getattr(_make_ttgir, "_flagblas_itt_patched", False):
            _orig_func = _make_ttgir.__func__

            def _make_ttgir_wrapper(mod, metadata, options):
                itt = getattr(options, _FIELD_NAME, None)
                if itt is None:
                    return _orig_func(mod, metadata, options)
                with _env_lock:
                    saved = os.environ.get(_ENV_NAME)
                    os.environ[_ENV_NAME] = "1" if itt else "0"
                    try:
                        return _orig_func(mod, metadata, options)
                    finally:
                        if saved is None:
                            os.environ.pop(_ENV_NAME, None)
                        else:
                            os.environ[_ENV_NAME] = saved

            _make_ttgir_wrapper._flagblas_itt_patched = True
            hip_backend.make_ttgir = staticmethod(_make_ttgir_wrapper)

        _installed = True
        return True
