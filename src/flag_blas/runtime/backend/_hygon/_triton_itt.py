"""Runtime injection of per-kernel compilation options for the Triton HCU backend on the FlagBLAS side.

Background
----------
The following compilation switches of Triton (flagtree 0.6.1+hcu3.6) are controlled by global knobs (env read dynamically),
which only support process scope, but on Hygon gfx936 their impact on kernels varies by shape, requiring per-kernel control:

1. in-thread-transpose (ITT): env ``TRITON_HIP_USE_IN_THREAD_TRANSPOSE``.
   Large/square shapes benefit +5~15%, thin/small shapes regress (v1.2 already dispatches by shape).
2. buffer ops: env ``AMDGCN_USE_BUFFER_OPS`` (default True).
   Large/medium output shapes benefit -1~-4%, but 2048x2048x16384 +43% severely regresses,
   128x4096x1024 +3.8% and other small shapes slightly regress (v1.3 dispatches by shape).

Approach
--------
Without modifying site-packages source, perform two runtime injections at FlagBLAS import time:

1. Dynamically add fields ``use_in_thread_transpose`` and ``use_buffer_ops`` (default None)
   to ``HIPOptions`` (frozen dataclass). Triton's ``parse_options`` filters compilation options
   from launch kwargs by ``HIPOptions.__dataclass_fields__``, and the compilation cache key
   (``compute_cache_key`` uses ``str(options)``) is serialized based on dataclass fields --
   the new fields are automatically absorbed and distinguish the compiled artifacts of different variants.
2. Wrap ``HIPBackend.make_ttgir``: at compile time, temporarily set the corresponding env according
   to the field values in ``options`` (the knob reads env dynamically on each access), and restore
   env after compilation. When a field is None, leave env untouched
   (keep the global default, artifact exactly identical to no injection).

Idempotent: repeated calls are safe; if site-packages already has the same patch applied manually (fields already exist), skip injection.
"""

import dataclasses
import os
import threading

# (HIPOptions field name, env name) list -- register new per-kernel compilation options here
_OPTION_FIELDS = (
    ("use_in_thread_transpose", "TRITON_HIP_USE_IN_THREAD_TRANSPOSE"),
    ("use_buffer_ops", "AMDGCN_USE_BUFFER_OPS"),
)
# Old methods that must be deleted before regenerating __init__/__repr__/__eq__/__hash__ for the frozen dataclass
# (dataclass's _set_new_attribute does not overwrite existing attributes)
_DATACLASS_METHODS = (
    "__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__",
    "__match_args__",
)

_installed = False
_install_lock = threading.Lock()
# Mutual exclusion for env switching during make_ttgir compilation (compilation is synchronous; the lock only prevents interleaving in multi-threaded compilation)
_env_lock = threading.Lock()


def install_triton_itt_patch():
    """Inject per-kernel compilation option support into the Triton HCU backend.

    Returns:
        bool: True means injection is complete (or already supported), False means the current environment has no Triton HCU backend.
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

        # 1) Dynamically add fields to HIPOptions (skip if site-packages is already patched manually)
        for field_name, _env_name in _OPTION_FIELDS:
            if field_name not in hip_options.__dataclass_fields__:
                annotations = dict(getattr(hip_options, "__annotations__", {}))
                annotations[field_name] = bool
                hip_options.__annotations__ = annotations
                # Class attribute default: avoid "non-default follows default" when regenerating __init__
                setattr(hip_options, field_name, None)
                hip_options.__dataclass_fields__[field_name] = dataclasses.field(default=None)
                for _name in _DATACLASS_METHODS:
                    if _name in hip_options.__dict__:
                        delattr(hip_options, _name)
                # Preserve the original @dataclass(frozen=True) semantics (frozen generates __hash__ and a disabled-assignment __setattr__)
                dataclasses.dataclass(hip_options, frozen=True)

        # 2) Wrap make_ttgir: switch the corresponding env by options field values at compile time
        _make_ttgir = hip_backend.__dict__.get("make_ttgir")
        if _make_ttgir is not None and not getattr(_make_ttgir, "_flagblas_itt_patched", False):
            _orig_func = _make_ttgir.__func__

            def _make_ttgir_wrapper(mod, metadata, options):
                to_set = {}
                for field_name, env_name in _OPTION_FIELDS:
                    val = getattr(options, field_name, None)
                    if val is not None:
                        to_set[env_name] = "1" if val else "0"
                if not to_set:
                    return _orig_func(mod, metadata, options)
                with _env_lock:
                    saved = {env: os.environ.get(env) for env in to_set}
                    for env, val in to_set.items():
                        os.environ[env] = val
                    try:
                        return _orig_func(mod, metadata, options)
                    finally:
                        for env, val in saved.items():
                            if val is None:
                                os.environ.pop(env, None)
                            else:
                                os.environ[env] = val

            _make_ttgir_wrapper._flagblas_itt_patched = True
            hip_backend.make_ttgir = staticmethod(_make_ttgir_wrapper)

        _installed = True
        return True
