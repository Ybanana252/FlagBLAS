# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPU SDK 2.1 L2 reference exclusions, verified by isolated calls on 2026-09-14.

Unsupported entry points print ``cublasX..._v2 is not supported`` and can abort
the process, so mark skips before execution rather than catching exceptions.
This helper is called only for T-Head correctness tests using ``--ref cuda``.
"""

import pytest


_UNSUPPORTED_FAMILIES = {
    "gbmv",
    "symv",
    "hemv",
    "trmv",
    "tbmv",
    "tpmv",
    "sbmv",
    "hbmv",
    "spmv",
    "hpmv",
    "trsv",
    "tbsv",
    "tpsv",
    "her",
    "her2",
    "hpr",
    "hpr2",
}
_UNSUPPORTED_VARIANTS = {
    "gemv": {"cgemv", "zgemv"},
    "ger": {"cgeru", "cgerc", "zgeru", "zgerc"},
    "syr": {"csyr", "zsyr"},
}


def skip_unsupported_l2_references(items):
    for item in items:
        filename = item.path.name
        if not filename.startswith("test_") or not filename.endswith(".py"):
            continue
        family = filename[5:-3]
        if family in _UNSUPPORTED_FAMILIES:
            operator = family.upper()
        elif family in _UNSUPPORTED_VARIANTS:
            # Some files use per-dtype test names; GER and balanced SYR also
            # select the operator via parameter values / parametrized marks.
            names = set((item.originalname or item.name).split("_"))
            names.update(mark.name for mark in item.iter_markers())
            params = getattr(getattr(item, "callspec", None), "params", {})
            names.update(value for value in params.values() if isinstance(value, str))
            unsupported = names & _UNSUPPORTED_VARIANTS[family]
            if not unsupported:
                continue
            operator = "/".join(sorted(unsupported)).upper()
        else:
            continue
        item.add_marker(
            pytest.mark.skip(
                reason=(
                    f"T-Head --ref cuda: PPU SDK 2.1 official BLAS does not support "
                    f"{operator} (reports 'is not supported' and may abort); "
                    "use --ref cpu to test the FlagBLAS implementation"
                )
            )
        )
