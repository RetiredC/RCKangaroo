#!/usr/bin/env python3
"""Build/run shared walk-policy checks on macOS/Linux without CUDA.

Uses the repository's unchanged Ec.cpp; temporary binaries are removed on exit.
This does not compile or execute CUDA kernels or validate GPU performance.
"""
import os
from itertools import product
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
compiler = shlex.split(os.environ.get("CXX", "clang++"))
common = compiler + ["-std=c++17", "-O2", "-fwrapv", "-Wno-unknown-pragmas",
                     "-I" + str(root / "tests/compat"), "-I" + str(root)]
with tempfile.TemporaryDirectory(prefix="rck-portable-walk-") as tmp:
    tmp = Path(tmp)
    probe = tmp / "config.cpp"
    probe.write_text('#include "ExperimentalWalk.h"\n'
                     'static_assert(RCK_EXPERIMENTAL_SOTA_HALF == EXPECT_SOTA, "sota option");\n'
                     'static_assert(RCK_EXPERIMENTAL_JUMP_CAP == EXPECT_CAP, "cap option");\n'
                     'static_assert(RCK_FORCE_CUDA == EXPECT_FORCE, "force option");\n'
                     'static_assert(bool(RCK_USE_CUDA_WALK) == bool(EXPECT_SOTA || EXPECT_FORCE), "backend choice");\n')
    for sota, cap, force in product([0, 1], repeat=3):
        options = [f"-DEXPECT_SOTA={sota}", f"-DEXPECT_CAP={cap}", f"-DEXPECT_FORCE={force}"]
        if sota or cap or force:  # First case checks header defaults with no options.
            options += [f"-DRCK_EXPERIMENTAL_SOTA_HALF={sota}",
                        f"-DRCK_EXPERIMENTAL_JUMP_CAP={cap}", f"-DRCK_FORCE_CUDA={force}"]
        subprocess.run(common + options + ["-fsyntax-only", str(probe)], check=True)
    print("header option/backend combinations: 8 PASS", flush=True)
    binary = tmp / "portable_walk"
    subprocess.run(common + [str(root / "tests/portable_walk.cpp"), str(root / "Ec.cpp"),
                             str(root / "tests/compat/portable_utils.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
