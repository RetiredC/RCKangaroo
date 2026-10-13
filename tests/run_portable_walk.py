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
                     'static_assert(RCK_EXPERIMENTAL_NARROW_PADDING == EXPECT_NARROW, "padding option");\n'
                     'static_assert(RCK_FORCE_CUDA == EXPECT_FORCE, "force option");\n'
                     'static_assert(bool(RCK_USE_CUDA_WALK) == bool(EXPECT_SOTA || EXPECT_FORCE), "backend choice");\n')
    valid = rejected = 0
    for sota, cap, force, narrow in product([0, 1], repeat=4):
        options = [f"-DEXPECT_SOTA={sota}", f"-DEXPECT_CAP={cap}", f"-DEXPECT_FORCE={force}",
                   f"-DEXPECT_NARROW={narrow}"]
        if sota or cap or force or narrow:  # First case checks defaults with no options.
            options += [f"-DRCK_EXPERIMENTAL_SOTA_HALF={sota}",
                        f"-DRCK_EXPERIMENTAL_JUMP_CAP={cap}", f"-DRCK_FORCE_CUDA={force}",
                        f"-DRCK_EXPERIMENTAL_NARROW_PADDING={narrow}"]
        command = common + options + ["-fsyntax-only", str(probe)]
        if narrow and not cap:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 or "Narrow padding requires RCK_EXPERIMENTAL_JUMP_CAP" not in result.stderr:
                raise RuntimeError(f"missing narrow-without-cap rejection: {command}\n{result.stderr}")
            rejected += 1
        else:
            subprocess.run(command, check=True)
            valid += 1
    print(f"header option/backend combinations: {valid} valid, {rejected} invalid rejected PASS", flush=True)
    binary = tmp / "portable_walk"
    subprocess.run(common + [str(root / "tests/portable_walk.cpp"), str(root / "Ec.cpp"),
                             str(root / "tests/compat/portable_utils.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
