#!/bin/bash -eu
# ClusterFuzzLite build script — installs pyhrp and compiles each Python
# harness in tests/fuzz/ via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC"

# Pin pip so the build environment is reproducible and only changes through a
# reviewed bump (the same rationale as the SHA-pinned base image).
pip3 install --upgrade "pip==24.3.1"

# Install the package and its runtime dependencies (numpy, polars, scipy, ...)
# into the build environment so PyInstaller can discover and bundle pyhrp into
# each frozen fuzzer binary.
pip3 install .

# PyInstaller does not discover numpy's C-extension submodules on its own, so
# the frozen fuzzer crashes at runtime with
# "No module named 'numpy._core._exceptions'". --collect-all pulls in every
# numpy submodule, data file and shared library. polars and scipy are bundled
# correctly via the static import graph and PyInstaller's built-in hooks.
for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer" --collect-all numpy
done
