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

# PyInstaller does not discover the compiled extension modules of numpy/scipy on
# their own, so the frozen fuzzer crashes at runtime (numpy: "No module named
# 'numpy._core._exceptions'"; scipy: "No module named 'scipy._cyutility'").
# --collect-all pulls in every submodule, data file and shared library for both.
# polars is bundled correctly via the static import graph.
for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer" --collect-all numpy --collect-all scipy
done
