# ClusterFuzzLite build image for pyhrp.
# Uses the OSS-Fuzz Python base image, which provides Atheris and the
# compile_python_fuzzer helper.
# Pinned by SHA256 so the build image only changes through reviewed updates.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:99c714c91ec30778a3ce3ae5b37491a81fc043808bc8f2955159c2f608f80116

COPY . $SRC
WORKDIR $SRC
COPY .clusterfuzzlite/build.sh $SRC/build.sh
