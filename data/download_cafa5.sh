#!/bin/sh
# download the cafa5 function test set (temporal holdout, as packaged by bioreason-pro) into
# data/cafa5.
set -e
d="$(dirname "$0")/cafa5/test/data"
mkdir -p "$d"
curl -fsSL "https://huggingface.co/datasets/wanglab/bioreason-pro-test-data/resolve/main/data/test-00000-of-00001.parquet" \
  -o "$d/test-00000-of-00001.parquet"
