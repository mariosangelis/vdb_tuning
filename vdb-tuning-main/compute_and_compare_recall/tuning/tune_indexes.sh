#!/usr/bin/env bash
set -euo pipefail

sudo python3 -u tune_ivfflat.py --dataset gist --warm 2>&1 | tee build_ivfflat_gist.txt
sudo python3 -u tune_ivfflat.py --dataset deep --warm 2>&1 | tee build_ivfflat_deep.txt
sudo python3 -u tune_hnsw.py --dataset gist --warm 2>&1 | tee build_hnsw_gist.txt
sudo python3 -u tune_hnsw.py --dataset deep --warm 2>&1 | tee build_hnsw_deep.txt
sudo python3 -u tune_parallel_ivfflat.py --dataset gist 2>&1 | tee build_ivfflat_parallel_gist.txt
sudo python3 -u tune_parallel_ivfflat.py --dataset deep 2>&1 | tee build_ivfflat_parallel_deep.txt

echo "ALL DONE"
