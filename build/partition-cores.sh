#!/usr/bin/env bash
# =============================================================================
# partition-cores.sh — auto-generate CPU core assignment for TPCH frontier
# =============================================================================
# Rewrites vdb-tuning-main/TPCH/code/experiment_config.json so that
#   postgres_cores / postgres_numa_nodes / topk_producers_cores /
#   olap_producers_cores
# match the ACTUAL CPUs of this machine. The committed values target a 64-core
# CloudLab box (producers on cores 32-63) and would crash on smaller VMs.
#
# Policy (per the design: server on one socket, clients on the rest):
#   - MULTI-NUMA host: PostgreSQL -> NUMA node 0 (its cores + memory),
#                      producers  -> all cores on the OTHER node(s).
#   - SINGLE-NUMA host: split that node's cores by --pg-frac (default 0.5):
#                      first fraction -> PostgreSQL, remainder -> producers.
#
# Runs inside the container (default) or on the host; edits the bind-mounted
# repo either way. Idempotent (keeps a .orig backup on first change).
#
# Usage:
#   ./partition-cores.sh                 # auto, half/half on single-NUMA
#   ./partition-cores.sh --pg-frac 0.75  # give PG 3/4 of a single node
# =============================================================================
set -euo pipefail

PG_FRAC="0.5"
while [ $# -gt 0 ]; do
    case "$1" in
        --pg-frac) PG_FRAC="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Locate the config json (container path first, else walk from repo root).
if [ -f /vdb_tuning/vdb-tuning-main/TPCH/code/experiment_config.json ]; then
    EC=/vdb_tuning/vdb-tuning-main/TPCH/code/experiment_config.json
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    EC="${REPO}/vdb-tuning-main/TPCH/code/experiment_config.json"
fi
[ -f "${EC}" ] || { echo "ERROR: experiment_config.json not found at ${EC}" >&2; exit 1; }
[ -f "${EC}.orig" ] || cp "${EC}" "${EC}.orig"

python3 - "${EC}" "${PG_FRAC}" <<'PYEOF'
import json, os, sys, subprocess, re

ec_path, pg_frac = sys.argv[1], float(sys.argv[2])

def numa_map():
    """Return {node: [cpus]} from numactl --hardware, or {} if unavailable."""
    try:
        out = subprocess.check_output(["numactl", "--hardware"], text=True)
    except Exception:
        return {}
    m = {}
    for line in out.splitlines():
        g = re.match(r"node (\d+) cpus: (.*)", line.strip())
        if g:
            cpus = [int(x) for x in g.group(2).split()] if g.group(2).strip() else []
            m[int(g.group(1))] = cpus
    return m

nodes = numa_map()
nproc = os.cpu_count() or 1

if len(nodes) >= 2:
    # Multi-NUMA: PG on node 0, producers on the rest.
    node0 = sorted(nodes[0])
    others = sorted(c for n, cpus in nodes.items() if n != 0 for c in cpus)
    pg_cores = node0
    prod_cores = others if others else node0
    pg_numa = "0"
    policy = f"multi-NUMA: PG=node0 ({len(pg_cores)} cpus), producers=other nodes ({len(prod_cores)} cpus)"
else:
    # Single-NUMA (or numactl absent): split the online cores by pg_frac.
    all_cores = sorted(nodes.get(0, list(range(nproc))))
    n = len(all_cores)
    split = max(1, min(n - 1, round(n * pg_frac)))
    pg_cores = all_cores[:split]
    prod_cores = all_cores[split:] or all_cores[:1]
    pg_numa = "0"
    policy = f"single-NUMA: {n} cpus, PG={pg_cores[0]}-{pg_cores[-1]} ({len(pg_cores)}), producers={prod_cores[0]}-{prod_cores[-1]} ({len(prod_cores)})"

cfg = json.load(open(ec_path))
hw = cfg.setdefault("hardware", {})
hw["num_pg_cores"] = len(pg_cores)
hw["postgres_cores"] = f"{pg_cores[0]}-{pg_cores[-1]}" if len(pg_cores) > 1 else str(pg_cores[0])
hw["postgres_numa_nodes"] = pg_numa
hw["topk_producers_cores"] = prod_cores
hw["olap_producers_cores"] = prod_cores
json.dump(cfg, open(ec_path, "w"), indent=2)

print(f"partition-cores: {policy}")
print(f"  postgres_cores       = {hw['postgres_cores']}")
print(f"  postgres_numa_nodes  = {hw['postgres_numa_nodes']}")
print(f"  producers_cores      = {prod_cores[0]}..{prod_cores[-1]} ({len(prod_cores)} cores)")
PYEOF

echo "partition-cores: updated ${EC}"
