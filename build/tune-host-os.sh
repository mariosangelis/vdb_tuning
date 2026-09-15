#!/usr/bin/env bash
# =============================================================================
# tune-host-os.sh — HOST OS tuning for benchmark-grade isolation
# =============================================================================
# Run ONCE on the HOST (not in the container) before the TPCH frontier / any
# timing-sensitive experiment. These settings affect the whole VM kernel and do
# NOT apply from inside a container, so they must be set on the host.
#
# Consolidates the repo's apply_all_optimizations.sh + SETUP_SGS.md steps, made
# portable and idempotent:
#   - CPU governor -> performance
#   - Turbo boost off        (Intel intel_pstate/no_turbo, or AMD/generic cpufreq/boost)
#   - SMT / hyperthreading off
#   - Transparent Huge Pages off
#   - NUMA balancing off
#   - CPU C-states off        (best-effort)
#   - Swap: ensure >= SWAP_GB swapfile
#
# Every knob is probed first and SKIPPED gracefully if absent (cloud VMs and ARM
# often lack turbo/SMT/governor files). A summary reports what applied vs skipped.
#
# NOTE: these are runtime settings and reset on reboot — re-run after a reboot.
# NOTE: on AWS Graviton (ARM) there is no turbo and no SMT (1 thread/core), so
#       those steps report "not present" — that is expected, not an error.
#
# Usage:  sudo ./tune-host-os.sh              # apply
#         sudo ./tune-host-os.sh --verify     # only print current state
#         SWAP_GB=10 sudo ./tune-host-os.sh   # override swap size (default 10)
# =============================================================================
set -uo pipefail   # not -e: a missing knob must not abort the whole script

SWAP_GB="${SWAP_GB:-10}"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

if [ "$(id -u)" -ne 0 ]; then exec sudo -E bash "$0" "$@"; fi

ok()   { echo -e "  \033[1;32m[applied]\033[0m $*"; }
skip() { echo -e "  \033[1;33m[skip]\033[0m    $*"; }
info() { echo -e "  \033[1;34m[info]\033[0m    $*"; }

ARCH="$(uname -m)"
echo "=== Host OS tuning (arch=${ARCH}) ==="

verify() {
    echo -e "\n=== VERIFICATION ==="
    echo "  governor:        $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "  turbo(boost):    $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a) (1=on,0=off)"
    echo "  intel no_turbo:  $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a) (1=turbo off)"
    echo "  SMT control:     $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)"
    echo "  SMT active:      $(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo n/a) (0=off)"
    echo "  THP enabled:     $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
    echo "  numa_balancing:  $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"
    echo "  swappiness:      $(cat /proc/sys/vm/swappiness 2>/dev/null || echo n/a)"
    echo "  swap total:      $(free -h | awk '/^Swap:/{print $2}')"
    echo "  online CPUs:     $(nproc)   NUMA nodes: $(command -v numactl >/dev/null && numactl --hardware 2>/dev/null | awk '/available:/{print $2}' || echo '?')"
}

if [ "${VERIFY_ONLY}" -eq 1 ]; then verify; exit 0; fi

# --- 1. CPU governor -> performance -----------------------------------------
if compgen -G "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor" >/dev/null; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null || true; done
    ok "CPU governor -> performance"
else
    skip "CPU governor (cpufreq not exposed — common on cloud VMs)"
fi

# --- 2. Turbo boost off (vendor-specific) -----------------------------------
if [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo && ok "Intel turbo -> off (no_turbo=1)"
elif [ -w /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 0 > /sys/devices/system/cpu/cpufreq/boost && ok "AMD/generic boost -> off"
else
    skip "Turbo/boost control (not present — e.g. ARM Graviton has no turbo)"
fi

# --- 3. SMT / hyperthreading off --------------------------------------------
if [ -w /sys/devices/system/cpu/smt/control ]; then
    cur="$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
    if [ "${cur}" = "notsupported" ] || [ "${cur}" = "notimplemented" ]; then
        skip "SMT control (${cur})"
    else
        echo off > /sys/devices/system/cpu/smt/control && ok "SMT/hyperthreading -> off"
    fi
else
    skip "SMT control (not present — ARM Graviton is 1 thread/core already)"
fi

# --- 4. Transparent Huge Pages off ------------------------------------------
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    echo never > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
    ok "Transparent Huge Pages -> never"
else
    skip "THP (not present)"
fi

# --- 5. NUMA balancing off ---------------------------------------------------
if [ -w /proc/sys/kernel/numa_balancing ]; then
    echo 0 > /proc/sys/kernel/numa_balancing && ok "NUMA balancing -> off"
else
    skip "numa_balancing (not present)"
fi

# --- 6. CPU C-states off (best effort) --------------------------------------
if command -v cpupower >/dev/null 2>&1; then
    for i in 1 2 3 4 5 6 7 8 9; do cpupower idle-set -d "$i" >/dev/null 2>&1 || true; done
    ok "C-states disabled via cpupower (best effort)"
elif [ -w /sys/module/intel_idle/parameters/max_cstate ]; then
    echo 0 > /sys/module/intel_idle/parameters/max_cstate 2>/dev/null && ok "intel_idle max_cstate -> 0"
else
    skip "C-state control (cpupower/intel_idle not present)"
fi

# --- 7. Swap: ensure >= SWAP_GB ---------------------------------------------
cur_swap_kb="$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)"
cur_swap_gb=$(( cur_swap_kb / 1024 / 1024 ))
if [ "${cur_swap_gb}" -ge "${SWAP_GB}" ]; then
    info "Swap already ${cur_swap_gb}GB (>= ${SWAP_GB}GB target)"
elif [ ! -e /swapfile ]; then
    info "Creating ${SWAP_GB}GB /swapfile ..."
    if fallocate -l "${SWAP_GB}G" /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024)) status=none; then
        chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile \
            && grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
        ok "Swap: ${SWAP_GB}GB /swapfile enabled"
    else
        skip "Swap: could not create /swapfile"
    fi
else
    skip "Swap: /swapfile exists but total < target; leaving as-is"
fi

verify
echo -e "\n\033[1;32mDone.\033[0m Settings are runtime-only; re-run after reboot."
