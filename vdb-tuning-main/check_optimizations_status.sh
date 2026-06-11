#!/bin/bash
# Check status of all system optimizations

echo "=== SYSTEM OPTIMIZATION STATUS CHECK ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# Function to check status
check_status() {
    local name="$1"
    local current="$2"
    local expected="$3"

    if [ "$current" = "$expected" ]; then
        echo "✓ $name: $current (optimal)"
    else
        echo "✗ $name: $current (expected: $expected)"
    fi
}

# 1. CPU Governor
echo "1. CPU GOVERNOR"
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "N/A")
check_status "CPU Governor" "$gov" "performance"

# 2. Turbo Boost
echo -e "\n2. TURBO BOOST"
turbo=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "N/A")
check_status "Turbo Boost" "$turbo" "0"

# 3. CPU Frequencies
echo -e "\n3. CPU FREQUENCIES"
echo "Current frequencies (sample):"
grep "cpu MHz" /proc/cpuinfo | head -3
min_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || echo "0")
max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "0")
echo "Min freq: $((min_freq/1000)) MHz, Max freq: $((max_freq/1000)) MHz"

# 4. THP
echo -e "\n4. TRANSPARENT HUGE PAGES"
thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[.*\]' | tr -d '[]')
check_status "THP Enabled" "$thp" "never"
thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag | grep -o '\[.*\]' | tr -d '[]')
check_status "THP Defrag" "$thp_defrag" "never"

# 5. NUMA Balancing
echo -e "\n5. NUMA BALANCING"
numa_bal=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo "N/A")
check_status "NUMA Balancing" "$numa_bal" "0"

# 6. C-States
echo -e "\n6. CPU C-STATES"
if command -v cpupower >/dev/null 2>&1; then
    echo "C-State status (CPU0):"
    cpupower -c 0 idle-info | grep -E "Available|DISABLED" | head -5
else
    echo "cpupower not available"
fi

# 7. Swappiness
echo -e "\n7. SWAPPINESS"
swappiness=$(cat /proc/sys/vm/swappiness)
check_status "Swappiness" "$swappiness" "0"

# Summary
echo -e "\n=== SUMMARY ==="
echo "All settings should show ✓ for optimal benchmark performance."
echo "Run 'sudo bash apply_all_optimizations.sh' to apply all optimizations."