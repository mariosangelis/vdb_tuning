#!/bin/bash
# Apply all system optimizations for benchmarking
# Generated from system_optimizations_applied.md

echo "=== APPLYING ALL SYSTEM OPTIMIZATIONS ==="
echo "This script will apply all performance optimizations for benchmarking"
echo

# 1. CPU Governor to Performance Mode
echo "1. Setting CPU Governor to performance mode..."
for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $i >/dev/null 2>&1
done
echo "   ✓ CPU Governor set to performance"

# 2. Disable CPU Turbo Boost
echo -e "\n2. Disabling CPU Turbo Boost..."
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null 2>&1
echo "   ✓ Turbo Boost disabled"

# 3. Fix CPU Frequency to 2.8GHz
echo -e "\n3. Fixing CPU frequency to 2.8GHz..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/; do
    if [ -d "$cpu" ]; then
        echo 2800000 | sudo tee ${cpu}scaling_min_freq >/dev/null 2>&1
        echo 2800000 | sudo tee ${cpu}scaling_max_freq >/dev/null 2>&1
    fi
done
echo "   ✓ CPU frequency fixed at 2.8GHz"

# 4. Disable Transparent Huge Pages
echo -e "\n4. Disabling Transparent Huge Pages..."
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null
echo "   ✓ THP disabled"

# 5. Disable NUMA Balancing
echo -e "\n5. Disabling NUMA balancing..."
echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null
echo "   ✓ NUMA balancing disabled"

# 6. Disable CPU C-States
echo -e "\n6. Disabling CPU C-States..."
for i in {1..9}; do
    sudo cpupower idle-set -d $i >/dev/null 2>&1
done
# Alternative method for Intel
echo 0 | sudo tee /sys/module/intel_idle/parameters/max_cstate >/dev/null 2>&1
echo "   ✓ C-States disabled"

# 7. Set Swappiness to 0
echo -e "\n7. Setting swappiness to 0..."
echo 0 | sudo tee /proc/sys/vm/swappiness >/dev/null
echo "   ✓ Swappiness set to 0"

# 8. Move IRQs away from benchmark CPUs (optional)
echo -e "\n8. Moving IRQs away from benchmark CPUs..."
for irq in /proc/irq/*/smp_affinity_list; do
    if [ -w "$irq" ]; then
        echo "32-63" | sudo tee "$irq" >/dev/null 2>&1
    fi
done
echo "   ✓ IRQ affinity adjusted"

# Verification
echo -e "\n=== VERIFICATION ==="
echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo "Turbo Boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'N/A') (0=disabled)"
echo "CPU Frequency: $(grep "cpu MHz" /proc/cpuinfo | head -1)"
echo "THP Status: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "NUMA Balancing: $(cat /proc/sys/kernel/numa_balancing)"
echo "Swappiness: $(cat /proc/sys/vm/swappiness)"

echo -e "\n✅ All optimizations applied!"
echo "Note: These changes are temporary and will reset on reboot."
echo "For detailed documentation, see: system_optimizations_applied.md"