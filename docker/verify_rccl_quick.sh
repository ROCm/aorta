#!/bin/bash

# Quick verification commands to check RCCL in Docker container

echo "========================================"
echo "Quick RCCL warp_speed_v1 Verification"
echo "========================================"

# 1. Check environment variable
echo -e "\n1. RCCL Installation Path:"
echo "   RCCL_ROOT = $RCCL_ROOT"

# 2. Check if library exists
echo -e "\n2. RCCL Library:"
if [ -f "/opt/rccl-warpspeed/lib/librccl.so" ]; then
    echo "   [OK] Found at /opt/rccl-warpspeed/lib/librccl.so"
    readelf -d /opt/rccl-warpspeed/lib/librccl.so | grep SONAME
else
    echo "   [ERROR] NOT FOUND at /opt/rccl-warpspeed/lib/"
fi

# 3. Check LD_LIBRARY_PATH priority
echo -e "\n3. Library Path Priority:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -n rccl

# 4. Check git branch (if source available)
echo -e "\n4. Git Branch Check:"
if [ -d "/opt/rccl/.git" ]; then
    cd /opt/rccl
    echo "   Branch: $(git branch --show-current)"
    echo "   Commit: $(git log --oneline -1)"
else
    echo "   [INFO] Source not available"
fi

# 5. Python check
echo -e "\n5. Python RCCL Check:"
python3 -c "
import os
# Check environment
rccl_root = os.environ.get('RCCL_ROOT', 'NOT SET')
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
print(f'   RCCL_ROOT in Python: {rccl_root}')
if '/opt/rccl-warpspeed/lib' in ld_path.split(':')[0]:
    print('   [OK] /opt/rccl-warpspeed/lib is FIRST in LD_LIBRARY_PATH')
else:
    print('   [WARN] Check LD_LIBRARY_PATH order')
"

echo -e "\n========================================"
echo "If all checks show /opt/rccl-warpspeed, you're using warp_speed_v1"
echo "========================================
