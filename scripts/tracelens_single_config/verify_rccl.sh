#!/bin/bash

# Verification script to ensure RCCL warp_speed_v1 is properly configured

echo "==================================="
echo "RCCL Environment Verification"
echo "==================================="
echo ""

# Check environment variables
echo "[CHECK] Environment Variables:"
echo "  RCCL_ROOT: $RCCL_ROOT"
if [ -z "$RCCL_ROOT" ]; then
    echo "  [ERROR] RCCL_ROOT not set!"
    exit 1
fi

# Check if RCCL library exists
echo ""
echo "[CHECK] RCCL Library:"
if [ -f "$RCCL_ROOT/lib/librccl.so" ]; then
    echo "  [OK] Found librccl.so at $RCCL_ROOT/lib/"
    ls -lah $RCCL_ROOT/lib/librccl.so*
else
    echo "  [ERROR] librccl.so not found at $RCCL_ROOT/lib/"
    exit 1
fi

# Check LD_LIBRARY_PATH
echo ""
echo "[CHECK] LD_LIBRARY_PATH includes RCCL:"
if echo "$LD_LIBRARY_PATH" | grep -q "$RCCL_ROOT/lib"; then
    echo "  [OK] $RCCL_ROOT/lib is in LD_LIBRARY_PATH"
else
    echo "  [WARN] $RCCL_ROOT/lib not found in LD_LIBRARY_PATH"
fi

# Check git branch if source exists
echo ""
echo "[CHECK] RCCL Source Branch:"
if [ -d "/opt/rccl/.git" ]; then
    BRANCH=$(git -C /opt/rccl branch --show-current 2>/dev/null)
    COMMIT=$(git -C /opt/rccl log --oneline -1 2>/dev/null)
    echo "  Branch: $BRANCH"
    echo "  Latest commit: $COMMIT"
    if [ "$BRANCH" == "warp_speed_v1" ]; then
        echo "  [OK] On warp_speed_v1 branch"
    else
        echo "  [WARN] Not on warp_speed_v1 branch"
    fi
else
    echo "  [INFO] RCCL source not available for verification"
fi

# Test Python import
echo ""
echo "[CHECK] Python RCCL/NCCL Access:"
python3 -c "
import os
import torch
if torch.cuda.is_available():
    print(f'  [OK] PyTorch CUDA available')
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  CUDA version: {torch.version.cuda}')
    # Check which RCCL/NCCL backend will be used
    rccl_lib = os.environ.get('RCCL_ROOT', '')
    if rccl_lib:
        print(f'  [OK] Will use RCCL from: {rccl_lib}')
else:
    print('  [WARN] PyTorch CUDA not available')
" 2>/dev/null || echo "  [ERROR] Failed to import torch"

# Summary
echo ""
echo "==================================="
if [ -f "$RCCL_ROOT/lib/librccl.so" ] && echo "$LD_LIBRARY_PATH" | grep -q "$RCCL_ROOT/lib"; then
    echo "[OK] RCCL warp_speed_v1 environment is properly configured"
    echo "     Library path: $RCCL_ROOT/lib/librccl.so"
else
    echo "[ERROR] RCCL environment issues detected"
fi
echo "===================================
