#!/usr/bin/env bash
# validate.sh
#
# Dry-run the federation pipeline without launching training.
# Verifies:
#   - All scripts pass bash -n / python -c "import"
#   - Required ckpts exist on each side
#   - SSH between coordinator host and RTX6000 works
#   - rsync between coordinator host and RTX6000 works
#   - Federation directory is writable on both sides
#   - Trainer scripts have the federation patch applied

set -uo pipefail

: "${FED_DIR:=/home/alexm/OpenMythos/fed}"
: "${FED_RTX_HOST:=alexm@kebab-rtx6000.lan}"
: "${FED_RTX_DIR:=/home/alexm/OpenMythos/fed}"
: "${FED_BOOTSTRAP_CKPT_CLUSTER:=}"
: "${FED_BOOTSTRAP_CKPT_RTX:=}"

REPO=/home/alexm/OpenMythos
FAIL=0

ok()   { echo "  ✓ $*"; }
fail() { echo "  ✗ $*"; FAIL=$((FAIL+1)); }
sect() { echo; echo "[$1]"; }

sect "syntax checks"
for f in coordinator.py sync_hook.py; do
    if python3 -c "import ast; ast.parse(open('$REPO/training/federation/$f').read())" 2>/dev/null; then
        ok "$f compiles"
    else
        fail "$f failed to compile"
    fi
done
for f in launch_federation.sh abort_federation.sh validate.sh; do
    if bash -n "$REPO/training/federation/$f" 2>/dev/null; then
        ok "$f bash -n OK"
    else
        fail "$f bash -n FAILED"
    fi
done

sect "trainer patches applied"
for tr in 3b_varT_act_v3.py 3b_varT_pondernet_joint.py; do
    if grep -q "fed_sync_hook" "$REPO/training/$tr"; then
        ok "$tr contains federation hook"
    else
        fail "$tr is NOT patched"
    fi
done

sect "SSH connectivity"
if ssh -o ConnectTimeout=10 "$FED_RTX_HOST" "echo ok" 2>/dev/null | grep -q ok; then
    ok "SSH to $FED_RTX_HOST works"
else
    fail "SSH to $FED_RTX_HOST FAILED"
fi

sect "rsync round-trip"
TMP=$(mktemp)
echo "test_$(date +%s)" > "$TMP"
if rsync -az --timeout=30 "$TMP" "$FED_RTX_HOST:/tmp/fed_validate_test" 2>/dev/null; then
    if ssh -q "$FED_RTX_HOST" "cat /tmp/fed_validate_test" 2>/dev/null | grep -q test_; then
        ok "rsync push and read works"
    else
        fail "rsync push appeared to succeed but file unreadable"
    fi
    ssh -q "$FED_RTX_HOST" "rm -f /tmp/fed_validate_test" 2>/dev/null
else
    fail "rsync push to $FED_RTX_HOST FAILED"
fi
rm -f "$TMP"

sect "fed directories"
if mkdir -p "$FED_DIR" 2>/dev/null && [ -w "$FED_DIR" ]; then
    ok "$FED_DIR writable on cluster side"
else
    fail "$FED_DIR not writable on cluster side"
fi
if ssh -q "$FED_RTX_HOST" "mkdir -p $FED_RTX_DIR && [ -w $FED_RTX_DIR ] && echo ok" 2>/dev/null | grep -q ok; then
    ok "$FED_RTX_DIR writable on RTX6000"
else
    fail "$FED_RTX_DIR not writable on RTX6000"
fi

sect "bootstrap checkpoints"
if [ -n "$FED_BOOTSTRAP_CKPT_CLUSTER" ]; then
    if [ -f "$FED_BOOTSTRAP_CKPT_CLUSTER" ]; then
        size_gb=$(stat -c %s "$FED_BOOTSTRAP_CKPT_CLUSTER" | awk '{printf "%.1f", $1/1073741824}')
        ok "cluster bootstrap exists ($size_gb GB)"
    else
        fail "cluster bootstrap missing: $FED_BOOTSTRAP_CKPT_CLUSTER"
    fi
else
    echo "  - FED_BOOTSTRAP_CKPT_CLUSTER not set (skipping check)"
fi
if [ -n "$FED_BOOTSTRAP_CKPT_RTX" ]; then
    if ssh -q "$FED_RTX_HOST" "test -f $FED_BOOTSTRAP_CKPT_RTX"; then
        ok "rtx bootstrap exists"
    else
        fail "rtx bootstrap missing: $FED_BOOTSTRAP_CKPT_RTX (rsync from cluster first)"
    fi
else
    echo "  - FED_BOOTSTRAP_CKPT_RTX not set (skipping check)"
fi

sect "federation scripts on RTX6000"
for f in coordinator.py sync_hook.py; do
    if ssh -q "$FED_RTX_HOST" "test -f $REPO/training/federation/$f"; then
        ok "RTX6000 has $f"
    else
        fail "RTX6000 missing $f (rsync federation/ to it)"
    fi
done

sect "no in-flight training"
NODES_200G="kebab-spark-200g kebab-gx10-200g kebab-gx10-2-200g kebab-gx10-3-200g"
for h in $NODES_200G; do
    n=$(ssh -q -o ConnectTimeout=5 alexm@"$h" "pgrep -fc 'python3 .*training/3b_varT' 2>/dev/null || echo 0")
    if [ "${n:-0}" -eq 0 ]; then
        ok "$h: idle"
    else
        fail "$h: $n training procs running (federation cannot launch)"
    fi
done
n_rtx=$(ssh -q -o ConnectTimeout=10 "$FED_RTX_HOST" "pgrep -fc 'python3 .*training/3b_varT' 2>/dev/null || echo 0")
if [ "${n_rtx:-0}" -eq 0 ]; then
    ok "RTX6000: idle"
else
    fail "RTX6000: $n_rtx training procs running"
fi

echo
if [ $FAIL -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED"
    exit 0
else
    echo "✗ $FAIL CHECKS FAILED"
    exit 1
fi
