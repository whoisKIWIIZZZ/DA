#!/bin/bash
# 离线分析脚本 —— 对已有 bridge_ckpts 运行全部分析
# 用法: bash tools/analyze_bridge.sh <bridge_ckpt_dir>
#   or: bash tools/analyze_bridge.sh output_Models/uda/office-home/AC/diff_align/bridge_ckpts
#
# 适用于 rsync 场景:
#   rsync -avz server:/path/to/bridge_ckpts/ ./bridge_analysis/
#   bash tools/analyze_bridge.sh ./bridge_analysis/

set -e

BRIDGE_DIR="${1:-output_Models/uda/office-home/AC/diff_align/bridge_ckpts}"

if [ ! -d "$BRIDGE_DIR" ]; then
    echo "错误: 目录不存在: $BRIDGE_DIR"
    echo "用法: bash tools/analyze_bridge.sh <bridge_ckpt_dir>"
    exit 1
fi

echo "========================================="
echo "Bridge 离线分析"
echo "  Checkpoint 目录: $BRIDGE_DIR"
echo "========================================="

# 自动检测 python
PYTHON=${PYTHON:-python}

echo ""
echo "========== Layer 0-1: 几何恒等式 + 模态分解 =========="
$PYTHON tools/analyze_crossover.py "$BRIDGE_DIR"

echo ""
echo "========== Layer 2: NTK 共识算子 + 信任半径 =========="
$PYTHON tools/analyze_ntk.py "$BRIDGE_DIR"

echo ""
echo "========== 全部完成 =========="
echo "结果目录:"
echo "  ${BRIDGE_DIR}/analysis/       ← Layer 0-1 分析"
echo "  ${BRIDGE_DIR}/ntk_analysis/   ← Layer 2 NTK 分析"
