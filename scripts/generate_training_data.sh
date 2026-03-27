#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Generate the full training + validation dataset for Phase 2 SFT.
#
# Usage:
#   bash scripts/generate_training_data.sh
#
# Output goes to data/traces/train_*.json and data/traces/val_*.json
# -----------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

GENERATOR="python3 -m data.generators.generate_dataset"

REGULAR_FAMILIES=("erdos_renyi" "barabasi_albert" "random_tree" "grid")
HARD_FAMILIES=("bridge" "bottleneck" "high_girth")
ALGORITHMS=("bfs" "dfs" "dijkstra")

TRAIN_COUNT_REGULAR=100
TRAIN_COUNT_HARD=50
VAL_COUNT_REGULAR=20
VAL_COUNT_HARD=10

NODE_COUNT=15
SEED=42

echo "================================================================="
echo "  Graph of Thoughts — Training Data Generation"
echo "================================================================="
echo ""

# -------------------------------------------------------------------
# Regular families
# -------------------------------------------------------------------
for algo in "${ALGORITHMS[@]}"; do
    for family in "${REGULAR_FAMILIES[@]}"; do
        weighted_flag=""
        if [ "$algo" = "dijkstra" ]; then
            weighted_flag="--weighted"
        fi

        echo "[train] ${algo} / ${family}  (${TRAIN_COUNT_REGULAR} traces, n=${NODE_COUNT})"
        $GENERATOR \
            --algorithm "$algo" \
            --family "$family" \
            --n "$NODE_COUNT" \
            --count "$TRAIN_COUNT_REGULAR" \
            --seed "$SEED" \
            $weighted_flag \
            --out "data/traces/train_${algo}_${family}.json"

        echo "[val]   ${algo} / ${family}  (${VAL_COUNT_REGULAR} traces, n=${NODE_COUNT})"
        $GENERATOR \
            --algorithm "$algo" \
            --family "$family" \
            --n "$NODE_COUNT" \
            --count "$VAL_COUNT_REGULAR" \
            --seed $((SEED + 10000)) \
            $weighted_flag \
            --out "data/traces/val_${algo}_${family}.json"

        echo ""
    done
done

# -------------------------------------------------------------------
# Hard-case families
# -------------------------------------------------------------------
for algo in "${ALGORITHMS[@]}"; do
    for family in "${HARD_FAMILIES[@]}"; do
        weighted_flag=""
        if [ "$algo" = "dijkstra" ]; then
            weighted_flag="--weighted"
        fi

        echo "[train] ${algo} / ${family}  (${TRAIN_COUNT_HARD} traces)"
        $GENERATOR \
            --algorithm "$algo" \
            --family "$family" \
            --n "$NODE_COUNT" \
            --count "$TRAIN_COUNT_HARD" \
            --seed "$SEED" \
            $weighted_flag \
            --out "data/traces/train_${algo}_${family}.json"

        echo "[val]   ${algo} / ${family}  (${VAL_COUNT_HARD} traces)"
        $GENERATOR \
            --algorithm "$algo" \
            --family "$family" \
            --n "$NODE_COUNT" \
            --count "$VAL_COUNT_HARD" \
            --seed $((SEED + 10000)) \
            $weighted_flag \
            --out "data/traces/val_${algo}_${family}.json"

        echo ""
    done
done

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo "================================================================="
echo "  Generation complete!"
echo "================================================================="
echo ""
echo "Training files:"
ls -lh data/traces/train_*.json 2>/dev/null || echo "  (none found)"
echo ""
echo "Validation files:"
ls -lh data/traces/val_*.json 2>/dev/null || echo "  (none found)"
echo ""
echo "Total disk usage:"
du -sh data/traces/
