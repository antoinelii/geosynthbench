#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Running GeoSynthBench demo dataset generation..."

# Config
OUT_DIR="data_bulky"
TASKS="e1,d1,s1,n1,a1"
PER_TASK="100,100,100,100,100"
SEED=12345
OVERWRITE=False

# Ensure dependencies are synced
echo "📦 Syncing environment with uv..."
uv sync

# Run generation script
echo "🧠 Generating datasets..."
uv run python scripts/generation_demo.py \
  --out "$OUT_DIR" \
  --tasks "$TASKS" \
  --per-task "$PER_TASK" \
  --seed "$SEED" \
  --overwrite "$OVERWRITE"

echo "✅ Generation complete!"
echo "📁 Output directory: $OUT_DIR"
echo "📊 Structure: $OUT_DIR/<task>/dataset.jsonl"