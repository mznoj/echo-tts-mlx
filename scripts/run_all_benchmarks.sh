#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source .venv/bin/activate

echo "=== Tier 3: f16 ==="
python -m benchmarks.run_benchmarks --tier 3 --quantize none --output benchmark_tier3_f16.json --no-quality-check --force-synthetic-reference
echo "=== Tier 3: 8bit ==="
python -m benchmarks.run_benchmarks --tier 3 --quantize 8bit --output benchmark_tier3_8bit.json --no-quality-check --force-synthetic-reference
echo "=== Tier 3: 4bit ==="
python -m benchmarks.run_benchmarks --tier 3 --quantize 4bit --output benchmark_tier3_4bit.json --no-quality-check --force-synthetic-reference
echo "=== ALL BENCHMARKS COMPLETE ==="
