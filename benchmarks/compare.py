from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _pct_change(*, baseline: float, current: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (current - baseline) / baseline * 100.0


def _status_icon(change_pct: float) -> str:
    # Lower wall time is better, so negative change is an improvement.
    if change_pct <= -2.0:
        return "✅"
    if change_pct >= 10.0:
        return "❌"
    if change_pct >= 5.0:
        return "⚠️"
    return ""


def _fmt_ms(value_ms: float | None) -> str:
    if value_ms is None:
        return "-"
    return f"{value_ms:.1f}ms"


def _fmt_s(value_s: float | None) -> str:
    if value_s is None:
        return "-"
    return f"{value_s:.2f}s"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _print_tier1(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    if not keys:
        return

    print("Component Microbenchmarks (Tier 1):")
    print(f"{'benchmark':30} {'baseline':>12} {'current':>12} {'change':>10}")
    for key in keys:
        b = baseline.get(key, {})
        c = current.get(key, {})
        b_ms = b.get("median_ms")
        c_ms = c.get("median_ms")
        change = _pct_change(baseline=float(b_ms), current=float(c_ms)) if b_ms is not None and c_ms is not None else 0.0
        icon = _status_icon(change)
        change_str = f"{change:+.1f}% {icon}".rstrip()
        print(f"{key:30} {_fmt_ms(b_ms):>12} {_fmt_ms(c_ms):>12} {change_str:>10}")
    print()


def _print_tier2(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    if not keys:
        return

    print("Pipeline Benchmarks (Tier 2):")
    print(f"{'benchmark':30} {'baseline':>12} {'current':>12} {'change':>10}")
    for key in keys:
        b = baseline.get(key, {})
        c = current.get(key, {})
        b_s = b.get("wall_time_s")
        c_s = c.get("wall_time_s")
        if b_s is None or c_s is None:
            # Scaling/ttfb entries do not always include wall_time_s.
            b_s = b.get("ttfb_s")
            c_s = c.get("ttfb_s")
        change = _pct_change(baseline=float(b_s), current=float(c_s)) if b_s is not None and c_s is not None else 0.0
        icon = _status_icon(change)
        change_str = f"{change:+.1f}% {icon}".rstrip()
        print(f"{key:30} {_fmt_s(b_s):>12} {_fmt_s(c_s):>12} {change_str:>10}")
    print()


def _print_tier3(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    if not keys:
        return

    print("Cross-Implementation (Tier 3):")
    print(f"{'case':30} {'baseline':>12} {'current':>12} {'change':>10}")
    for key in keys:
        b = baseline.get(key, {})
        c = current.get(key, {})
        b_s = b.get("wall_time_s")
        c_s = c.get("wall_time_s")
        change = _pct_change(baseline=float(b_s), current=float(c_s)) if b_s is not None and c_s is not None else 0.0
        icon = _status_icon(change)
        change_str = f"{change:+.1f}% {icon}".rstrip()
        print(f"{key:30} {_fmt_s(b_s):>12} {_fmt_s(c_s):>12} {change_str:>10}")
    print()


def _extract_blockwise_seconds(entry: Any) -> float | None:
    if not isinstance(entry, dict) or "skipped" in entry:
        return None

    for key in ("wall_time_s", "ttfb_s", "ttfb_audio_s"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    samples: list[float] = []
    points = entry.get("points")
    if isinstance(points, list):
        for point in points:
            if isinstance(point, dict) and isinstance(point.get("wall_time_s"), (int, float)):
                samples.append(float(point["wall_time_s"]))

    for value in entry.values():
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("wall_time_s"), (int, float)):
            samples.append(float(value["wall_time_s"]))
        configs = value.get("configs")
        if isinstance(configs, dict):
            for cfg in configs.values():
                if not isinstance(cfg, dict):
                    continue
                wall_s = cfg.get("wall_s", cfg.get("wall_time_s"))
                if isinstance(wall_s, (int, float)):
                    samples.append(float(wall_s))

    if not samples:
        return None
    return float(sum(samples) / float(len(samples)))


def _print_blockwise_all(baseline_tier2: dict[str, Any], current_tier2: dict[str, Any]) -> None:
    blockwise_keys = sorted(
        key
        for key in (set(baseline_tier2.keys()) | set(current_tier2.keys()))
        if key.startswith("bench_blockwise_")
    )
    if not blockwise_keys:
        return

    print("Blockwise Benchmarks (Tier 2):")
    print(f"{'benchmark':30} {'baseline':>12} {'current':>12} {'change':>10}")
    for key in blockwise_keys:
        baseline_value = _extract_blockwise_seconds(baseline_tier2.get(key, {}))
        current_value = _extract_blockwise_seconds(current_tier2.get(key, {}))
        change = (
            _pct_change(baseline=float(baseline_value), current=float(current_value))
            if baseline_value is not None and current_value is not None
            else 0.0
        )
        icon = _status_icon(change)
        change_str = f"{change:+.1f}% {icon}".rstrip()
        print(f"{key:30} {_fmt_s(baseline_value):>12} {_fmt_s(current_value):>12} {change_str:>10}")
    print()


def _print_blockwise_vs_standard(baseline_tier2: dict[str, Any], current_tier2: dict[str, Any]) -> None:
    baseline = baseline_tier2.get("bench_blockwise_vs_standard")
    current = current_tier2.get("bench_blockwise_vs_standard")
    if not isinstance(baseline, dict) or not isinstance(current, dict):
        return
    if "skipped" in baseline or "skipped" in current:
        return

    print("Blockwise vs Standard (Tier 2):")
    print(
        f"{'config':30} {'base ovh':>10} {'curr ovh':>10} {'ovh Δ':>10} "
        f"{'base ttfb':>10} {'curr ttfb':>10} {'ttfb Δ':>10}"
    )

    frame_keys = sorted(set(baseline.keys()) | set(current.keys()))
    for frame_key in frame_keys:
        b_cfgs = baseline.get(frame_key, {}).get("configs", {})
        c_cfgs = current.get(frame_key, {}).get("configs", {})
        config_keys = sorted(set(b_cfgs.keys()) | set(c_cfgs.keys()))
        for config_key in config_keys:
            b = b_cfgs.get(config_key, {})
            c = c_cfgs.get(config_key, {})
            b_overhead = b.get("overhead_ratio")
            c_overhead = c.get("overhead_ratio")
            b_speedup = b.get("ttfb_speedup")
            c_speedup = c.get("ttfb_speedup")

            overhead_delta = (
                _pct_change(baseline=float(b_overhead), current=float(c_overhead))
                if b_overhead is not None and c_overhead is not None
                else 0.0
            )
            speedup_delta = (
                _pct_change(baseline=float(b_speedup), current=float(c_speedup))
                if b_speedup is not None and c_speedup is not None
                else 0.0
            )
            icon = "❌" if overhead_delta > 10.0 else ("⚠️" if overhead_delta > 5.0 else "")
            overhead_delta_str = f"{overhead_delta:+.1f}% {icon}".rstrip()
            speedup_delta_str = f"{speedup_delta:+.1f}%"

            label = f"{frame_key}/{config_key}"
            print(
                f"{label:30} {_fmt_ratio(b_overhead):>10} {_fmt_ratio(c_overhead):>10} {overhead_delta_str:>10} "
                f"{_fmt_ratio(b_speedup):>10} {_fmt_ratio(c_speedup):>10} {speedup_delta_str:>10}"
            )
    print()


def _print_tier3_blockwise(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    baseline_blockwise = baseline.get("tier3_blockwise", {})
    current_blockwise = current.get("tier3_blockwise", {})
    if not isinstance(baseline_blockwise, dict) or not isinstance(current_blockwise, dict):
        return

    keys = sorted((set(baseline_blockwise.keys()) | set(current_blockwise.keys())) - {"skipped"})
    if not keys:
        return

    print("Cross-Implementation Blockwise (Tier 3):")
    print(f"{'case':30} {'baseline':>12} {'current':>12} {'change':>10}")
    for key in keys:
        b = baseline_blockwise.get(key, {})
        c = current_blockwise.get(key, {})
        b_s = b.get("wall_time_s", b.get("ttfb_audio_s")) if isinstance(b, dict) else None
        c_s = c.get("wall_time_s", c.get("ttfb_audio_s")) if isinstance(c, dict) else None
        change = _pct_change(baseline=float(b_s), current=float(c_s)) if b_s is not None and c_s is not None else 0.0
        icon = _status_icon(change)
        change_str = f"{change:+.1f}% {icon}".rstrip()
        print(f"{key:30} {_fmt_s(b_s):>12} {_fmt_s(c_s):>12} {change_str:>10}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark_results.json files")
    parser.add_argument("baseline", type=Path, help="Baseline JSON report")
    parser.add_argument("current", type=Path, help="Current JSON report")
    args = parser.parse_args(argv)

    base = _load(args.baseline)
    cur = _load(args.current)

    _print_tier1(base.get("tier1", {}), cur.get("tier1", {}))
    base_tier2 = base.get("tier2", {})
    cur_tier2 = cur.get("tier2", {})
    _print_tier2(base_tier2, cur_tier2)
    _print_blockwise_all(base_tier2, cur_tier2)
    _print_blockwise_vs_standard(base_tier2, cur_tier2)
    _print_tier3(base.get("tier3", {}), cur.get("tier3", {}))
    _print_tier3_blockwise(base, cur)

    print("Thresholds: ✅ >2% improvement, ⚠️ >5% regression, ❌ >10% regression")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
