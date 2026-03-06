from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compare import main


def test_compare_prints_tier3_table(tmp_path: Path, capsys) -> None:
    baseline = {
        "tier3": {
            "case_a": {"wall_time_s": 10.0},
            "case_b": {"wall_time_s": 20.0},
        }
    }
    current = {
        "tier3": {
            "case_a": {"wall_time_s": 9.5},
            "case_b": {"wall_time_s": 21.0},
        }
    }

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    rc = main([str(baseline_path), str(current_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Cross-Implementation (Tier 3):" in captured.out
    assert "case_a" in captured.out
    assert "case_b" in captured.out


def test_compare_prints_blockwise_vs_standard_section(tmp_path: Path, capsys) -> None:
    baseline = {
        "tier2": {
            "bench_blockwise_vs_standard": {
                "320_frames": {
                    "configs": {
                        "2_blocks": {"overhead_ratio": 1.90, "ttfb_speedup": 1.20},
                    }
                }
            }
        }
    }
    current = {
        "tier2": {
            "bench_blockwise_vs_standard": {
                "320_frames": {
                    "configs": {
                        "2_blocks": {"overhead_ratio": 2.20, "ttfb_speedup": 1.35},
                    }
                }
            }
        }
    }

    baseline_path = tmp_path / "baseline_blockwise.json"
    current_path = tmp_path / "current_blockwise.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    rc = main([str(baseline_path), str(current_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Blockwise vs Standard (Tier 2):" in captured.out
    assert "320_frames/2_blocks" in captured.out


def test_compare_finds_all_blockwise_keys(tmp_path: Path, capsys) -> None:
    baseline = {
        "tier2": {
            "bench_blockwise_breakdown": {
                "bw_single": {"wall_time_s": 1.0},
            },
            "bench_blockwise_scale_blocks": {
                "points": [{"num_blocks": 1, "wall_time_s": 1.2}],
            },
            "bench_blockwise_vs_standard": {
                "320_frames": {
                    "configs": {
                        "2_blocks": {"wall_s": 2.0, "overhead_ratio": 1.1, "ttfb_speedup": 1.2},
                    }
                }
            },
        }
    }
    current = {
        "tier2": {
            "bench_blockwise_breakdown": {
                "bw_single": {"wall_time_s": 1.1},
            },
            "bench_blockwise_scale_blocks": {
                "points": [{"num_blocks": 1, "wall_time_s": 1.3}],
            },
            "bench_blockwise_vs_standard": {
                "320_frames": {
                    "configs": {
                        "2_blocks": {"wall_s": 2.2, "overhead_ratio": 1.2, "ttfb_speedup": 1.3},
                    }
                }
            },
        }
    }

    baseline_path = tmp_path / "baseline_blockwise_all.json"
    current_path = tmp_path / "current_blockwise_all.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    rc = main([str(baseline_path), str(current_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Blockwise Benchmarks (Tier 2):" in captured.out
    assert "bench_blockwise_breakdown" in captured.out
    assert "bench_blockwise_scale_blocks" in captured.out
    assert "bench_blockwise_vs_standard" in captured.out


def test_compare_tier3_blockwise(tmp_path: Path, capsys) -> None:
    baseline = {
        "tier3_blockwise": {
            "case_a_bw": {"wall_time_s": 10.0, "ttfb_audio_s": 1.0},
            "case_b_bw": {"wall_time_s": 12.0, "ttfb_audio_s": 1.2},
        }
    }
    current = {
        "tier3_blockwise": {
            "case_a_bw": {"wall_time_s": 9.0, "ttfb_audio_s": 0.9},
            "case_b_bw": {"wall_time_s": 13.0, "ttfb_audio_s": 1.3},
        }
    }

    baseline_path = tmp_path / "baseline_t3_blockwise.json"
    current_path = tmp_path / "current_t3_blockwise.json"
    baseline_path.write_text(json.dumps(baseline))
    current_path.write_text(json.dumps(current))

    rc = main([str(baseline_path), str(current_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Cross-Implementation Blockwise (Tier 3):" in captured.out
    assert "case_a_bw" in captured.out
    assert "case_b_bw" in captured.out
